"""
Object Detection Module
=======================
Coordinates object detection across multiple strategies:
  1. YOLO (primary) - identifies objects by class name (best accuracy)
  2. Color-based HSV segmentation (fallback for when YOLO isn't available)
  3. Contour-based detection (fallback for dark objects on light backgrounds)

Design rationale:
- YOLO is primary because it can actually tell a phone from a mouse
  (which color/contour methods cannot do reliably).
- I kept color detection as fallback so the system still works even if
  the YOLO model file is missing or onnxruntime isn't installed.
- Each object in procedure.json specifies whether to use YOLO (yolo_class)
  or color detection (color_name + HSV range).

Author: Ruijie Zheng
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Import YOLO detector
try:
    from .yolo_detector import YOLODetector, DISPLAY_NAMES, CLASS_COLORS, CLASS_ALIASES, COCO_CLASSES
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False


class ObjectStatus(Enum):
    """Status of a tracked object."""
    NOT_DETECTED = "not_detected"
    DETECTED = "detected"
    BEING_MOVED = "being_moved"
    IN_TARGET = "in_target"


@dataclass
class DetectedObject:
    """Represents a detected object with its properties."""
    object_id: str
    name: str
    
    # Position (screen coordinates)
    center: Tuple[int, int] = (0, 0)
    bounding_box: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x, y, w, h
    
    # Position (normalized table coordinates)
    table_position: Optional[Tuple[float, float]] = None
    
    # Status
    status: ObjectStatus = ObjectStatus.NOT_DETECTED
    confidence: float = 0.0
    
    # Tracking
    visible: bool = False
    frames_since_seen: int = 0
    
    # Color info for visualization
    color_bgr: Tuple[int, int, int] = (255, 255, 255)


@dataclass
class ObjectConfig:
    """Configuration for detecting a specific object."""
    object_id: str
    name: str
    color_lower: np.ndarray  # HSV lower bound
    color_upper: np.ndarray  # HSV upper bound
    color_name: str
    min_area: int = 3000
    color_bgr: Tuple[int, int, int] = field(default_factory=lambda: (255, 255, 255))
    # YOLO class name to match (e.g., "cell phone", "bottle")
    # leave empty for color-only detection
    yolo_class: str = ""


class ObjectDetector:
    """Detects objects using YOLO first, falls back to color/contour if needed."""
    
    def __init__(self, objects_config: List[Dict] = None):
        """
        Initialize object detector.
        
        Args:
            objects_config: List of object configurations from JSON
        """
        self.object_configs: Dict[str, ObjectConfig] = {}
        self.detected_objects: Dict[str, DetectedObject] = {}
        
        # Initialize YOLO detector
        self.yolo: Optional[YOLODetector] = None
        if YOLO_AVAILABLE:
            # Slightly lower than default 0.35 for better phone/bottle detection
            self.yolo = YOLODetector(confidence_threshold=0.30)
            if not self.yolo.is_available:
                self.yolo = None
                
        # Cache YOLO detections to avoid running twice per frame
        self._last_yolo_detections: List[Dict] = []
        
        # Frame skip for YOLO -- running inference every single frame is way too
        # slow on CPU (~200ms). Skipping 2 frames means we still get ~10 updates/sec
        # at 30fps which is plenty for tracking objects that don't move fast.
        self._frame_count = 0
        self._yolo_skip_frames = 2
        
        # General detection results (all YOLO-detected objects)
        self.general_objects: List[DetectedObject] = []
        
        # Color mapping for visualization
        self.color_map = {
            'red': (0, 0, 255),
            'blue': (255, 0, 0),
            'green': (0, 255, 0),
            'yellow': (0, 255, 255),
            'orange': (0, 165, 255),
            'purple': (128, 0, 128),
            'white': (255, 255, 255),
            'dark': (128, 128, 128),
            'any': (200, 200, 0),
        }
        
        if objects_config:
            self.load_config(objects_config)
            
    def load_config(self, objects_config: List[Dict]):
        """Load object configurations from config list."""
        for obj in objects_config:
            config = ObjectConfig(
                object_id=obj['id'],
                name=obj['name'],
                color_lower=np.array(obj.get('color_lower', [0, 0, 0])),
                color_upper=np.array(obj.get('color_upper', [180, 255, 255])),
                color_name=obj.get('color_name', 'yolo'),
                min_area=obj.get('min_area', 3000),
                color_bgr=self.color_map.get(obj.get('color_name', 'white'), (255, 255, 255)),
                yolo_class=obj.get('yolo_class', ''),
            )
            # If yolo_class not specified, infer from name
            if not config.yolo_class:
                name_lower = config.name.lower()
                # Map common names to COCO class names
                name_to_coco = {
                    'phone': 'cell phone', 'cellphone': 'cell phone',
                    'mobile': 'cell phone', 'smartphone': 'cell phone',
                    'mouse': 'mouse', 'cup': 'cup', 'mug': 'cup',
                    'bottle': 'bottle', 'laptop': 'laptop',
                    'keyboard': 'keyboard', 'book': 'book',
                    'remote': 'remote', 'scissors': 'scissors',
                    'monitor': 'tv', 'screen': 'tv',
                    'bowl': 'bowl', 'glass': 'wine glass',
                    'plant': 'potted plant', 'plate': 'bowl',
                    'pen': 'pen', 'pencil': 'pencil', 'marker': 'marker',
                    'eraser': 'eraser', 'cards': 'cards', 'card': 'card',
                    'wallet': 'wallet', 'notebook': 'notebook',
                    'stapler': 'stapler', 'ruler': 'ruler',
                    'calculator': 'calculator', 'headphones': 'headphones',
                }
                config.yolo_class = name_to_coco.get(name_lower, name_lower)
            
            # Get color from YOLO class colors if available
            if YOLO_AVAILABLE and config.yolo_class in CLASS_COLORS:
                config.color_bgr = CLASS_COLORS[config.yolo_class]
            
            self.object_configs[config.object_id] = config
            
            # Initialize detected object
            self.detected_objects[config.object_id] = DetectedObject(
                object_id=config.object_id,
                name=config.name,
                color_bgr=config.color_bgr
            )
    
    def detect_objects(self, frame: np.ndarray, 
                       marker_detector=None,
                       marker_rects: list = None,
                       hand_position: tuple = None) -> Dict[str, DetectedObject]:
        """Main detection loop -- tries YOLO, then color, then contour for each object."""
        if marker_rects is None:
            marker_rects = []
        
        # Run YOLO detection (with frame skipping for performance)
        if self.yolo is not None:
            self._frame_count += 1
            if self._frame_count > self._yolo_skip_frames:
                raw_dets = self.yolo.detect(frame)
                # Filter out detections that overlap with ArUco markers
                self._last_yolo_detections = self._filter_marker_overlaps(
                    raw_dets, marker_rects)
                self._frame_count = 0
            # else: reuse cached _last_yolo_detections
        else:
            self._last_yolo_detections = []
        
        # Only compute HSV if we actually need color-based detection
        # (avoids wasted work when YOLO found everything)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) if not self._last_yolo_detections else None
        
        for obj_id, config in self.object_configs.items():
            detected_obj = None
            
            # Try YOLO detection first
            if self._last_yolo_detections and config.yolo_class:
                detected_obj = self._detect_by_yolo(config, marker_detector,
                                                    hand_position=hand_position)
            
            # Fallback to color/contour detection if YOLO didn't find it
            if detected_obj is None or not detected_obj.visible:
                if hsv is None:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Pen/pencil/marker: use dedicated elongated-object detector
                if config.name.lower() in ('pen', 'pencil', 'marker'):
                    detected_obj = self._detect_pen(frame, config, marker_detector,
                                                    marker_rects=marker_rects)
                elif config.color_name in ('any', 'dark'):
                    detected_obj = self._detect_by_contour(frame, config, marker_detector,
                                                           marker_rects=marker_rects)
                elif config.color_name != 'yolo':
                    detected_obj = self._detect_single_object(hsv, frame, config, marker_detector)
                else:
                    # yolo-only mode, no fallback
                    detected_obj = DetectedObject(
                        object_id=config.object_id, name=config.name,
                        color_bgr=config.color_bgr
                    )
            
            # Update tracking
            if detected_obj and detected_obj.visible:
                self.detected_objects[obj_id] = detected_obj
                self.detected_objects[obj_id].frames_since_seen = 0
            else:
                self.detected_objects[obj_id].frames_since_seen += 1
                self.detected_objects[obj_id].visible = False
                if self.detected_objects[obj_id].frames_since_seen > 30:
                    self.detected_objects[obj_id].status = ObjectStatus.NOT_DETECTED
        
        # Build general objects list from ALL YOLO detections
        self.general_objects = self._build_general_objects(marker_detector)
                    
        return self.detected_objects
    
    @staticmethod
    def _filter_marker_overlaps(detections: list, marker_rects: list) -> list:
        """Remove YOLO detections that overlap with ArUco marker areas.
        
        Markers should NEVER be recognized as objects -- this is a hard rule.
        """
        if not marker_rects:
            return detections
        
        filtered = []
        for det in detections:
            x, y, w, h = det['bbox']
            overlaps_marker = False
            for mx, my, mw, mh in marker_rects:
                # Intersection
                ox = max(0, min(x + w, mx + mw) - max(x, mx))
                oy = max(0, min(y + h, my + mh) - max(y, my))
                overlap_area = ox * oy
                det_area = max(w * h, 1)
                marker_area = max(mw * mh, 1)
                # Skip if overlap is significant relative to either area
                if overlap_area > 0.25 * marker_area or overlap_area > 0.25 * det_area:
                    overlaps_marker = True
                    break
            if not overlaps_marker:
                filtered.append(det)
        return filtered
    
    def _detect_by_yolo(self, config: ObjectConfig,
                         marker_detector=None,
                         hand_position: tuple = None) -> Optional[DetectedObject]:
        """Match a configured object against YOLO results by class name.
        
        When the hand is detected, prioritize detections near the hand.
        Uses CLASS_ALIASES for non-COCO objects (pen->knife/toothbrush, etc.)
        """
        target_class = config.yolo_class.lower()
        best_match = None
        
        # Build list of COCO classes to search for
        # If target_class is a direct COCO class, search for it.
        # If not, use aliases (pen -> knife/toothbrush, eraser -> book/remote, etc.)
        search_classes = []
        if YOLO_AVAILABLE and target_class in [c.lower() for c in COCO_CLASSES]:
            search_classes = [target_class]
        elif YOLO_AVAILABLE and target_class in CLASS_ALIASES:
            search_classes = [c.lower() for c in CLASS_ALIASES[target_class]]
        else:
            search_classes = [target_class]
        
        # Collect all candidates matching any of the search classes
        candidates = [det for det in self._last_yolo_detections
                      if det['class_name'].lower() in search_classes]
        
        if not candidates:
            return None
        
        if hand_position and len(candidates) > 1:
            # Prefer the detection closest to the hand
            hx, hy = hand_position
            candidates.sort(key=lambda d: (
                (d['center'][0] - hx) ** 2 + (d['center'][1] - hy) ** 2
            ))
            best_match = candidates[0]
        else:
            # Pick highest confidence
            for det in candidates:
                if best_match is None or det['confidence'] > best_match['confidence']:
                    best_match = det
        
        if best_match is None:
            return None
        
        x, y, w, h = best_match['bbox']
        cx, cy = best_match['center']
        
        detected = DetectedObject(
            object_id=config.object_id,
            name=config.name,
            center=(cx, cy),
            bounding_box=(x, y, w, h),
            visible=True,
            status=ObjectStatus.DETECTED,
            confidence=best_match['confidence'],
            color_bgr=config.color_bgr,
        )
        
        # Convert to table coordinates if marker detector available
        if marker_detector and marker_detector.is_calibrated():
            table_pos = marker_detector.screen_to_table((cx, cy))
            if table_pos:
                detected.table_position = table_pos
        
        return detected

    def _build_general_objects(self, marker_detector=None) -> List[DetectedObject]:
        """
        Build general objects list from ALL YOLO detections.
        These are shown to the user so they can see what the system recognizes.
        """
        results = []
        
        for i, det in enumerate(self._last_yolo_detections):
            x, y, w, h = det['bbox']
            cx, cy = det['center']
            
            obj = DetectedObject(
                object_id=f"yolo_{i}",
                name=det['display_name'],
                center=(cx, cy),
                bounding_box=(x, y, w, h),
                visible=True,
                status=ObjectStatus.DETECTED,
                confidence=det['confidence'],
                color_bgr=det['color_bgr'],
            )
            
            if marker_detector and marker_detector.is_calibrated():
                table_pos = marker_detector.screen_to_table((cx, cy))
                if table_pos:
                    obj.table_position = table_pos
            
            results.append(obj)
        
        return results
    
    def _detect_single_object(self, hsv: np.ndarray, frame: np.ndarray,
                               config: ObjectConfig, 
                               marker_detector=None) -> DetectedObject:
        """Detect a single object by color."""
        
        # Handle red color wrap-around in HSV
        # (red sits at both ends of the hue wheel, 0 and 180, so need two ranges)
        if config.color_name == 'red':
            # Red wraps around 0/180 in HSV
            mask1 = cv2.inRange(hsv, config.color_lower, config.color_upper)
            # Also check upper red range
            lower2 = np.array([170, 100, 100])
            upper2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, config.color_lower, config.color_upper)
        
        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = DetectedObject(
            object_id=config.object_id,
            name=config.name,
            color_bgr=config.color_bgr
        )
        
        if contours:
            # Find largest contour
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            
            if area >= config.min_area:
                # Get bounding box
                x, y, w, h = cv2.boundingRect(largest)
                
                # Get center using moments
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2
                
                detected.center = (cx, cy)
                detected.bounding_box = (x, y, w, h)
                detected.visible = True
                detected.status = ObjectStatus.DETECTED
                detected.confidence = min(area / (config.min_area * 3), 1.0)
                
                # Convert to table coordinates if marker detector available
                if marker_detector and marker_detector.is_calibrated():
                    table_pos = marker_detector.screen_to_table((cx, cy))
                    if table_pos:
                        detected.table_position = table_pos
                        
        return detected
    
    @staticmethod
    def _contour_overlaps_marker(cx, cy, bx, by, bw, bh, marker_rects: list) -> bool:
        """Return True if a contour bbox significantly overlaps any marker rect."""
        if not marker_rects:
            return False
        for mx, my, mw, mh in marker_rects:
            # Expand marker rect by 20% to catch edge contours
            pad_x, pad_y = int(mw * 0.2), int(mh * 0.2)
            ex, ey = mx - pad_x, my - pad_y
            ew, eh = mw + 2 * pad_x, mh + 2 * pad_y
            # Check if contour center is inside expanded marker
            if ex <= cx <= ex + ew and ey <= cy <= ey + eh:
                return True
            # Check bbox overlap
            ox = max(0, min(bx + bw, mx + mw) - max(bx, mx))
            oy = max(0, min(by + bh, my + mh) - max(by, my))
            overlap_area = ox * oy
            det_area = max(bw * bh, 1)
            if overlap_area > 0.25 * det_area:
                return True
        return False

    def _detect_by_contour(self, frame: np.ndarray, config: ObjectConfig,
                            marker_detector=None,
                            marker_rects: list = None) -> DetectedObject:
        """
        Detect an object using edge/contour detection (for dark or any-color objects).
        Works for phones, mice, keyboards, etc.
        """
        if marker_rects is None:
            marker_rects = []
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        
        # Use adaptive thresholding to find distinct objects
        # This catches dark objects on light backgrounds AND light objects on dark backgrounds
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 25, 8)
        
        # Edge-based approach as supplement
        edges = cv2.Canny(blurred, 30, 100)
        combined = cv2.bitwise_or(thresh, edges)
        
        # Clean up
        kernel = np.ones((7, 7), np.uint8)
        combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
        combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel, iterations=1)
        
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detected = DetectedObject(
            object_id=config.object_id, name=config.name, color_bgr=config.color_bgr
        )
        
        if contours:
            # Filter by area and aspect ratio
            valid_contours = []
            for c in contours:
                area = cv2.contourArea(c)
                if area >= config.min_area:
                    x, y, w, h = cv2.boundingRect(c)
                    aspect = max(w, h) / (min(w, h) + 1)
                    # Skip contours overlapping ArUco markers
                    cx_c, cy_c = x + w // 2, y + h // 2
                    if self._contour_overlaps_marker(cx_c, cy_c, x, y, w, h, marker_rects):
                        continue
                    # Filter out very elongated shapes (likely edges of table/screen)
                    if aspect < 6:
                        valid_contours.append((c, area))
            
            if valid_contours:
                # Pick largest valid contour
                largest, area = max(valid_contours, key=lambda x: x[1])
                x, y, w, h = cv2.boundingRect(largest)
                M = cv2.moments(largest)
                if M["m00"] > 0:
                    cx = int(M["m10"] / M["m00"])
                    cy = int(M["m01"] / M["m00"])
                else:
                    cx, cy = x + w // 2, y + h // 2
                
                detected.center = (cx, cy)
                detected.bounding_box = (x, y, w, h)
                detected.visible = True
                detected.status = ObjectStatus.DETECTED
                detected.confidence = min(area / (config.min_area * 3), 1.0)
                
                if marker_detector and marker_detector.is_calibrated():
                    table_pos = marker_detector.screen_to_table((cx, cy))
                    if table_pos:
                        detected.table_position = table_pos
        
        return detected
    
    def _detect_pen(self, frame: np.ndarray, config: ObjectConfig,
                     marker_detector=None,
                     marker_rects: list = None) -> DetectedObject:
        """Detect a pen/pencil/marker using shape analysis.
        
        A pen is thin and elongated -- exactly the shape that generic contour
        detection rejects. This method specifically looks for high-aspect-ratio
        dark objects using multiple techniques.
        Marker areas are excluded so printed ArUco papers are never mistaken
        for a pen.
        """
        if marker_rects is None:
            marker_rects = []
        h_frame, w_frame = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        detected = DetectedObject(
            object_id=config.object_id, name=config.name, color_bgr=config.color_bgr
        )
        
        candidates = []
        
        # Strategy 1: Adaptive threshold (dark objects on lighter background)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                        cv2.THRESH_BINARY_INV, 15, 6)
        # Use small kernel to preserve thin shapes
        kernel_small = np.ones((3, 3), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small, iterations=1)
        contours1, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Strategy 2: Edge detection (catches pen outlines)
        edges = cv2.Canny(blurred, 40, 120)
        kernel_edge = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel_edge, iterations=2)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel_edge, iterations=1)
        contours2, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        all_contours = list(contours1) + list(contours2)
        
        for c in all_contours:
            area = cv2.contourArea(c)
            if area < 300:  # too small
                continue
            
            x, y, w, bh = cv2.boundingRect(c)
            cx_c, cy_c = x + w // 2, y + bh // 2
            
            # === Hard rule: skip anything overlapping a calibration marker ===
            if self._contour_overlaps_marker(cx_c, cy_c, x, y, w, bh, marker_rects):
                continue
            
            aspect = max(w, bh) / (min(w, bh) + 1)
            
            # Pen characteristics: elongated (aspect > 2.5) OR medium-sized dark blob
            # Also accept a rotated bounding box for diagonal pens
            if len(c) >= 5:
                rect = cv2.minAreaRect(c)
                rw, rh = rect[1]
                rot_aspect = max(rw, rh) / (min(rw, rh) + 1)
            else:
                rot_aspect = aspect
            
            # Score: prefer elongated, pen-sized objects
            is_elongated = rot_aspect >= 2.5 or aspect >= 2.5
            is_pen_sized = 300 <= area <= 50000  # not too big (not a table edge)
            is_not_huge = w < w_frame * 0.5 and bh < h_frame * 0.5
            
            if is_pen_sized and is_not_huge:
                # Higher score for more pen-like shapes
                score = area
                if is_elongated:
                    score *= 3.0  # strongly prefer elongated
                if rot_aspect >= 4:
                    score *= 2.0  # very pen-like aspect ratio
                candidates.append((c, score, area))
        
        if candidates:
            # Pick best scoring candidate
            best_contour, best_score, best_area = max(candidates, key=lambda x: x[1])
            x, y, w, bh = cv2.boundingRect(best_contour)
            M = cv2.moments(best_contour)
            if M["m00"] > 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
            else:
                cx, cy = x + w // 2, y + bh // 2
            
            detected.center = (cx, cy)
            detected.bounding_box = (x, y, w, bh)
            detected.visible = True
            detected.status = ObjectStatus.DETECTED
            detected.confidence = min(best_area / 2000, 1.0)
            
            if marker_detector and marker_detector.is_calibrated():
                table_pos = marker_detector.screen_to_table((cx, cy))
                if table_pos:
                    detected.table_position = table_pos
        
        return detected

    def get_object(self, object_id: str) -> Optional[DetectedObject]:
        """Get a specific detected object by ID."""
        return self.detected_objects.get(object_id)
    
    def is_object_visible(self, object_id: str) -> bool:
        """Check if an object is currently visible."""
        obj = self.detected_objects.get(object_id)
        return obj is not None and obj.visible
    
    def get_all_visible_objects(self) -> List[DetectedObject]:
        """Get list of all currently visible objects."""
        return [obj for obj in self.detected_objects.values() if obj.visible]
    
    def draw_debug(self, frame: np.ndarray, show_mask: bool = False) -> np.ndarray:
        """
        Draw debug visualization of detected objects.
        
        Args:
            frame: BGR image to draw on
            show_mask: Whether to overlay detection masks
            
        Returns:
            Frame with debug visualization
        """
        output = frame.copy()
        
        for obj_id, obj in self.detected_objects.items():
            if obj.visible:
                x, y, w, h = obj.bounding_box
                color = obj.color_bgr
                
                # Draw bounding box
                cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)
                
                # Draw center point
                cv2.circle(output, obj.center, 5, color, -1)
                
                # Draw label
                label = f"{obj.name}"
                if obj.table_position:
                    label += f" ({obj.table_position[0]:.2f}, {obj.table_position[1]:.2f})"
                    
                cv2.putText(output, label, (x, y - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                           
        return output
    
    def get_color_calibration_mask(self, frame: np.ndarray, 
                                    object_id: str) -> Optional[np.ndarray]:
        """Get the detection mask for a specific object (for calibration)."""
        if object_id not in self.object_configs:
            return None
            
        config = self.object_configs[object_id]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        if config.color_name == 'red':
            mask1 = cv2.inRange(hsv, config.color_lower, config.color_upper)
            lower2 = np.array([170, 100, 100])
            upper2 = np.array([180, 255, 255])
            mask2 = cv2.inRange(hsv, lower2, upper2)
            mask = cv2.bitwise_or(mask1, mask2)
        else:
            mask = cv2.inRange(hsv, config.color_lower, config.color_upper)
            
        return mask


class ColorCalibrator:
    """
    Helper class for calibrating object colors interactively.
    Useful for adjusting HSV ranges to match actual objects.
    """
    
    def __init__(self):
        self.window_name = "Color Calibrator"
        self.h_low, self.s_low, self.v_low = 0, 100, 100
        self.h_high, self.s_high, self.v_high = 180, 255, 255
        
    def create_trackbars(self):
        """Create calibration window with trackbars."""
        cv2.namedWindow(self.window_name)
        cv2.createTrackbar("H Low", self.window_name, self.h_low, 180, lambda x: None)
        cv2.createTrackbar("H High", self.window_name, self.h_high, 180, lambda x: None)
        cv2.createTrackbar("S Low", self.window_name, self.s_low, 255, lambda x: None)
        cv2.createTrackbar("S High", self.window_name, self.s_high, 255, lambda x: None)
        cv2.createTrackbar("V Low", self.window_name, self.v_low, 255, lambda x: None)
        cv2.createTrackbar("V High", self.window_name, self.v_high, 255, lambda x: None)
        
    def get_values(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current trackbar values as HSV bounds."""
        h_low = cv2.getTrackbarPos("H Low", self.window_name)
        h_high = cv2.getTrackbarPos("H High", self.window_name)
        s_low = cv2.getTrackbarPos("S Low", self.window_name)
        s_high = cv2.getTrackbarPos("S High", self.window_name)
        v_low = cv2.getTrackbarPos("V Low", self.window_name)
        v_high = cv2.getTrackbarPos("V High", self.window_name)
        
        lower = np.array([h_low, s_low, v_low])
        upper = np.array([h_high, s_high, v_high])
        return lower, upper
    
    def show_mask(self, frame: np.ndarray):
        """Show the mask based on current trackbar values."""
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower, upper = self.get_values()
        mask = cv2.inRange(hsv, lower, upper)
        
        # Show both mask and result
        result = cv2.bitwise_and(frame, frame, mask=mask)
        combined = np.hstack([frame, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result])
        
        # Scale down if too wide
        h, w = combined.shape[:2]
        if w > 1920:
            scale = 1920 / w
            combined = cv2.resize(combined, (int(w * scale), int(h * scale)))
            
        cv2.imshow(self.window_name, combined)
        
    def print_values(self):
        """Print current HSV values in config format."""
        lower, upper = self.get_values()
        print(f'"color_lower": {lower.tolist()},')
        print(f'"color_upper": {upper.tolist()},')


if __name__ == "__main__":
    # Run color calibration tool
    print("Color Calibration Tool")
    print("Press 'p' to print current values, 'q' to quit")
    
    calibrator = ColorCalibrator()
    calibrator.create_trackbars()
    
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        calibrator.show_mask(frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            calibrator.print_values()
            
    cap.release()
    cv2.destroyAllWindows()
