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
    from .yolo_detector import YOLODetector, DISPLAY_NAMES, CLASS_COLORS
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
    # YOLO class name to match (e.g., "cell phone", "cup", "mouse")
    yolo_class: str = ""


class ObjectDetector:
    """
    Detects objects using YOLO (primary) or color-based segmentation (fallback).
    
    Detection priority:
      1. YOLO model identifies objects by class name (cell phone, mouse, cup, etc.)
      2. Color-based HSV detection (when color_name is set and YOLO not available)
      3. Contour-based detection (for 'dark'/'any' color_name)
    """
    
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
            self.yolo = YOLODetector(confidence_threshold=0.3)
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
                       marker_detector=None) -> Dict[str, DetectedObject]:
        """
        Detect all configured objects in the frame.
        Uses YOLO first (if available), falls back to color/contour detection.
        
        Args:
            frame: BGR image
            marker_detector: Optional MarkerDetector for coordinate transformation
            
        Returns:
            Dictionary of detected objects
        """
        # Run YOLO detection (with frame skipping for performance)
        if self.yolo is not None:
            self._frame_count += 1
            if self._frame_count > self._yolo_skip_frames:
                self._last_yolo_detections = self.yolo.detect(frame)
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
                detected_obj = self._detect_by_yolo(config, marker_detector)
            
            # Fallback to color/contour detection if YOLO didn't find it
            if detected_obj is None or not detected_obj.visible:
                if hsv is None:
                    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                    
                if config.color_name in ('any', 'dark'):
                    detected_obj = self._detect_by_contour(frame, config, marker_detector)
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
    
    def _detect_by_yolo(self, config: ObjectConfig,
                         marker_detector=None) -> Optional[DetectedObject]:
        """
        Find a configured object in the YOLO detection results.
        
        Args:
            config: Object configuration with yolo_class set
            marker_detector: Optional for coordinate transformation
            
        Returns:
            DetectedObject if found, None otherwise
        """
        # Search through YOLO detections for matching class
        target_class = config.yolo_class.lower()
        best_match = None
        
        for det in self._last_yolo_detections:
            if det['class_name'].lower() == target_class:
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
    
    def _detect_by_contour(self, frame: np.ndarray, config: ObjectConfig,
                            marker_detector=None) -> DetectedObject:
        """
        Detect an object using edge/contour detection (for dark or any-color objects).
        Works for phones, mice, keyboards, etc.
        """
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
