"""
Marker Detection Module
=======================
Detects ArUco markers at the four table corners and computes homography.

Design choices:
- Using ArUco 4x4_50 dictionary -- small markers, fast to detect, and the
  assignment recommends fiducial markers for defining a coordinate system.
- The homography maps any screen pixel to normalized table coords (0-1),
  which is how we check if an object reached its target zone regardless
  of camera angle or distance.
- If markers aren't available (user presses SPACE to skip), the system
  falls back to screen-relative coordinates (camera frame = workspace).

Corner marker IDs (print from markers/ folder):
- ID 0: Top-Left
- ID 1: Top-Right
- ID 2: Bottom-Right
- ID 3: Bottom-Left

Author: Ruijie Zheng
"""

import cv2
import cv2.aruco as aruco
import numpy as np
from typing import Optional, Tuple, Dict, List


class MarkerDetector:
    """Detects ArUco markers and computes table homography for coordinate transformation."""
    
    # Corner marker IDs (place these at table corners)
    CORNER_IDS = {
        'top_left': 0,
        'top_right': 1,
        'bottom_right': 2,
        'bottom_left': 3
    }
    
    def __init__(self, dictionary_type: int = aruco.DICT_4X4_50):
        """
        Initialize the marker detector.
        
        Args:
            dictionary_type: ArUco dictionary type (default: 4x4_50 for robustness)
        """
        self.dictionary = aruco.getPredefinedDictionary(dictionary_type)
        self.parameters = aruco.DetectorParameters()
        
        # Tune detection for phone-displayed markers (screen glare, moiré, lower contrast)
        # Smaller adaptive window helps with uneven screen lighting
        self.parameters.adaptiveThreshWinSizeMin = 3
        self.parameters.adaptiveThreshWinSizeMax = 23
        self.parameters.adaptiveThreshWinSizeStep = 4
        # Lower threshold to catch lower-contrast markers on screens
        self.parameters.adaptiveThreshConstant = 7
        # Relax corner quality requirements for screen-captured markers
        self.parameters.minMarkerPerimeterRate = 0.02
        self.parameters.maxMarkerPerimeterRate = 4.0
        self.parameters.polygonalApproxAccuracyRate = 0.05
        # More permissive bit extraction for screen reflections
        self.parameters.perspectiveRemoveIgnoredMarginPerCell = 0.2
        self.parameters.maxErroneousBitsInBorderRate = 0.5
        self.parameters.errorCorrectionRate = 0.6
        # Corner refinement for better homography accuracy
        self.parameters.cornerRefinementMethod = aruco.CORNER_REFINE_SUBPIX
        self.parameters.cornerRefinementWinSize = 5
        self.parameters.cornerRefinementMaxIterations = 30
        self.parameters.cornerRefinementMinAccuracy = 0.1
        
        self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)
        
        # Homography matrix (screen -> table normalized coords)
        self.homography: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        
        # Detected corner positions in screen coordinates
        self.corner_positions: Dict[str, np.ndarray] = {}
        
        # Table boundary polygon (screen coordinates)
        self.table_boundary: Optional[np.ndarray] = None
        
        # Debug info from last detection
        self._last_corners = []
        self._last_ids = None
        self._last_rejected = []
        
        # Virtual table dimensions (normalized 0-1)
        self.table_width = 1.0
        self.table_height = 1.0
    
    def reset(self):
        """Clear calibration so the user can re-define the working region."""
        self.homography = None
        self.inverse_homography = None
        self.corner_positions = {}
        self.table_boundary = None
        # Debug info from last detection
        self._last_corners = []
        self._last_ids = None
        self._last_rejected = []
        
    def detect_markers(self, frame: np.ndarray) -> Tuple[List, List, List]:
        """Detect ArUco markers, handling possible horizontal flip.
        
        Tries detection on both the raw frame AND a horizontally-flipped
        copy, since we don't know if the camera or code has already
        mirrored the image.  Picks whichever orientation finds more
        markers and remaps coordinates to the input frame.
        """
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_flipped = cv2.flip(gray, 1)
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        
        # Try all 4 combos: (original | flipped) × (raw | CLAHE)
        attempts = [
            (gray,                  False),  # as-is, raw
            (clahe.apply(gray),     False),  # as-is, CLAHE
            (gray_flipped,          True),   # flipped, raw
            (clahe.apply(gray_flipped), True),  # flipped, CLAHE
        ]
        
        best_corners, best_ids, best_rejected = [], None, []
        best_count = 0
        best_was_flipped = False
        
        for img, was_flipped in attempts:
            c, i, r = self.detector.detectMarkers(img)
            count = 0 if i is None else len(i)
            if count > best_count:
                best_corners, best_ids, best_rejected = c, i, r
                best_count = count
                best_was_flipped = was_flipped
        
        corners, ids, rejected = best_corners, best_ids, best_rejected
        
        # If we detected on the flipped image, remap coords back
        if best_was_flipped and corners is not None and len(corners) > 0:
            corners = self._flip_corner_coords(corners, w)
        if best_was_flipped and rejected is not None and len(rejected) > 0:
            rejected = self._flip_corner_coords(rejected, w)
        
        # Store for debug drawing
        self._last_corners = corners if corners is not None else []
        self._last_ids = ids
        self._last_rejected = rejected if rejected is not None else []
        
        return corners, ids, rejected
    
    @staticmethod
    def _flip_corner_coords(corner_list, frame_width: int):
        """Mirror x-coordinates of detected corners back to the original frame."""
        flipped = []
        for c in corner_list:
            fc = c.copy()
            fc[:, :, 0] = frame_width - 1 - fc[:, :, 0]
            flipped.append(fc)
        return flipped
        self._last_rejected = rejected
        
        return corners, ids, rejected
    
    def update_table_boundary(self, frame: np.ndarray) -> bool:
        """Look for 4 corner markers and recompute homography if all found."""
        corners, ids, _ = self.detect_markers(frame)
        
        if ids is None:
            return False
            
        ids = ids.flatten()
        
        # Find corner markers
        detected_corners = {}
        for corner_name, marker_id in self.CORNER_IDS.items():
            if marker_id in ids:
                idx = list(ids).index(marker_id)
                # Get center of marker
                marker_corners = corners[idx][0]
                center = marker_corners.mean(axis=0)
                detected_corners[corner_name] = center
                
        self.corner_positions = detected_corners
        
        # Need all 4 corners to compute homography
        if len(detected_corners) == 4:
            self._compute_homography()
            return True
            
        return False
    
    def _compute_homography(self):
        """Compute homography matrix from detected corners."""
        # Source points (screen coordinates)
        src_points = np.float32([
            self.corner_positions['top_left'],
            self.corner_positions['top_right'],
            self.corner_positions['bottom_right'],
            self.corner_positions['bottom_left']
        ])
        
        # Destination points (normalized table coordinates 0-1)
        dst_points = np.float32([
            [0, 0],
            [self.table_width, 0],
            [self.table_width, self.table_height],
            [0, self.table_height]
        ])
        
        # Compute homography (screen -> table)
        self.homography, _ = cv2.findHomography(src_points, dst_points)
        
        # Inverse homography (table -> screen)
        self.inverse_homography, _ = cv2.findHomography(dst_points, src_points)
        
        # Store boundary polygon
        self.table_boundary = src_points.reshape((-1, 1, 2)).astype(np.int32)
        
    def screen_to_table(self, screen_point: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Screen px -> normalized table coords (0-1). None if not calibrated."""
        if self.homography is None:
            return None
            
        point = np.array([[screen_point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.homography)
        return tuple(transformed[0][0])
    
    def table_to_screen(self, table_point: Tuple[float, float]) -> Optional[Tuple[int, int]]:
        """Normalized table coords -> screen px. None if not calibrated."""
        if self.inverse_homography is None:
            return None
            
        point = np.array([[table_point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(point, self.inverse_homography)
        return tuple(map(int, transformed[0][0]))
    
    def is_point_on_table(self, screen_point: Tuple[float, float]) -> bool:
        """
        Check if a screen point is within the table boundary.
        
        Args:
            screen_point: (x, y) in screen coordinates
            
        Returns:
            True if point is inside table boundary
        """
        if self.table_boundary is None:
            return False
            
        result = cv2.pointPolygonTest(self.table_boundary, screen_point, False)
        return result >= 0
    
    def get_boundary_corners(self) -> Optional[np.ndarray]:
        """Get table boundary as polygon corners for visualization."""
        return self.table_boundary
    
    def is_calibrated(self) -> bool:
        """Check if table is properly calibrated with all markers."""
        return self.homography is not None and len(self.corner_positions) == 4
    
    def get_missing_markers(self) -> List[str]:
        """Get list of marker corner names that weren't detected."""
        missing = []
        for corner_name in self.CORNER_IDS.keys():
            if corner_name not in self.corner_positions:
                missing.append(corner_name)
        return missing
    
    def draw_debug(self, frame: np.ndarray) -> np.ndarray:
        """
        Draw debug visualization of detected markers and boundary.
        
        Args:
            frame: BGR image to draw on
            
        Returns:
            Frame with debug visualization
        """
        output = frame.copy()
        
        # Detect and draw markers
        corners, ids, _ = self.detect_markers(frame)
        if ids is not None:
            aruco.drawDetectedMarkers(output, corners, ids)
            
        # Draw table boundary
        if self.table_boundary is not None:
            cv2.polylines(output, [self.table_boundary], True, (0, 255, 0), 2)
            
            # Draw corner labels
            corner_names = ['TL', 'TR', 'BR', 'BL']
            for i, name in enumerate(corner_names):
                pt = tuple(self.table_boundary[i][0])
                cv2.putText(output, name, pt, cv2.FONT_HERSHEY_SIMPLEX, 
                           0.6, (255, 255, 0), 2)
                           
        return output
    
    def draw_calibration_feedback(self, frame: np.ndarray) -> np.ndarray:
        """Draw live marker detection feedback during calibration.
        
        Shows detected markers (green), rejected candidates (red outline),
        and a status bar so the user can see what the detector finds.
        """
        output = frame.copy()
        
        corners = self._last_corners
        ids = self._last_ids
        rejected = self._last_rejected
        
        # Draw detected markers with green border + ID label
        if ids is not None and len(corners) > 0:
            aruco.drawDetectedMarkers(output, corners, ids)
        
        # Draw rejected candidates as small red outlines
        if rejected is not None:
            for rej in rejected:
                pts = rej.reshape((-1, 1, 2)).astype(np.int32)
                cv2.polylines(output, [pts], True, (0, 0, 180), 1)
        
        # Status bar at top showing detection results
        h, w = output.shape[:2]
        n_detected = 0 if ids is None else len(ids)
        n_rejected = 0 if rejected is None else len(rejected)
        
        # Which IDs were found?
        found_ids = []
        if ids is not None:
            found_ids = sorted(ids.flatten().tolist())
        
        id_map = {0: 'TL', 1: 'TR', 2: 'BR', 3: 'BL'}
        found_str = ", ".join(f"ID{i}({id_map.get(i, '?')})" for i in found_ids)
        missing = [f"ID{mid}({id_map[mid]})" for mid in sorted(self.CORNER_IDS.values())
                   if mid not in found_ids]
        
        # Draw status panel
        panel_h = 75
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, panel_h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        # Line 1: detection count
        color = (0, 255, 0) if n_detected == 4 else (0, 200, 255)
        cv2.putText(output, f"Markers: {n_detected}/4 detected  |  {n_rejected} candidates rejected",
                    (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)
        
        # Line 2: found IDs
        if found_str:
            cv2.putText(output, f"Found: {found_str}",
                        (10, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 255, 0), 1)
        
        # Line 3: missing IDs
        if missing:
            miss_str = ", ".join(missing)
            cv2.putText(output, f"Missing: {miss_str}",
                        (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 100, 255), 1)
        elif n_detected >= 4:
            cv2.putText(output, "All 4 found! Press SPACE to confirm.",
                        (10, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2)
        
        return output
    
    def get_marker_rects(self) -> List[Tuple[int, int, int, int]]:
        """Return bounding rectangles (x, y, w, h) of last detected markers.
        
        Useful for filtering out YOLO detections that overlap with markers.
        """
        rects = []
        if self._last_ids is not None and len(self._last_corners) > 0:
            for c in self._last_corners:
                pts = c[0]  # shape (4,2)
                x_min = int(pts[:, 0].min())
                y_min = int(pts[:, 1].min())
                x_max = int(pts[:, 0].max())
                y_max = int(pts[:, 1].max())
                # Add some margin
                margin = max(x_max - x_min, y_max - y_min) // 4
                rects.append((x_min - margin, y_min - margin,
                              x_max - x_min + 2 * margin,
                              y_max - y_min + 2 * margin))
        return rects


def generate_corner_markers(output_dir: str, marker_size: int = 600):
    """
    Generate printable corner markers for table calibration.
    
    Args:
        output_dir: Directory to save marker images
        marker_size: Size of marker in pixels
    """
    import os
    
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    margin = 100
    canvas_size = marker_size + margin * 2
    
    corner_labels = {
        0: "TOP-LEFT",
        1: "TOP-RIGHT", 
        2: "BOTTOM-RIGHT",
        3: "BOTTOM-LEFT"
    }
    
    os.makedirs(output_dir, exist_ok=True)
    
    for marker_id, label in corner_labels.items():
        # Generate marker
        img = aruco.generateImageMarker(dictionary, marker_id, marker_size)
        
        # Add white border
        canvas = 255 * np.ones((canvas_size, canvas_size), dtype=np.uint8)
        canvas[margin:margin+marker_size, margin:margin+marker_size] = img
        
        # Add label
        cv2.putText(canvas, f"ID {marker_id} - {label}", (30, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
        
        filename = f"corner_marker_{marker_id}_{label.lower().replace('-', '_')}.png"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, canvas)
        print(f"Generated: {filename}")


if __name__ == "__main__":
    # Generate markers when run directly
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    markers_dir = os.path.join(script_dir, "..", "markers")
    generate_corner_markers(markers_dir)
    print(f"\nMarkers saved to: {markers_dir}")
    print("Print these and place at table corners for calibration.")
