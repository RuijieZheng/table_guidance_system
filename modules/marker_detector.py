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
        self.detector = aruco.ArucoDetector(self.dictionary, self.parameters)
        
        # Homography matrix (screen -> table normalized coords)
        self.homography: Optional[np.ndarray] = None
        self.inverse_homography: Optional[np.ndarray] = None
        
        # Detected corner positions in screen coordinates
        self.corner_positions: Dict[str, np.ndarray] = {}
        
        # Table boundary polygon (screen coordinates)
        self.table_boundary: Optional[np.ndarray] = None
        
        # Virtual table dimensions (normalized 0-1)
        self.table_width = 1.0
        self.table_height = 1.0
        
    def detect_markers(self, frame: np.ndarray) -> Tuple[List, List, List]:
        """Detect ArUco markers and return (corners, ids, rejected)."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = self.detector.detectMarkers(gray)
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
