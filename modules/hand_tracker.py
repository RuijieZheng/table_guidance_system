"""
Hand Tracking Module
====================
Tracks hand position using MediaPipe to detect user-object interactions.

Design choices:
- Using the MediaPipe Tasks API (not legacy mp.solutions.hands) because
  the legacy API was removed in newer versions (0.10.30+).
- The hand_landmarker.task model is loaded from models/ folder.
- Hand state is classified into OPEN/POINTING/PINCHING/GRABBING based
  on finger landmark distances -- this helps determine if the user is
  actively interacting with (picking up / moving) an object.
- Hand tracking is a bonus feature for the assignment but adds a lot
  to the demo since you can see the skeleton overlay in real time.

Author: Ruijie Zheng
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum

# Try to import MediaPipe Tasks API
# NOTE: The old mp.solutions.hands API was removed in MediaPipe 0.10.30+
# so we MUST use the Tasks API (HandLandmarker). If MediaPipe isn't installed,
# hand tracking is just disabled gracefully -- the rest of the system still works.
HAS_MEDIAPIPE = False
mp = None

try:
    import mediapipe as mp_module
    mp = mp_module
    from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions, RunningMode
    HAS_MEDIAPIPE = True
    print("MediaPipe Tasks API loaded successfully.")
except ImportError as e:
    print(f"Warning: MediaPipe Tasks API not available: {e}")
    print("Hand tracking disabled.")


class HandState(Enum):
    """Current state of the tracked hand."""
    NOT_DETECTED = "not_detected"
    OPEN = "open"
    POINTING = "pointing"
    PINCHING = "pinching"
    GRABBING = "grabbing"


@dataclass
class HandInfo:
    """Information about a tracked hand."""
    detected: bool = False
    
    # Key positions (screen coordinates)
    palm_center: Tuple[int, int] = (0, 0)
    index_tip: Tuple[int, int] = (0, 0)
    thumb_tip: Tuple[int, int] = (0, 0)
    wrist: Tuple[int, int] = (0, 0)
    
    # Table coordinates (if available)
    palm_table_pos: Optional[Tuple[float, float]] = None
    
    # State
    state: HandState = HandState.NOT_DETECTED
    pinch_distance: float = 0.0
    
    # Landmarks for visualization
    landmarks: Optional[List] = None
    
    # Movement tracking
    velocity: Tuple[float, float] = (0.0, 0.0)
    is_moving: bool = False


class HandTracker:
    """Tracks hand position and gestures using MediaPipe Tasks API."""
    
    # Landmark indices -- see MediaPipe hand model docs for the full diagram
    # https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker
    WRIST = 0
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_TIP = 20
    
    def __init__(self, max_hands: int = 1, 
                 detection_confidence: float = 0.5,
                 tracking_confidence: float = 0.5):
        """
        Initialize hand tracker.
        
        Args:
            max_hands: Maximum number of hands to track
            detection_confidence: Minimum detection confidence
            tracking_confidence: Minimum tracking confidence
        """
        self.enabled = HAS_MEDIAPIPE
        self.hand_info = HandInfo()
        self.landmarker = None
        
        # Previous positions for velocity calculation
        self._prev_palm_pos: Optional[Tuple[int, int]] = None
        self._velocity_smoothing = 0.3
        
        # These thresholds were tuned by trial and error while testing with my webcam.
        # pinch_threshold: how close thumb+index need to be (normalized 0-1 space)
        # grab_threshold: slightly larger -- all fingers curled = grabbing
        # movement_threshold: how many px/frame before we consider "moving"
        self.pinch_threshold = 0.06
        self.grab_threshold = 0.08
        self.movement_threshold = 5.0
        
        # Proximity detection
        self.proximity_radius = 80  # Pixels
        
        if self.enabled:
            self._init_landmarker(max_hands, detection_confidence, tracking_confidence)
    
    def _init_landmarker(self, max_hands: int, detection_conf: float, tracking_conf: float):
        """Initialize the MediaPipe HandLandmarker."""
        # Find model file
        model_paths = [
            os.path.join(os.path.dirname(__file__), "..", "models", "hand_landmarker.task"),
            os.path.join(os.path.dirname(__file__), "models", "hand_landmarker.task"),
            "models/hand_landmarker.task",
            "hand_landmarker.task",
        ]
        
        model_path = None
        for path in model_paths:
            if os.path.exists(path):
                model_path = os.path.abspath(path)
                break
        
        if not model_path:
            print("Warning: hand_landmarker.task model file not found.")
            print("Hand tracking disabled. Download from:")
            print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task")
            self.enabled = False
            return
        
        try:
            # Create options for the hand landmarker
            options = HandLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=model_path),
                running_mode=RunningMode.IMAGE,
                num_hands=max_hands,
                min_hand_detection_confidence=detection_conf,
                min_hand_presence_confidence=tracking_conf,
                min_tracking_confidence=tracking_conf
            )
            
            self.landmarker = HandLandmarker.create_from_options(options)
            print(f"HandLandmarker initialized with model: {model_path}")
        except Exception as e:
            print(f"Error initializing HandLandmarker: {e}")
            self.enabled = False
    
    def process_frame(self, frame: np.ndarray, 
                      marker_detector=None) -> HandInfo:
        """Run hand detection on a frame and return updated HandInfo."""
        if not self.enabled or self.landmarker is None:
            return HandInfo()
            
        h, w = frame.shape[:2]
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Detect hands
        try:
            result = self.landmarker.detect(mp_img)
        except Exception as e:
            print(f"Hand detection error: {e}")
            return HandInfo()
        
        self.hand_info = HandInfo()
        
        if result.hand_landmarks and len(result.hand_landmarks) > 0:
            # Track first hand only for now
            # TODO: maybe support two-handed interaction later?
            landmarks = result.hand_landmarks[0]
            self.hand_info.detected = True
            
            # Convert landmarks to screen coordinates and store for visualization
            screen_landmarks = []
            for lm in landmarks:
                x = int(lm.x * w)
                y = int(lm.y * h)
                screen_landmarks.append((x, y))
            
            self.hand_info.landmarks = screen_landmarks
            
            # Extract key positions
            self.hand_info.wrist = screen_landmarks[self.WRIST]
            self.hand_info.thumb_tip = screen_landmarks[self.THUMB_TIP]
            self.hand_info.index_tip = screen_landmarks[self.INDEX_TIP]
            
            # Palm center (approximate using middle finger MCP)
            self.hand_info.palm_center = screen_landmarks[self.MIDDLE_MCP]
            
            # Calculate pinch distance (normalized)
            thumb = landmarks[self.THUMB_TIP]
            index = landmarks[self.INDEX_TIP]
            self.hand_info.pinch_distance = np.sqrt(
                (thumb.x - index.x)**2 + 
                (thumb.y - index.y)**2 + 
                (thumb.z - index.z)**2
            )
            
            # Determine hand state
            self.hand_info.state = self._determine_hand_state(landmarks)
            
            # Calculate velocity
            if self._prev_palm_pos:
                dx = self.hand_info.palm_center[0] - self._prev_palm_pos[0]
                dy = self.hand_info.palm_center[1] - self._prev_palm_pos[1]
                
                # Smooth velocity
                vx = self._velocity_smoothing * dx + (1 - self._velocity_smoothing) * self.hand_info.velocity[0]
                vy = self._velocity_smoothing * dy + (1 - self._velocity_smoothing) * self.hand_info.velocity[1]
                
                self.hand_info.velocity = (vx, vy)
                speed = np.sqrt(vx**2 + vy**2)
                self.hand_info.is_moving = speed > self.movement_threshold
                
            self._prev_palm_pos = self.hand_info.palm_center
            
            # Convert to table coordinates if available
            if marker_detector and marker_detector.is_calibrated():
                table_pos = marker_detector.screen_to_table(self.hand_info.palm_center)
                if table_pos:
                    self.hand_info.palm_table_pos = table_pos
        else:
            self._prev_palm_pos = None
            
        return self.hand_info
    
    def _determine_hand_state(self, landmarks) -> HandState:
        """Determine hand state based on finger positions."""
        # Get key landmark positions
        thumb_tip = landmarks[self.THUMB_TIP]
        index_tip = landmarks[self.INDEX_TIP]
        middle_tip = landmarks[self.MIDDLE_TIP]
        ring_tip = landmarks[self.RING_TIP]
        pinky_tip = landmarks[self.PINKY_TIP]
        
        # Get MCP joints (knuckles)
        index_mcp = landmarks[self.INDEX_MCP]
        middle_mcp = landmarks[self.MIDDLE_MCP]
        ring_mcp = landmarks[self.RING_MCP]
        pinky_mcp = landmarks[self.PINKY_MCP]
        
        # Check pinch (thumb tip close to index tip)
        pinch_dist = np.sqrt(
            (thumb_tip.x - index_tip.x)**2 + 
            (thumb_tip.y - index_tip.y)**2
        )
        if pinch_dist < self.pinch_threshold:
            return HandState.PINCHING
            
        # Check if fingers are curled (grabbing)
        fingers_curled = 0
        if index_tip.y > index_mcp.y:
            fingers_curled += 1
        if middle_tip.y > middle_mcp.y:
            fingers_curled += 1
        if ring_tip.y > ring_mcp.y:
            fingers_curled += 1
        if pinky_tip.y > pinky_mcp.y:
            fingers_curled += 1
            
        if fingers_curled >= 3:
            return HandState.GRABBING
            
        # Check if only index pointing
        if (index_tip.y < index_mcp.y and 
            middle_tip.y > middle_mcp.y and
            ring_tip.y > ring_mcp.y):
            return HandState.POINTING
            
        return HandState.OPEN
    
    def is_near_point(self, point: Tuple[int, int], radius: int = None) -> bool:
        """Check if palm center is close enough to a screen point."""
        if not self.hand_info.detected:
            return False
            
        if radius is None:
            radius = self.proximity_radius
            
        dx = self.hand_info.palm_center[0] - point[0]
        dy = self.hand_info.palm_center[1] - point[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        return distance < radius
    
    def is_interacting_with_object(self, object_center: Tuple[int, int],
                                    object_size: Tuple[int, int] = (100, 100)) -> bool:
        """True if hand is near the object AND in a grab/pinch state."""
        if not self.hand_info.detected:
            return False
            
        # Check proximity to object
        proximity_radius = max(object_size) // 2 + self.proximity_radius
        if not self.is_near_point(object_center, proximity_radius):
            return False
            
        # Check if hand is in interacting state
        interacting_states = [HandState.PINCHING, HandState.GRABBING]
        return self.hand_info.state in interacting_states
    
    # Hand connections for drawing (MediaPipe hand topology)
    HAND_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # Thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # Index
        (0, 9), (9, 10), (10, 11), (11, 12), # Middle
        (0, 13), (13, 14), (14, 15), (15, 16), # Ring
        (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
        (5, 9), (9, 13), (13, 17)             # Palm
    ]
    
    def draw_debug(self, frame: np.ndarray, draw_landmarks: bool = True) -> np.ndarray:
        """Draw hand skeleton + state label on frame."""
        if not self.enabled:
            return frame
            
        output = frame.copy()
        
        if self.hand_info.detected:
            # Draw full hand landmarks manually
            if draw_landmarks and self.hand_info.landmarks:
                landmarks = self.hand_info.landmarks
                
                # Draw connections
                for connection in self.HAND_CONNECTIONS:
                    start_idx, end_idx = connection
                    if start_idx < len(landmarks) and end_idx < len(landmarks):
                        start_pt = landmarks[start_idx]
                        end_pt = landmarks[end_idx]
                        cv2.line(output, start_pt, end_pt, (0, 255, 0), 2)
                
                # Draw landmark points
                for i, pt in enumerate(landmarks):
                    # Different colors for fingertips
                    if i in [4, 8, 12, 16, 20]:  # Fingertips
                        cv2.circle(output, pt, 6, (255, 0, 0), -1)
                    else:
                        cv2.circle(output, pt, 4, (0, 255, 0), -1)
            
            # Draw palm center
            cv2.circle(output, self.hand_info.palm_center, 10, (255, 0, 255), -1)
            
            # Draw index fingertip
            cv2.circle(output, self.hand_info.index_tip, 8, (255, 255, 0), -1)
            
            # Draw proximity radius
            cv2.circle(output, self.hand_info.palm_center, self.proximity_radius, 
                      (255, 0, 255), 1)
            
            # Draw pinch line if pinching
            if self.hand_info.state == HandState.PINCHING:
                cv2.line(output, self.hand_info.thumb_tip, self.hand_info.index_tip,
                        (0, 255, 255), 3)
                        
            # Draw velocity vector
            if self.hand_info.is_moving:
                vx, vy = self.hand_info.velocity
                scale = 3
                end_point = (
                    int(self.hand_info.palm_center[0] + vx * scale),
                    int(self.hand_info.palm_center[1] + vy * scale)
                )
                cv2.arrowedLine(output, self.hand_info.palm_center, end_point,
                               (0, 255, 0), 2)
                               
            # State label
            state_colors = {
                HandState.OPEN: (255, 255, 255),
                HandState.POINTING: (255, 255, 0),
                HandState.PINCHING: (0, 255, 255),
                HandState.GRABBING: (0, 165, 255),
            }
            color = state_colors.get(self.hand_info.state, (255, 255, 255))
            cv2.putText(output, f"Hand: {self.hand_info.state.value}", 
                       (10, output.shape[0] - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                       
        return output
    
    def release(self):
        """Release resources."""
        if self.enabled and self.landmarker:
            self.landmarker.close()


if __name__ == "__main__":
    # Test hand tracking
    print("Hand Tracking Test")
    print("Press 'q' to quit")
    
    tracker = HandTracker()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        frame = cv2.flip(frame, 1)
        hand_info = tracker.process_frame(frame)
        
        # Draw debug visualization
        output = tracker.draw_debug(frame)
        
        # Display info
        if hand_info.detected:
            info_text = [
                f"State: {hand_info.state.value}",
                f"Pinch Dist: {hand_info.pinch_distance:.3f}",
                f"Moving: {hand_info.is_moving}",
                f"Velocity: ({hand_info.velocity[0]:.1f}, {hand_info.velocity[1]:.1f})"
            ]
            for i, text in enumerate(info_text):
                cv2.putText(output, text, (10, 30 + i * 25),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow("Hand Tracking", output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    tracker.release()
    cap.release()
    cv2.destroyAllWindows()
