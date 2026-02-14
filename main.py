"""
Table Guidance System - Main Application
=========================================
Entry point for the physical task guidance system.

This ties together all five modules (marker detection, object detection,
hand tracking, state management, visualization) into a single pipeline:
  camera frame -> perception -> state update -> render overlays -> display

Design choice: I separated the pipeline into distinct modules so each
can be tested/swapped independently. The main loop just orchestrates them.

I made ArUco markers optional (press SPACE to skip) because not everyone
has a printer handy, and screen-relative mode still demonstrates the core
pipeline well enough for a demo.

Controls:
- SPACE: Start the procedure / skip calibration
- R: Reset and start over
- Q or ESC: Quit
- D: Toggle debug info
- C: Open color calibration tool

Author: Ruijie Zheng
Date: February 2026
"""

import cv2
import numpy as np
import os
import sys
import argparse
import json
from pathlib import Path

# Add modules to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from modules import (
    MarkerDetector, generate_corner_markers,
    ObjectDetector, DetectedObject, ObjectStatus, ColorCalibrator,
    HandTracker, HandInfo, HandState,
    StateManager, SystemStatus, ProcedureState, StepState,
    Visualizer, VisualizationConfig
)


class TableGuidanceSystem:
    """Main application class for the Table Guidance System."""
    
    def __init__(self, config_path: str = None, camera_id: int = 0):
        """
        Initialize the Table Guidance System.
        
        Args:
            config_path: Path to procedure configuration JSON
            camera_id: Camera device ID (default: 0)
        """
        # Paths
        self.base_dir = Path(__file__).parent
        if config_path is None:
            config_path = self.base_dir / "config" / "procedure.json"
        self.config_path = Path(config_path)
        
        # Load procedure config
        with open(self.config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize modules
        print("Initializing modules...")
        self.marker_detector = MarkerDetector()
        self.object_detector = ObjectDetector(self.config.get('objects', []))
        self.hand_tracker = HandTracker()
        self.state_manager = StateManager(str(self.config_path))
        self.visualizer = Visualizer()
        
        # Camera
        self.camera_id = camera_id
        self.cap = None
        
        # Marker mode (can be disabled to use screen-relative coords)
        self.use_markers = True
        
        # Debug mode
        self.debug_mode = False
        self.show_masks = False
        
        # Window name
        self.window_name = "Table Guidance System"
        
        # Callbacks
        self.state_manager.set_callbacks(
            on_step_complete=self._on_step_complete,
            on_procedure_complete=self._on_procedure_complete
        )
        
        print(f"Loaded procedure: {self.state_manager.procedure_name}")
        print(f"Steps: {len(self.state_manager.steps)}")
        
    def _on_step_complete(self, step):
        """Callback when a step is completed."""
        print(f"✓ Completed: {step.instruction}")
        duration = step.get_duration()
        if duration:
            print(f"  Time: {duration:.1f}s")
            
    def _on_procedure_complete(self):
        """Callback when entire procedure is completed."""
        duration = self.state_manager.get_procedure_duration()
        print("\n" + "=" * 50)
        print("🎉 TABLE SET SUCCESSFULLY! 🎉")
        if duration:
            print(f"Total time: {duration:.1f}s")
        print("=" * 50 + "\n")
        
    def start(self):
        """Start the guidance system."""
        # Initialize camera
        print(f"Opening camera {self.camera_id}...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print("Error: Could not open camera")
            return False
            
        # Set camera resolution (optional, adjust as needed)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        
        # Start calibration
        self.state_manager.start_calibration()
        
        print("\n" + "=" * 50)
        print("TABLE GUIDANCE SYSTEM")
        print("=" * 50)
        print("\nSetup Options:")
        print("  Option A: With ArUco Markers (precise mode)")
        print("    1. Place 4 markers at table corners")
        print("    2. Wait for auto-calibration, then press SPACE")
        print("  Option B: Without Markers (simple mode)")
        print("    1. Just press SPACE to skip calibration")
        print("    2. The camera frame = your workspace")
        print("\n  Then place colored objects and press SPACE to begin")
        print("\nControls: SPACE=Start/Skip, R=Reset, Q=Quit, D=Debug, C=Calibrate")
        print("=" * 50 + "\n")
        
        return True
        
    def run(self):
        """Main loop."""
        if not self.start():
            return
            
        try:
            while True:
                ret, frame = self.cap.read()
                if not ret:
                    print("Error: Failed to read frame")
                    break
                    
                # Mirror the camera so it feels natural (like a selfie camera)
                frame = cv2.flip(frame, 1)
                
                # Process frame
                output = self.process_frame(frame)
                
                # Show result
                cv2.imshow(self.window_name, output)
                
                # Handle input
                key = cv2.waitKey(1) & 0xFF
                if not self.handle_input(key):
                    break
                    
                # Clear step completed flag after a few frames
                if self.state_manager.status.step_just_completed:
                    cv2.waitKey(500)  # Show feedback briefly
                    self.state_manager.clear_step_completed_flag()
                    
        finally:
            self.cleanup()
            
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        """Run one frame through the full pipeline: detect -> update state -> render."""
        h, w = frame.shape[:2]
        
        # --- Core pipeline: perception -> state -> render ---
        # This is the heart of the system. Each step feeds the next.
        
        # 1. Detect table boundary (only if using markers)
        table_visible = False
        missing_markers = []
        if self.use_markers:
            table_visible = self.marker_detector.update_table_boundary(frame)
            missing_markers = self.marker_detector.get_missing_markers()
        
        # 2. Detect objects
        # Pass marker_detector only if using markers and calibrated
        md = self.marker_detector if (self.use_markers and self.marker_detector.is_calibrated()) else None
        detected_objects = self.object_detector.detect_objects(frame, md)
        
        # 3. In no-marker mode, convert screen coords to normalized (0-1) "table" coords
        if not self.use_markers:
            table_visible = True  # Always "visible" in no-marker mode
            for obj_id, obj in detected_objects.items():
                if obj.visible:
                    # Normalize screen position to 0-1 range
                    nx = obj.center[0] / w
                    ny = obj.center[1] / h
                    obj.table_position = (nx, ny)
        
        # 4. Track hand
        hand_info = self.hand_tracker.process_frame(frame, md)
        
        # 5. Determine hand-object interaction
        hand_near_object = False
        current_obj_id = self.state_manager.get_current_object_id()
        if current_obj_id and hand_info.detected:
            obj = detected_objects.get(current_obj_id)
            if obj and obj.visible:
                hand_near_object = self.hand_tracker.is_interacting_with_object(
                    obj.center, 
                    (obj.bounding_box[2], obj.bounding_box[3])
                )
                
        # 6. Update state
        object_positions = {}
        for obj_id, obj in detected_objects.items():
            if obj.visible and obj.table_position:
                object_positions[obj_id] = obj.table_position
                
        self.state_manager.update(
            table_visible=table_visible,
            object_positions=object_positions,
            hand_near_object=hand_near_object,
            missing_markers=missing_markers
        )
        
        # 7. Render visualization
        hand_pos = hand_info.palm_center if hand_info.detected else None
        output = self.visualizer.render(
            frame,
            self.marker_detector,
            self.state_manager,
            detected_objects,
            hand_pos,
            use_markers=self.use_markers
        )
        # Show general YOLO detections (all objects YOLO sees) with labeled bounding boxes.
        # This shows ALL objects YOLO sees, not just the ones in our procedure --
        # makes the demo more impressive and proves the detection actually works.
        # 7.5 Draw YOLO-detected objects with their actual names
        for gen_obj in self.object_detector.general_objects:
            if gen_obj.visible:
                x, y, bw, bh = gen_obj.bounding_box
                color = gen_obj.color_bgr
                label = f"{gen_obj.name} {gen_obj.confidence:.0%}"
                cv2.rectangle(output, (x, y), (x + bw, y + bh), color, 2)
                # Label background for readability
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
                cv2.rectangle(output, (x, y - th - 8), (x + tw + 4, y), color, -1)
                cv2.putText(output, label, (x + 2, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
        
        # 8. Always draw hand tracking overlay when hand is detected
        if hand_info.detected:
            output = self.hand_tracker.draw_debug(output, draw_landmarks=True)
        
        # 9. Debug overlays
        if self.debug_mode:
            output = self._draw_debug_info(output, detected_objects, hand_info)
            
        return output
        
    def _draw_debug_info(self, frame: np.ndarray,
                         detected_objects: dict,
                         hand_info: HandInfo) -> np.ndarray:
        """Draw debug information on frame."""
        output = frame.copy()
        
        # Draw marker detection debug
        output = self.marker_detector.draw_debug(output)
        
        # Draw hand tracking debug
        if hand_info.detected:
            output = self.hand_tracker.draw_debug(output, draw_landmarks=True)
            
        # Debug info panel
        debug_info = [
            f"FPS: {self.cap.get(cv2.CAP_PROP_FPS):.0f}",
            f"State: {self.state_manager.procedure_state.value}",
            f"Table: {'Yes' if self.marker_detector.is_calibrated() else 'No'}",
            f"Hand: {hand_info.state.value if hand_info.detected else 'None'}",
        ]
        
        # Object positions
        for obj_id, obj in detected_objects.items():
            if obj.visible and obj.table_position:
                debug_info.append(
                    f"{obj.name}: ({obj.table_position[0]:.2f}, {obj.table_position[1]:.2f})"
                )
                
        # Draw debug panel
        y = 100
        for text in debug_info:
            cv2.putText(output, text, (10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            y += 20
            
        return output
        
    def handle_input(self, key: int) -> bool:
        """Process a keypress. Returns False if user wants to quit."""
        if key == ord('q') or key == 27:  # Q or ESC
            return False
            
        elif key == ord(' '):  # SPACE - Start procedure or skip calibration
            if self.state_manager.procedure_state == ProcedureState.CALIBRATING:
                if self.marker_detector.is_calibrated():
                    self.state_manager.complete_calibration()
                else:
                    print("\nSkipping marker calibration (using screen-relative mode)...")
                    self.state_manager.skip_calibration()
                    self.use_markers = False
            elif self.state_manager.procedure_state == ProcedureState.INITIALIZING:
                print("\nStarting procedure...")
                self.state_manager.start_procedure()
                
        elif key == ord('r'):  # R - Reset
            print("\nResetting procedure...")
            self.state_manager.reset()
            self.state_manager.start_calibration()
            
        elif key == ord('d'):  # D - Toggle debug
            self.debug_mode = not self.debug_mode
            print(f"Debug mode: {'ON' if self.debug_mode else 'OFF'}")
            
        elif key == ord('c'):  # C - Color calibration
            self._run_color_calibration()
            
        elif key == ord('m'):  # M - Generate markers
            self._generate_markers()
            
        return True
        
    def _run_color_calibration(self):
        """Run the color calibration tool."""
        print("\nOpening color calibration tool...")
        print("Adjust sliders to match your objects")
        print("Press 'p' to print values, 'q' to close")
        
        calibrator = ColorCalibrator()
        calibrator.create_trackbars()
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
                
            frame = cv2.flip(frame, 1)
            calibrator.show_mask(frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                print("\nCurrent HSV values:")
                calibrator.print_values()
                
        cv2.destroyWindow("Color Calibrator")
        
    def _generate_markers(self):
        """Generate ArUco markers for printing."""
        markers_dir = self.base_dir / "markers"
        print(f"\nGenerating markers in: {markers_dir}")
        generate_corner_markers(str(markers_dir))
        print("Markers generated! Print them and place at table corners.")
        
    def cleanup(self):
        """Clean up resources."""
        print("\nCleaning up...")
        if self.cap:
            self.cap.release()
        self.hand_tracker.release()
        cv2.destroyAllWindows()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Table Guidance System - AR-based task guidance"
    )
    parser.add_argument(
        '--config', '-c',
        type=str,
        default=None,
        help='Path to procedure configuration JSON'
    )
    parser.add_argument(
        '--camera', '-cam',
        type=int,
        default=0,
        help='Camera device ID (default: 0)'
    )
    parser.add_argument(
        '--generate-markers', '-g',
        action='store_true',
        help='Generate ArUco markers and exit'
    )
    
    args = parser.parse_args()
    
    # Generate markers only
    if args.generate_markers:
        base_dir = Path(__file__).parent
        markers_dir = base_dir / "markers"
        generate_corner_markers(str(markers_dir))
        print(f"\nMarkers saved to: {markers_dir}")
        print("Print these and place at table corners:")
        print("  - Marker 0: Top-Left")
        print("  - Marker 1: Top-Right")
        print("  - Marker 2: Bottom-Right")
        print("  - Marker 3: Bottom-Left")
        return
    
    # Run main application
    system = TableGuidanceSystem(
        config_path=args.config,
        camera_id=args.camera
    )
    system.run()


if __name__ == "__main__":
    main()
