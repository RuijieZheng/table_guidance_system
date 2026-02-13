"""
Visualization Module
====================
Handles all AR overlay rendering including:
- Table boundary visualization
- Target zones with perspective warping
- Guidance arrows
- Status overlays
- Completion feedback
"""

import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from .state_manager import (
    StateManager, SystemStatus, ProcedureState, StepState, TargetZone
)
from .marker_detector import MarkerDetector
from .object_detector import DetectedObject, ObjectStatus


@dataclass
class VisualizationConfig:
    """Configuration for visualization appearance."""
    # Colors (BGR)
    color_boundary: Tuple[int, int, int] = (0, 255, 0)  # Green
    color_target_pending: Tuple[int, int, int] = (255, 165, 0)  # Orange
    color_target_active: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    color_target_complete: Tuple[int, int, int] = (0, 255, 0)  # Green
    color_arrow: Tuple[int, int, int] = (255, 0, 255)  # Magenta
    color_text: Tuple[int, int, int] = (255, 255, 255)  # White
    color_text_bg: Tuple[int, int, int] = (0, 0, 0)  # Black
    color_success: Tuple[int, int, int] = (0, 255, 0)  # Green
    color_warning: Tuple[int, int, int] = (0, 165, 255)  # Orange
    color_error: Tuple[int, int, int] = (0, 0, 255)  # Red
    
    # Sizes
    boundary_thickness: int = 3
    target_thickness: int = 2
    arrow_thickness: int = 3
    font_scale: float = 0.7
    font_thickness: int = 2
    
    # Animation
    pulse_speed: float = 2.0  # Hz
    

class Visualizer:
    """Renders AR overlays on the camera feed."""
    
    def __init__(self, config: VisualizationConfig = None):
        """Initialize visualizer with optional config."""
        self.config = config or VisualizationConfig()
        self.frame_count = 0
        
        # Celebration animation state
        self._celebration_start_time: Optional[float] = None
        self._celebration_duration = 2.0  # seconds
        
    def render(self, 
               frame: np.ndarray,
               marker_detector: MarkerDetector,
               state_manager: StateManager,
               detected_objects: Dict[str, DetectedObject],
               hand_position: Optional[Tuple[int, int]] = None,
               use_markers: bool = True) -> np.ndarray:
        """
        Render all visualizations on the frame.
        
        Args:
            frame: BGR image to draw on
            marker_detector: For coordinate transformation
            state_manager: Current procedure state
            detected_objects: Dict of detected objects
            hand_position: Optional hand position in screen coords
            use_markers: Whether markers are being used
            
        Returns:
            Frame with all overlays rendered
        """
        self.frame_count += 1
        output = frame.copy()
        status = state_manager.status
        h, w = frame.shape[:2]
        
        # Layer 1: Table boundary
        if use_markers:
            output = self._draw_table_boundary(output, marker_detector, status)
        
        # Layer 2: Target zones
        if use_markers and marker_detector.is_calibrated():
            output = self._draw_target_zones(
                output, marker_detector, state_manager, detected_objects
            )
        elif not use_markers:
            output = self._draw_target_zones_screen(
                output, state_manager, detected_objects, w, h
            )
        
        # Layer 3: Object highlights
        output = self._draw_object_highlights(
            output, state_manager, detected_objects
        )
        
        # Layer 4: Guidance arrow
        if status.procedure_state == ProcedureState.IN_PROGRESS:
            if use_markers:
                output = self._draw_guidance_arrow(
                    output, marker_detector, state_manager, detected_objects
                )
            else:
                output = self._draw_guidance_arrow_screen(
                    output, state_manager, detected_objects, w, h
                )
        
        # Layer 5: Hand indicator
        if hand_position and status.user_moving_item:
            output = self._draw_hand_indicator(output, hand_position)
        
        # Layer 6: Status overlay
        output = self._draw_status_overlay(output, status, state_manager)
        
        # Layer 7: Completion celebration
        if status.procedure_completed:
            output = self._draw_celebration(output)
        elif status.step_just_completed:
            output = self._draw_step_complete_feedback(output)
            
        return output
    
    def _draw_table_boundary(self, frame: np.ndarray, 
                             marker_detector: MarkerDetector,
                             status: SystemStatus) -> np.ndarray:
        """Draw the detected table boundary."""
        output = frame.copy()
        
        boundary = marker_detector.get_boundary_corners()
        if boundary is not None:
            # Draw filled semi-transparent polygon
            overlay = frame.copy()
            cv2.fillPoly(overlay, [boundary], (50, 50, 50))
            cv2.addWeighted(overlay, 0.2, output, 0.8, 0, output)
            
            # Draw boundary lines
            cv2.polylines(output, [boundary], True, 
                         self.config.color_boundary, 
                         self.config.boundary_thickness)
                         
            # Draw corner labels
            corner_names = ['TL(0)', 'TR(1)', 'BR(2)', 'BL(3)']
            for i, name in enumerate(corner_names):
                pt = tuple(boundary[i][0])
                cv2.circle(output, pt, 8, self.config.color_boundary, -1)
                cv2.putText(output, name, (pt[0] + 10, pt[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, 
                           self.config.color_boundary, 2)
        else:
            # Show missing markers warning
            if status.missing_markers:
                text = f"Missing markers: {', '.join(status.missing_markers)}"
                self._draw_text_with_bg(output, text, (10, 60), 
                                        self.config.color_warning)
                                        
        return output
    
    def _draw_target_zones(self, frame: np.ndarray,
                           marker_detector: MarkerDetector,
                           state_manager: StateManager,
                           detected_objects: Dict[str, DetectedObject]) -> np.ndarray:
        """Draw target zones with perspective warping."""
        output = frame.copy()
        
        current_step = state_manager.get_current_step()
        current_zone_id = current_step.target_zone_id if current_step else None
        completed_zones = state_manager.get_completed_zone_ids()
        
        for zone in state_manager.get_all_target_zones():
            # Determine zone color based on state
            if zone.zone_id in completed_zones:
                color = self.config.color_target_complete
                fill_alpha = 0.3
            elif zone.zone_id == current_zone_id:
                # Pulsing effect for active zone
                pulse = (np.sin(self.frame_count * 0.1 * self.config.pulse_speed) + 1) / 2
                color = tuple(int(c * (0.5 + 0.5 * pulse)) for c in self.config.color_target_active)
                fill_alpha = 0.2 + 0.1 * pulse
            else:
                color = self.config.color_target_pending
                fill_alpha = 0.1
                
            # Draw the zone with perspective
            self._draw_perspective_zone(output, marker_detector, zone, 
                                        color, fill_alpha)
                                        
            # Draw zone label
            screen_pos = marker_detector.table_to_screen(zone.position)
            if screen_pos:
                label = zone.name
                if zone.zone_id in completed_zones:
                    label += " ✓"
                self._draw_text_with_bg(output, label, 
                                        (screen_pos[0] - 30, screen_pos[1] + 5),
                                        color, font_scale=0.5)
                                        
        return output
    
    def _draw_perspective_zone(self, frame: np.ndarray,
                                marker_detector: MarkerDetector,
                                zone: TargetZone,
                                color: Tuple[int, int, int],
                                fill_alpha: float):
        """Draw a single target zone with perspective warping."""
        # Generate circle points in table coordinates
        num_points = 32
        angles = np.linspace(0, 2 * np.pi, num_points)
        
        table_points = []
        for angle in angles:
            x = zone.position[0] + zone.radius * np.cos(angle)
            y = zone.position[1] + zone.radius * np.sin(angle)
            table_points.append((x, y))
            
        # Transform to screen coordinates
        screen_points = []
        for tp in table_points:
            sp = marker_detector.table_to_screen(tp)
            if sp:
                screen_points.append(sp)
                
        if len(screen_points) < 3:
            return
            
        # Convert to numpy array
        pts = np.array(screen_points, dtype=np.int32).reshape((-1, 1, 2))
        
        # Draw filled zone with transparency
        overlay = frame.copy()
        cv2.fillPoly(overlay, [pts], color)
        cv2.addWeighted(overlay, fill_alpha, frame, 1 - fill_alpha, 0, frame)
        
        # Draw zone outline
        cv2.polylines(frame, [pts], True, color, self.config.target_thickness)
        
        # Draw center cross
        center_screen = marker_detector.table_to_screen(zone.position)
        if center_screen:
            size = 10
            cv2.line(frame, 
                    (center_screen[0] - size, center_screen[1]),
                    (center_screen[0] + size, center_screen[1]),
                    color, 2)
            cv2.line(frame,
                    (center_screen[0], center_screen[1] - size),
                    (center_screen[0], center_screen[1] + size),
                    color, 2)
    
    def _draw_target_zones_screen(self, frame: np.ndarray,
                                   state_manager: StateManager,
                                   detected_objects: Dict[str, DetectedObject],
                                   frame_w: int, frame_h: int) -> np.ndarray:
        """Draw target zones using screen-relative coordinates (no markers mode)."""
        output = frame.copy()
        
        current_step = state_manager.get_current_step()
        current_zone_id = current_step.target_zone_id if current_step else None
        completed_zones = state_manager.get_completed_zone_ids()
        
        for zone in state_manager.get_all_target_zones():
            # Convert normalized position to screen pixels
            cx = int(zone.position[0] * frame_w)
            cy = int(zone.position[1] * frame_h)
            radius = int(zone.radius * min(frame_w, frame_h))
            
            # Determine zone color based on state
            if zone.zone_id in completed_zones:
                color = self.config.color_target_complete
                fill_alpha = 0.3
            elif zone.zone_id == current_zone_id:
                pulse = (np.sin(self.frame_count * 0.1 * self.config.pulse_speed) + 1) / 2
                color = tuple(int(c * (0.5 + 0.5 * pulse)) for c in self.config.color_target_active)
                fill_alpha = 0.2 + 0.1 * pulse
            else:
                color = self.config.color_target_pending
                fill_alpha = 0.1
            
            # Draw filled circle with transparency
            overlay = output.copy()
            cv2.circle(overlay, (cx, cy), radius, color, -1)
            cv2.addWeighted(overlay, fill_alpha, output, 1 - fill_alpha, 0, output)
            
            # Draw outline
            cv2.circle(output, (cx, cy), radius, color, self.config.target_thickness)
            
            # Draw center cross
            size = 10
            cv2.line(output, (cx - size, cy), (cx + size, cy), color, 2)
            cv2.line(output, (cx, cy - size), (cx, cy + size), color, 2)
            
            # Draw zone label
            label = zone.name
            if zone.zone_id in completed_zones:
                label += " Done"
            self._draw_text_with_bg(output, label, (cx - 30, cy + 5),
                                    color, font_scale=0.5)
        
        return output
    
    def _draw_guidance_arrow_screen(self, frame: np.ndarray,
                                     state_manager: StateManager,
                                     detected_objects: Dict[str, DetectedObject],
                                     frame_w: int, frame_h: int) -> np.ndarray:
        """Draw guidance arrow in screen-relative mode (no markers)."""
        output = frame.copy()
        
        current_obj_id = state_manager.get_current_object_id()
        target_zone = state_manager.get_current_target_zone()
        
        if not current_obj_id or not target_zone:
            return output
            
        obj = detected_objects.get(current_obj_id)
        if not obj or not obj.visible:
            return output
        
        obj_center = obj.center
        target_screen = (int(target_zone.position[0] * frame_w),
                         int(target_zone.position[1] * frame_h))
        
        dx = target_screen[0] - obj_center[0]
        dy = target_screen[1] - obj_center[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 50:
            return output
            
        self._draw_dashed_line(output, obj_center, target_screen,
                               self.config.color_arrow, 
                               self.config.arrow_thickness)
        self._draw_animated_arrow(output, obj_center, target_screen)
        
        mid_point = ((obj_center[0] + target_screen[0]) // 2,
                     (obj_center[1] + target_screen[1]) // 2)
        self._draw_text_with_bg(output, f"{int(distance)}px", mid_point,
                                self.config.color_arrow, font_scale=0.5)
        
        return output
    
    def _draw_object_highlights(self, frame: np.ndarray,
                                 state_manager: StateManager,
                                 detected_objects: Dict[str, DetectedObject]) -> np.ndarray:
        """Draw highlights around detected objects."""
        output = frame.copy()
        
        current_obj_id = state_manager.get_current_object_id()
        
        for obj_id, obj in detected_objects.items():
            if not obj.visible:
                continue
                
            x, y, w, h = obj.bounding_box
            
            # Highlight current target object
            if obj_id == current_obj_id:
                # Pulsing highlight for current target
                pulse = (np.sin(self.frame_count * 0.15) + 1) / 2
                thickness = int(2 + 2 * pulse)
                
                # Draw glowing border
                cv2.rectangle(output, (x - 5, y - 5), (x + w + 5, y + h + 5),
                             self.config.color_target_active, thickness)
                             
                # Draw object name with highlight
                label = f">> {obj.name} <<"
                self._draw_text_with_bg(output, label, (x, y - 15),
                                        self.config.color_target_active)
            else:
                # Normal object highlight
                cv2.rectangle(output, (x, y), (x + w, y + h), obj.color_bgr, 2)
                cv2.putText(output, obj.name, (x, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj.color_bgr, 1)
                           
        return output
    
    def _draw_guidance_arrow(self, frame: np.ndarray,
                              marker_detector: MarkerDetector,
                              state_manager: StateManager,
                              detected_objects: Dict[str, DetectedObject]) -> np.ndarray:
        """Draw arrow from current object to target zone."""
        output = frame.copy()
        
        current_obj_id = state_manager.get_current_object_id()
        target_zone = state_manager.get_current_target_zone()
        
        if not current_obj_id or not target_zone:
            return output
            
        obj = detected_objects.get(current_obj_id)
        if not obj or not obj.visible:
            return output
            
        # Get positions
        obj_center = obj.center
        target_screen = marker_detector.table_to_screen(target_zone.position)
        
        if not target_screen:
            return output
            
        # Calculate arrow properties
        dx = target_screen[0] - obj_center[0]
        dy = target_screen[1] - obj_center[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        if distance < 50:  # Close enough, no arrow needed
            return output
            
        # Draw dashed line
        self._draw_dashed_line(output, obj_center, target_screen,
                               self.config.color_arrow, 
                               self.config.arrow_thickness)
        
        # Draw animated arrow tip
        self._draw_animated_arrow(output, obj_center, target_screen)
        
        # Draw distance indicator
        distance_text = f"{int(distance)}px"
        mid_point = ((obj_center[0] + target_screen[0]) // 2,
                     (obj_center[1] + target_screen[1]) // 2)
        self._draw_text_with_bg(output, distance_text, mid_point,
                                self.config.color_arrow, font_scale=0.5)
                                
        return output
    
    def _draw_dashed_line(self, frame: np.ndarray,
                          start: Tuple[int, int],
                          end: Tuple[int, int],
                          color: Tuple[int, int, int],
                          thickness: int):
        """Draw a dashed line between two points."""
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        distance = np.sqrt(dx**2 + dy**2)
        
        dash_length = 15
        gap_length = 10
        num_dashes = int(distance / (dash_length + gap_length))
        
        for i in range(num_dashes + 1):
            t_start = i * (dash_length + gap_length) / distance
            t_end = min((i * (dash_length + gap_length) + dash_length) / distance, 1.0)
            
            p1 = (int(start[0] + dx * t_start), int(start[1] + dy * t_start))
            p2 = (int(start[0] + dx * t_end), int(start[1] + dy * t_end))
            
            cv2.line(frame, p1, p2, color, thickness)
    
    def _draw_animated_arrow(self, frame: np.ndarray,
                              start: Tuple[int, int],
                              end: Tuple[int, int]):
        """Draw animated arrow moving from start to end."""
        # Calculate animation position (0 to 1)
        t = (self.frame_count % 60) / 60.0
        
        # Arrow head position along the line
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        
        arrow_pos = (
            int(start[0] + dx * t),
            int(start[1] + dy * t)
        )
        
        # Calculate arrow head direction
        angle = np.arctan2(dy, dx)
        arrow_size = 15
        
        # Arrow head points
        p1 = arrow_pos
        p2 = (int(arrow_pos[0] - arrow_size * np.cos(angle - np.pi/6)),
              int(arrow_pos[1] - arrow_size * np.sin(angle - np.pi/6)))
        p3 = (int(arrow_pos[0] - arrow_size * np.cos(angle + np.pi/6)),
              int(arrow_pos[1] - arrow_size * np.sin(angle + np.pi/6)))
        
        # Draw filled arrow head
        pts = np.array([p1, p2, p3], dtype=np.int32)
        cv2.fillPoly(frame, [pts], self.config.color_arrow)
    
    def _draw_hand_indicator(self, frame: np.ndarray,
                              hand_pos: Tuple[int, int]) -> np.ndarray:
        """Draw indicator showing hand is interacting."""
        output = frame.copy()
        
        # Draw pulsing ring around hand
        pulse = (np.sin(self.frame_count * 0.2) + 1) / 2
        radius = int(30 + 10 * pulse)
        
        cv2.circle(output, hand_pos, radius, (0, 255, 255), 2)
        cv2.putText(output, "Moving...", 
                   (hand_pos[0] - 30, hand_pos[1] - radius - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                   
        return output
    
    def _draw_status_overlay(self, frame: np.ndarray,
                              status: SystemStatus,
                              state_manager: StateManager) -> np.ndarray:
        """Draw status information overlay."""
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Semi-transparent status bar at top
        overlay = output.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (30, 30, 30), -1)
        cv2.addWeighted(overlay, 0.7, output, 0.3, 0, output)
        
        # Title / Current instruction
        cv2.putText(output, status.current_instruction, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, self.config.color_text, 2)
        
        # Progress bar
        progress = state_manager.get_progress_percentage()
        bar_width = 200
        bar_height = 15
        bar_x = w - bar_width - 20
        bar_y = 20
        
        # Background
        cv2.rectangle(output, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + bar_height),
                     (100, 100, 100), -1)
        # Fill
        fill_width = int(bar_width * progress / 100)
        cv2.rectangle(output, (bar_x, bar_y),
                     (bar_x + fill_width, bar_y + bar_height),
                     self.config.color_success, -1)
        # Border
        cv2.rectangle(output, (bar_x, bar_y),
                     (bar_x + bar_width, bar_y + bar_height),
                     self.config.color_text, 1)
        # Label
        progress_text = f"Progress: {int(progress)}%"
        cv2.putText(output, progress_text, (bar_x, bar_y + bar_height + 20),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config.color_text, 1)
        
        # Step counter
        step_text = f"Step {status.current_step_num}/{status.total_steps}"
        cv2.putText(output, step_text, (10, 55),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.color_text, 1)
        
        # Status indicators at bottom
        y_bottom = h - 30
        
        # Table visibility
        table_color = self.config.color_success if status.table_visible else self.config.color_error
        cv2.circle(output, (20, y_bottom), 8, table_color, -1)
        cv2.putText(output, "Table", (35, y_bottom + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, table_color, 1)
        
        # Object visibility
        obj_color = self.config.color_success if status.target_object_visible else self.config.color_warning
        cv2.circle(output, (110, y_bottom), 8, obj_color, -1)
        obj_text = status.target_object_name if status.target_object_visible else "Object"
        cv2.putText(output, obj_text, (125, y_bottom + 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, obj_color, 1)
        
        # In target indicator
        if status.object_in_target:
            cv2.circle(output, (230, y_bottom), 8, self.config.color_success, -1)
            cv2.putText(output, "In Zone!", (245, y_bottom + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.config.color_success, 1)
        
        # Controls hint
        controls = "SPACE: Start | R: Reset | Q: Quit | C: Calibrate Colors"
        cv2.putText(output, controls, (w - 450, h - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                   
        return output
    
    def _draw_step_complete_feedback(self, frame: np.ndarray) -> np.ndarray:
        """Draw feedback when a step is completed."""
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Green flash overlay
        overlay = output.copy()
        overlay[:] = (0, 100, 0)
        cv2.addWeighted(overlay, 0.2, output, 0.8, 0, output)
        
        # Success text
        text = "Great Job!"
        font_scale = 1.5
        thickness = 3
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 
                                    font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2
        
        # Shadow
        cv2.putText(output, text, (text_x + 2, text_y + 2),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 2)
        # Text
        cv2.putText(output, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, 
                   self.config.color_success, thickness)
                   
        return output
    
    def _draw_celebration(self, frame: np.ndarray) -> np.ndarray:
        """Draw celebration animation for procedure completion."""
        output = frame.copy()
        h, w = output.shape[:2]
        
        # Rainbow border effect
        border_colors = [
            (0, 0, 255), (0, 127, 255), (0, 255, 255),
            (0, 255, 0), (255, 0, 0), (255, 0, 127)
        ]
        border_idx = (self.frame_count // 5) % len(border_colors)
        border_color = border_colors[border_idx]
        cv2.rectangle(output, (5, 5), (w - 5, h - 5), border_color, 10)
        
        # Main success message
        text = "Table Set Successfully!"
        font_scale = 1.2
        thickness = 3
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX,
                                    font_scale, thickness)[0]
        text_x = (w - text_size[0]) // 2
        text_y = h // 2 - 20
        
        # Pulsing effect
        pulse = (np.sin(self.frame_count * 0.1) + 1) / 2
        scale = 1.0 + 0.1 * pulse
        
        # Shadow
        cv2.putText(output, text, (text_x + 3, text_y + 3),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * scale, (0, 0, 0), thickness + 2)
        # Text
        cv2.putText(output, text, (text_x, text_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale * scale,
                   self.config.color_success, thickness)
        
        # Celebration emoji
        emoji_y = text_y + 50
        cv2.putText(output, "Press R to restart or Q to quit", 
                   ((w - 300) // 2, emoji_y),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.config.color_text, 1)
                   
        return output
    
    def _draw_text_with_bg(self, frame: np.ndarray, text: str,
                           position: Tuple[int, int],
                           color: Tuple[int, int, int],
                           font_scale: float = None,
                           padding: int = 5):
        """Draw text with a semi-transparent background."""
        if font_scale is None:
            font_scale = self.config.font_scale
            
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = self.config.font_thickness
        
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        
        x, y = position
        bg_x1 = x - padding
        bg_y1 = y - text_size[1] - padding
        bg_x2 = x + text_size[0] + padding
        bg_y2 = y + padding
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2),
                     self.config.color_text_bg, -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        
        # Text
        cv2.putText(frame, text, (x, y), font, font_scale, color, thickness)
