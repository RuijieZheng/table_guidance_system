"""
State Management Module
=======================
Implements the FSM (finite state machine) that tracks procedure progress.

Design choices:
- The procedure flows: CALIBRATING -> INITIALIZING -> IN_PROGRESS -> COMPLETED
- Each step has its own state: PENDING / ACTIVE / IN_PROGRESS / COMPLETED
- Step completion is checked by Euclidean distance between the object's
  normalized table position and the target zone center. If distance < radius,
  the step is marked complete. Simple but works well.
- The state manager is decoupled from detection -- it just receives position
  data and makes state transition decisions. This makes it easy to swap out
  the detection backend without touching the procedure logic.

Author: Ruijie Zheng
"""

import json
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time


class ProcedureState(Enum):
    """Overall procedure state."""
    NOT_STARTED = "not_started"
    CALIBRATING = "calibrating"
    INITIALIZING = "initializing"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    ERROR = "error"


class StepState(Enum):
    """State of an individual step."""
    PENDING = "pending"
    ACTIVE = "active"
    IN_PROGRESS = "in_progress"  # User is actively moving the object
    COMPLETED = "completed"
    SKIPPED = "skipped"


class VisibilityState(Enum):
    """Visibility states for required elements."""
    VISIBLE = "visible"
    PARTIALLY_VISIBLE = "partially_visible"
    NOT_VISIBLE = "not_visible"


@dataclass
class TargetZone:
    """Represents a target zone where an object should be placed."""
    zone_id: str
    name: str
    position: Tuple[float, float]  # Normalized table coordinates (0-1)
    radius: float  # Normalized radius
    description: str = ""
    
    def is_point_inside(self, point: Tuple[float, float], tolerance: float = 0.0) -> bool:
        """Check if a point is inside this zone."""
        dx = point[0] - self.position[0]
        dy = point[1] - self.position[1]
        distance = (dx**2 + dy**2) ** 0.5
        return distance <= (self.radius + tolerance)


@dataclass
class ProcedureStep:
    """Represents a single step in the procedure."""
    step_number: int
    instruction: str
    object_id: str
    target_zone_id: str
    
    state: StepState = StepState.PENDING
    start_time: Optional[float] = None
    completion_time: Optional[float] = None
    
    def mark_active(self):
        """Mark this step as active."""
        self.state = StepState.ACTIVE
        self.start_time = time.time()
        
    def mark_in_progress(self):
        """Mark this step as in progress (user moving object)."""
        self.state = StepState.IN_PROGRESS
        
    def mark_completed(self):
        """Mark this step as completed."""
        self.state = StepState.COMPLETED
        self.completion_time = time.time()
        
    def get_duration(self) -> Optional[float]:
        """Get duration of this step in seconds."""
        if self.start_time is None:
            return None
        end = self.completion_time or time.time()
        return end - self.start_time


@dataclass
class SystemStatus:
    """Current status of the system for UI display."""
    # Procedure status
    procedure_state: ProcedureState = ProcedureState.NOT_STARTED
    current_step_num: int = 0
    total_steps: int = 0
    current_instruction: str = ""
    
    # Visibility status
    table_visible: bool = False
    target_object_visible: bool = False
    target_object_name: str = ""
    
    # Interaction status
    user_moving_item: bool = False
    object_in_target: bool = False
    
    # Completion
    step_just_completed: bool = False
    procedure_completed: bool = False
    
    # Calibration
    markers_detected: int = 0
    calibration_complete: bool = False
    missing_markers: List[str] = field(default_factory=list)
    
    # Debug info
    debug_messages: List[str] = field(default_factory=list)


class StateManager:
    """Manages the overall state of the guidance procedure."""
    
    def __init__(self, config_path: str = None):
        """
        Initialize state manager.
        
        Args:
            config_path: Path to procedure configuration JSON file
        """
        self.procedure_state = ProcedureState.NOT_STARTED
        self.steps: List[ProcedureStep] = []
        self.target_zones: Dict[str, TargetZone] = {}
        self.object_configs: List[Dict] = []
        
        self.current_step_index: int = 0
        self.procedure_name: str = ""
        self.description: str = ""
        
        # Completion tracking
        self.completed_steps: List[int] = []
        self.procedure_start_time: Optional[float] = None
        self.procedure_end_time: Optional[float] = None
        
        # How close an object needs to be to the target zone center to count as
        # "placed". A bit of tolerance is needed since the detection bbox center
        # won't be exactly at the physical center of the object on the table.
        self.placement_tolerance = 0.02
        self.hold_time_required = 0.5  # seconds to hold object in zone before it counts
        # ^ without this, objects that briefly pass through a zone would trigger completion
        self._in_zone_start_time: Optional[float] = None
        
        # Callbacks
        self._on_step_complete: Optional[Callable] = None
        self._on_procedure_complete: Optional[Callable] = None
        
        # Status for UI
        self.status = SystemStatus()
        
        if config_path:
            self.load_config(config_path)
    
    def load_config(self, config_path: str):
        """Load procedure configuration from JSON file."""
        with open(config_path, 'r') as f:
            config = json.load(f)
            
        self.procedure_name = config.get('procedure_name', 'Unnamed Procedure')
        self.description = config.get('description', '')
        self.object_configs = config.get('objects', [])
        
        # Load target zones
        for zone_data in config.get('target_zones', []):
            zone = TargetZone(
                zone_id=zone_data['id'],
                name=zone_data['name'],
                position=tuple(zone_data['position']),
                radius=zone_data.get('radius', 0.08),
                description=zone_data.get('description', '')
            )
            self.target_zones[zone.zone_id] = zone
            
        # Load steps
        for step_data in config.get('steps', []):
            step = ProcedureStep(
                step_number=step_data['step_number'],
                instruction=step_data['instruction'],
                object_id=step_data['object_id'],
                target_zone_id=step_data['target_zone_id']
            )
            self.steps.append(step)
            
        self.status.total_steps = len(self.steps)
        
    def start_calibration(self):
        """Begin calibration phase."""
        self.procedure_state = ProcedureState.CALIBRATING
        self.status.procedure_state = ProcedureState.CALIBRATING
        self.status.current_instruction = "Place markers at table corners (or press SPACE to skip)"

    def skip_calibration(self):
        """Skip marker calibration and go directly to initializing."""
        self.procedure_state = ProcedureState.INITIALIZING
        self.status.procedure_state = ProcedureState.INITIALIZING
        self.status.calibration_complete = True
        self.status.current_instruction = "Press SPACE to start (no markers mode)"
        
    def complete_calibration(self):
        """Mark calibration as complete."""
        self.procedure_state = ProcedureState.INITIALIZING
        self.status.procedure_state = ProcedureState.INITIALIZING
        self.status.calibration_complete = True
        self.status.current_instruction = "Place objects on the table, then press SPACE to start"
        
    def start_procedure(self):
        """Start the main procedure."""
        if self.procedure_state not in [ProcedureState.INITIALIZING, ProcedureState.NOT_STARTED]:
            return
            
        self.procedure_state = ProcedureState.IN_PROGRESS
        self.procedure_start_time = time.time()
        self.current_step_index = 0
        
        if self.steps:
            self.steps[0].mark_active()
            
        self._update_status()
        
    def get_current_step(self) -> Optional[ProcedureStep]:
        """Get the current active step."""
        if 0 <= self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None
    
    def get_current_target_zone(self) -> Optional[TargetZone]:
        """Get the target zone for the current step."""
        step = self.get_current_step()
        if step:
            return self.target_zones.get(step.target_zone_id)
        return None
    
    def get_current_object_id(self) -> Optional[str]:
        """Get the object ID for the current step."""
        step = self.get_current_step()
        return step.object_id if step else None
    
    def update(self, 
               table_visible: bool,
               object_positions: Dict[str, Optional[Tuple[float, float]]],
               hand_near_object: bool = False,
               missing_markers: List[str] = None):
        """Feed in the latest detection data and update FSM state."""
        self.status.table_visible = table_visible
        self.status.missing_markers = missing_markers or []
        self.status.markers_detected = 4 - len(self.status.missing_markers)
        
        # Handle calibration state
        if self.procedure_state == ProcedureState.CALIBRATING:
            if table_visible:
                self.complete_calibration()
            return
            
        # Handle main procedure
        if self.procedure_state != ProcedureState.IN_PROGRESS:
            return
            
        step = self.get_current_step()
        if not step:
            return
            
        target_zone = self.get_current_target_zone()
        current_obj_id = step.object_id
        
        # Check if target object is visible
        obj_pos = object_positions.get(current_obj_id)
        self.status.target_object_visible = obj_pos is not None
        
        # Find object name from config
        for obj_config in self.object_configs:
            if obj_config['id'] == current_obj_id:
                self.status.target_object_name = obj_config['name']
                break
        
        # Check if user is moving the item
        if hand_near_object and obj_pos is not None:
            step.mark_in_progress()
            self.status.user_moving_item = True
        else:
            self.status.user_moving_item = False
            
        # Check if object is in target zone
        if obj_pos is not None and target_zone is not None:
            in_zone = target_zone.is_point_inside(obj_pos, self.placement_tolerance)
            self.status.object_in_target = in_zone
            
            if in_zone:
                # Track how long object has been in zone
                if self._in_zone_start_time is None:
                    self._in_zone_start_time = time.time()
                elif time.time() - self._in_zone_start_time >= self.hold_time_required:
                    # Object held in position long enough - complete step
                    self._complete_current_step()
            else:
                self._in_zone_start_time = None
        else:
            self.status.object_in_target = False
            self._in_zone_start_time = None
            
        self._update_status()
        
    def _complete_current_step(self):
        """Complete the current step and advance."""
        step = self.get_current_step()
        if not step:
            return
            
        step.mark_completed()
        self.completed_steps.append(step.step_number)
        self.status.step_just_completed = True
        
        if self._on_step_complete:
            self._on_step_complete(step)
            
        # Advance to next step
        self.current_step_index += 1
        self._in_zone_start_time = None
        
        if self.current_step_index >= len(self.steps):
            # All steps completed
            self._complete_procedure()
        else:
            # Activate next step
            self.steps[self.current_step_index].mark_active()
            
    def _complete_procedure(self):
        """Mark the entire procedure as complete."""
        self.procedure_state = ProcedureState.COMPLETED
        self.procedure_end_time = time.time()
        self.status.procedure_completed = True
        
        if self._on_procedure_complete:
            self._on_procedure_complete()
            
    def _update_status(self):
        """Update the status object for UI display."""
        self.status.procedure_state = self.procedure_state
        self.status.current_step_num = self.current_step_index + 1
        
        step = self.get_current_step()
        if step:
            self.status.current_instruction = f"Step {step.step_number}: {step.instruction}"
        elif self.procedure_state == ProcedureState.COMPLETED:
            self.status.current_instruction = "Table Set Successfully! 🎉"
        else:
            self.status.current_instruction = ""
            
    def clear_step_completed_flag(self):
        """Clear the step_just_completed flag after UI has shown feedback."""
        self.status.step_just_completed = False
        
    def get_procedure_duration(self) -> Optional[float]:
        """Get total procedure duration in seconds."""
        if self.procedure_start_time is None:
            return None
        end = self.procedure_end_time or time.time()
        return end - self.procedure_start_time
    
    def get_progress_percentage(self) -> float:
        """Get completion percentage (0-100)."""
        if not self.steps:
            return 0.0
        return (len(self.completed_steps) / len(self.steps)) * 100
    
    def reset(self):
        """Reset the state manager for a new run."""
        self.procedure_state = ProcedureState.NOT_STARTED
        self.current_step_index = 0
        self.completed_steps = []
        self.procedure_start_time = None
        self.procedure_end_time = None
        self._in_zone_start_time = None
        
        # Reset all steps
        for step in self.steps:
            step.state = StepState.PENDING
            step.start_time = None
            step.completion_time = None
            
        # Reset status
        self.status = SystemStatus()
        self.status.total_steps = len(self.steps)
        
    def set_callbacks(self, 
                      on_step_complete: Callable = None,
                      on_procedure_complete: Callable = None):
        """Set callback functions for events."""
        self._on_step_complete = on_step_complete
        self._on_procedure_complete = on_procedure_complete
        
    def get_all_target_zones(self) -> List[TargetZone]:
        """Get all target zones."""
        return list(self.target_zones.values())
    
    def get_completed_zone_ids(self) -> List[str]:
        """Get IDs of zones that have been completed."""
        completed_zones = []
        for step in self.steps:
            if step.state == StepState.COMPLETED:
                completed_zones.append(step.target_zone_id)
        return completed_zones


if __name__ == "__main__":
    # Test state manager
    import os
    
    config_path = os.path.join(os.path.dirname(__file__), 
                               "..", "config", "procedure.json")
    
    manager = StateManager(config_path)
    
    print(f"Procedure: {manager.procedure_name}")
    print(f"Steps: {len(manager.steps)}")
    print(f"Target Zones: {len(manager.target_zones)}")
    
    for step in manager.steps:
        print(f"  Step {step.step_number}: {step.instruction}")
        
    for zone_id, zone in manager.target_zones.items():
        print(f"  Zone {zone_id}: {zone.name} at {zone.position}")
