# Table Guidance System

A Physical Task Guidance System that uses computer vision and AR overlays to guide users in organizing objects on a table. This project is a prototype for exploring how computer vision and real-time feedback can assist users in completing physical tasks.

## Project Overview

**Task Selected:** Project #9 Option 2 - Task Guidance System for Table Organization

**Customization Level:** N/A (This is a sensing/guidance system, not the DIY AT project)

### Features

- **ArUco Marker-Based Table Detection**: Uses 4 fiducial markers to define the working area and compute perspective transformation (homography)
- **Real-Time Object Detection**: Color-based detection for cups, bottles, and plates with configurable HSV ranges
- **Hand Tracking (Bonus)**: MediaPipe-based hand tracking to detect when users are interacting with objects
- **Perspective-Warped Visualizations**: Target zones are rendered with proper perspective to align with the table plane
- **State Management**: Tracks procedure progress through calibration → initialization → task execution → completion
- **Interactive Guidance**: Dynamic arrows pointing from objects to their target positions

## System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Main Application (main.py)                   │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Marker    │  │   Object    │  │    Hand     │              │
│  │  Detector   │  │  Detector   │  │   Tracker   │              │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘              │
│         │                │                │                      │
│         ▼                ▼                ▼                      │
│  ┌─────────────────────────────────────────────────────┐        │
│  │              State Manager (FSM)                     │        │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐ ┌─────────┐   │        │
│  │  │Calibrate│→│  Init   │→│In Progress│→│Complete│   │        │
│  │  └─────────┘ └─────────┘ └─────────┘ └─────────┘   │        │
│  └─────────────────────────────────────────────────────┘        │
│                          │                                       │
│                          ▼                                       │
│  ┌─────────────────────────────────────────────────────┐        │
│  │              Visualizer (AR Overlay)                 │        │
│  │  • Table boundary  • Target zones (perspective)     │        │
│  │  • Guidance arrows • Status overlay • Feedback      │        │
│  └─────────────────────────────────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

### Module Descriptions

| Module | File | Description |
|--------|------|-------------|
| Marker Detector | `modules/marker_detector.py` | ArUco marker detection, homography computation, coordinate transformation |
| Object Detector | `modules/object_detector.py` | Color-based object detection with HSV segmentation |
| Hand Tracker | `modules/hand_tracker.py` | MediaPipe hand tracking, gesture recognition |
| State Manager | `modules/state_manager.py` | Procedure state machine, step tracking, completion detection |
| Visualizer | `modules/visualizer.py` | AR overlay rendering with perspective warping |

## Setup Instructions

### Prerequisites

- Python 3.8+
- Webcam
- Printer (for ArUco markers)
- Colored objects: red cup, blue bottle, green plate (or calibrate for your objects)

### Installation

1. **Clone the repository** (if not already done):
   ```bash
   cd table_guidance_system
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Download hand tracking model** (if not already present):
   The model file `models/hand_landmarker.task` is required for hand tracking.
   It will show a download link if missing. Alternatively:
   ```bash
   mkdir models
   curl -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
   ```

5. **(Optional) Generate ArUco markers**:
   ```bash
   python main.py --generate-markers
   ```
   Print from the `markers/` folder and place at table corners.

6. **Calibrate colors** (if using different objects):
   Press `C` while the app is running to open the color calibration tool.
   Update `config/procedure.json` with your HSV values.

### Running the System

```bash
python main.py
```

**Two modes:**
- **With Markers**: Place 4 ArUco markers at table corners → system auto-calibrates → press SPACE → press SPACE again to start
- **Without Markers (recommended for quick demo)**: Just press SPACE to skip calibration → press SPACE again to start. The camera frame becomes your workspace.

**Command-line options:**
```bash
python main.py --help
python main.py --camera 1          # Use different camera
python main.py --config custom.json # Use custom procedure config
python main.py --generate-markers   # Generate markers only
```

### Controls

| Key | Action |
|-----|--------|
| SPACE | Start the procedure / Skip calibration |
| R | Reset and start over |
| Q | Quit the application |
| D | Toggle debug mode |
| C | Open color calibration tool |
| M | Generate ArUco markers |

## Task Workflow

1. **Calibration** (optional): System detects all 4 corner markers and computes table homography. Press SPACE to skip.
2. **Initialization**: User places objects randomly on the table
3. **Task 1**: Move the Red Cup to the Center zone
4. **Task 2**: Move the Blue Bottle to the Top-Right zone
5. **Task 3**: Move the Green Plate to the Top-Left zone
6. **Completion**: "Table Set Successfully!" message displayed

## Configuration

The procedure is defined in `config/procedure.json`:

```json
{
    "procedure_name": "Table Setting Task",
    "objects": [
        {
            "id": "cup",
            "name": "Cup",
            "color_lower": [0, 100, 100],
            "color_upper": [10, 255, 255],
            "color_name": "red"
        }
        // ... more objects
    ],
    "target_zones": [
        {
            "id": "zone_a",
            "name": "Center",
            "position": [0.5, 0.5],
            "radius": 0.08
        }
        // ... more zones
    ],
    "steps": [
        {
            "step_number": 1,
            "instruction": "Move the Red Cup to the Center",
            "object_id": "cup",
            "target_zone_id": "zone_a"
        }
        // ... more steps
    ]
}
```

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | ≥4.5 | Image processing, visualization |
| opencv-contrib-python | ≥4.5 | ArUco marker detection |
| numpy | ≥1.19 | Numerical operations |
| mediapipe | ≥0.10.30 | Hand tracking (Tasks API) |

## Project Structure

```
table_guidance_system/
├── main.py                 # Main application entry point
├── README.md               # This file
├── requirements.txt        # Python dependencies
├── config/
│   └── procedure.json      # Task configuration
├── modules/
│   ├── __init__.py
│   ├── marker_detector.py  # ArUco detection & homography
│   ├── object_detector.py  # Color-based object detection
│   ├── hand_tracker.py     # MediaPipe hand tracking
│   ├── state_manager.py    # Procedure state machine
│   └── visualizer.py       # AR overlay rendering
├── models/
│   └── hand_landmarker.task # MediaPipe hand model (auto-downloaded)
└── markers/                # Generated ArUco markers (optional)
```

## Demo Video

[Link to demo video - to be recorded]

The video should demonstrate:
1. Calibration phase (marker detection)
2. Object placement
3. Step-by-step guidance with visual feedback
4. Completion celebration

## Evaluation Criteria Met

| Criteria | Implementation |
|----------|----------------|
| System Integration | All modules communicate through well-defined interfaces |
| Robustness | Handles temporary occlusion, missing markers shown as warnings |
| Coordinate Transformation | Full homography-based table-relative positioning |
| Code Quality | Modular design, type hints, docstrings, clear structure |

## Future Improvements

- [ ] Add YOLO-based object detection for better accuracy
- [ ] Implement voice guidance for accessibility
- [ ] Add support for custom task sequences via UI
- [ ] Persist calibration settings between sessions
- [ ] Add multi-user support with different difficulty levels

## Author

[Your Name]  
[Your Email]  
February 2026

## License

MIT License - Feel free to use and modify for educational purposes.
