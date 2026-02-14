# Table Guidance System

A Physical Task Guidance System that uses computer vision and AR overlays to guide users in organizing objects on a table. This project is a prototype for exploring how computer vision and real-time feedback can assist users in completing physical tasks.

## Project Overview

**Task Selected:** Project #9 Option 2 - Task Guidance System for Table Organization

**Customization Level:** N/A (This is a sensing/guidance system, not the DIY AT project)

### Features

- **ArUco Marker-Based Table Detection**: Uses 4 fiducial markers to define the working area and compute perspective transformation (homography). Can be skipped for quick demo mode.
- **YOLO Object Detection (Primary)**: YOLOv5s ONNX model identifies 80 COCO classes (cup, bottle, cell phone, mouse, keyboard, laptop, etc.) — each object is labeled by name with confidence score
- **Color-Based Detection (Fallback)**: HSV segmentation for color-specific objects when YOLO model is unavailable
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
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │   Marker    │  │ YOLO Object │  │    Hand     │              │
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
| Object Detector | `modules/object_detector.py` | YOLO + color-based object detection (identifies objects by name) |
| YOLO Detector | `modules/yolo_detector.py` | YOLOv5s ONNX inference — recognizes 80 COCO classes |
| Hand Tracker | `modules/hand_tracker.py` | MediaPipe hand tracking, gesture recognition |
| State Manager | `modules/state_manager.py` | Procedure state machine, step tracking, completion detection |
| Visualizer | `modules/visualizer.py` | AR overlay rendering with perspective warping |

## Setup Instructions

### Prerequisites

- Python 3.8+
- Webcam
- Common desk objects (phone, mouse, cup, bottle, etc. — detected by YOLO)
- (Optional) Printer for ArUco markers

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

4. **Download models** (if not already present):
   ```bash
   mkdir models
   # Hand tracking model
   curl -o models/hand_landmarker.task https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
   # YOLO object detection model
   curl -L -o models/yolov5s.onnx https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx
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
2. **Initialization**: User places objects (phone, pen, bottle) randomly on the table
3. **Task 1**: Move the Phone to the Center
4. **Task 2**: Move the Pen to the Top-Right zone
5. **Task 3**: Move the Bottle to the Top-Left zone
6. **Completion**: "Table Set Successfully!" message displayed

YOLO continuously identifies all visible objects on the desk (labeled with name + confidence).

## Configuration

The procedure is defined in `config/procedure.json`:

```json
{
    "procedure_name": "Table Setting Task",
    "objects": [
        {
            "id": "object_3",
            "name": "Bottle",
            "yolo_class": "bottle",
            "color_name": "yolo"
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
| onnxruntime | ≥1.16 | YOLO model inference (no PyTorch needed) |
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
│   ├── object_detector.py  # YOLO + color-based object detection
│   ├── yolo_detector.py    # YOLOv5s ONNX inference engine
│   ├── hand_tracker.py     # MediaPipe hand tracking
│   ├── state_manager.py    # Procedure state machine
│   └── visualizer.py       # AR overlay rendering
├── models/
│   └── hand_landmarker.task # MediaPipe hand model (auto-downloaded)
└── markers/                # Generated ArUco markers (optional)
```

### Object Detection Details

The system uses **YOLOv5s** (via ONNX Runtime) as the primary detection method:
- Recognizes **80 COCO classes** including: cup, bottle, cell phone, mouse, keyboard, laptop, book, remote, scissors, etc.
- Each detected object is **labeled by name** with confidence score
- No PyTorch required — runs entirely on ONNX Runtime (CPU)
- Falls back to color-based HSV detection if YOLO model is unavailable

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

- [x] Add YOLO-based object detection for better accuracy
- [ ] Implement voice guidance for accessibility
- [ ] Add support for custom task sequences via UI
- [ ] Persist calibration settings between sessions
- [ ] Add multi-user support with different difficulty levels

## Author

Ruijie Zheng  
February 2026

## License

MIT License - Feel free to use and modify for educational purposes.
