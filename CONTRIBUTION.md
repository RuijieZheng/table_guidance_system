# Author Contribution Statement

**Author:** Ruijie Zheng
**Date:** 2026-02-14

## Short paragraph (copy into your Google Doc)
I designed and implemented the Table Guidance System: a 5-module pipeline that senses table objects and provides AR guidance (marker detection, object detection, hand tracking, state manager, and visualizer). I integrated MediaPipe Tasks API for hand tracking and ONNX Runtime with a YOLOv5s ONNX model for object recognition, implemented homography-based table coordinates with optional ArUco markers, and authored the state machine and AR overlays. I used AI tools only to assist with coding and debugging; all architecture, design decisions, and implementation choices are my own.

## What I implemented (concise bullet list)
- System architecture, module design, and integration (see `main.py`).
- YOLOv5s ONNX inference + ONNX Runtime integration (`modules/yolo_detector.py`).
- Multi-strategy object detection and fallback logic (`modules/object_detector.py`).
- MediaPipe Tasks API hand tracking and gesture classification (`modules/hand_tracker.py`).
- ArUco marker detection + homography and screen-relative fallback (`modules/marker_detector.py`).
- Finite state machine for procedure tracking (`modules/state_manager.py`).
- Perspective-warped AR overlays and guidance UI (`modules/visualizer.py`).
- README, documentation, and test utilities; fixed dependency/runtime issues.

## Suggested wording for the SURE report (short)
"I implemented the sensing and guidance pipeline, designed the finite-state
procedure, and integrated hand and object perception (YOLO + MediaPipe).
I led the engineering work, made all major design choices (listed above),
and used AI only as a coding/debugging assistant where helpful." 

> NOTE: If the SURE submission disallows AI-written text, use the bullets above
> as prompts and rewrite them in your own words before submitting.

## Links
- GitHub: https://github.com/RuijieZheng/table_guidance_system

---

*File auto-generated in repository by GitHub Copilot.*