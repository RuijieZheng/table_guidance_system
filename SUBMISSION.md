Project #9 — Table Guidance System
=================================

This file contains copy-paste-ready text you can drop into the Google Doc or into the submission email. Fill the marked placeholders before sending.

---

Google Doc skeleton (paste and expand):

1. Title & One-line summary
   - Title: Table Guidance System — <Your Name>
   - One-line: Real-time AR guidance system that detects objects and hand interactions to guide a user through a 3‑step table‑setting task.

2. Short description (1 paragraph)
   - Objective: Guide a user to place 3 objects (Phone, Pen, Bottle) into predefined target zones using a live camera feed and AR overlays.

3. System architecture (half a paragraph + diagram)
   - Modules: MarkerDetector (ArUco/homography), ObjectDetector (YOLO + color/contour fallbacks), HandTracker (MediaPipe Tasks API with pinch/grab detection), StateManager (FSM for procedure), Visualizer (perspective-warped overlays).

4. Implementation highlights (bulleted)
   - YOLOv5s ONNX via ONNX Runtime (CPU) for object recognition.
   - Homography from 4 ArUco markers for table-relative coordinates.
   - MediaPipe HandLandmarker for hand state (open/pinch/grab) and `holding` detection.
   - Finite-state procedure with 3 steps defined in `config/procedure.json`.
   - Visual guidance: perspective-warped target zones, animated guidance arrows, HUD with `Hand Near Object` and `Holding: <object>`.

5. How to run (copy the commands)
   - Conda (recommended):
     ```bash
     conda create -n project python=3.11 -y
     conda activate project
     pip install -r requirements.txt
     conda run -n project --no-capture-output python main.py
     ```
   - Models: download `models/hand_landmarker.task` and `models/yolov5s.onnx` as described in the README.

6. Test procedure (what you recorded)
   - Calibration (markers present or skipped), place objects randomly, run Steps 1→3, show hand detection while picking up an object, show final completion.

7. Results / Metrics (paste your numbers)
   - Trials: __/__ successful (e.g., 5/5)
   - Avg time per trial: __ s (e.g., 28s)
   - Failure modes observed: brief occlusion causing temporary lost detection; YOLO misclassification of object X in Y% of frames.

8. Limitations and future work (2–4 bullet points)
   - Example: Improve occlusion robustness (simple tracker), add voice guidance, persist calibration between runs, support multi-object simultaneous instructions.

9. AI use disclosure (mandatory)
   - I used AI-assisted coding help during development (tools for editing and debugging only). All design decisions and implementation logic are my own. (Adapt text from CONTRIBUTION.md if needed.)

10. Links & attachments
    - GitHub repo: https://github.com/RuijieZheng/table_guidance_system
    - Demo video (Google Drive): https://drive.google.com/file/d/1AK7lB2fSblzjdCt8OvcA9mTn3GGU87ad/view?usp=drive_link
    - Config used: `config/procedure.json` (3 steps: Phone, Pen, Bottle)

---

Email template (reply-all to assignment email):

Subject: [SURE Project #9] Table Guidance System — <Your Name>

Hi Anhong / Yuxuan / Chen,

Please find my submission for Project #9 attached below.

- Google Doc (report): <paste Google Doc link here — ensure Viewer permission>
- GitHub repo: https://github.com/RuijieZheng/table_guidance_system
- Demo video: https://drive.google.com/file/d/1AK7lB2fSblzjdCt8OvcA9mTn3GGU87ad/view?usp=drive_link

Short note: This demo is an actual screen recording of the system running and completing the 3-step task. I used AI tools for programming assistance only; architectural and implementation decisions are my own (see CONTRIBUTION.md).

Thanks,

<Your Name> — <UMICH email>

CC: <anyone you must cc>

---

Submission checklist (verify before sending):
- [ ] Google Doc is set to "Anyone with the link can view".
- [ ] Demo video is shareable (Anyone with link → Viewer).
- [ ] GitHub repo is public (or add reviewer as collaborator) and includes README + config + run instructions.
- [ ] Include a short line in the report disclosing AI use in programming.
- [ ] (Optional) Attach a short GIF or screenshots in the Google Doc.

---

If you want, I can:
- Draft the Google Doc for you (I will not write the report text, only the structure and placeholders), or
- Create a GitHub release and attach the demo.mp4 (you must confirm you want the MP4 in the repo), or
- Generate a short GIF from your local `table_guidance_system.mp4` for the README (I will not push the MP4 to git).


