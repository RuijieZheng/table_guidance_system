"""
YOLO Object Detector Module
============================
Uses YOLOv5s ONNX model with ONNX Runtime for real-time object detection.
Recognizes 80 COCO classes including common desk objects:
  cup, bottle, cell phone, mouse, keyboard, laptop, book, scissors, etc.

This module replaces contour-based detection with actual object identification.
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Optional, Tuple
import onnxruntime as ort

# COCO 80-class names (YOLOv5 / YOLOv8 format)
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck",
    "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra",
    "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]

# Map common COCO class names to user-friendly display names
DISPLAY_NAMES = {
    "cell phone": "Phone",
    "mouse": "Mouse",
    "cup": "Cup",
    "bottle": "Bottle",
    "laptop": "Laptop",
    "keyboard": "Keyboard",
    "book": "Book",
    "remote": "Remote",
    "scissors": "Scissors",
    "clock": "Clock",
    "vase": "Vase",
    "bowl": "Bowl",
    "wine glass": "Glass",
    "tv": "Monitor",
    "potted plant": "Plant",
    "backpack": "Backpack",
    "handbag": "Bag",
    "chair": "Chair",
    "dining table": "Table",
    "person": "Person",
}

# Colors for different object classes (BGR)
CLASS_COLORS = {
    "cell phone": (0, 165, 255),    # Orange
    "mouse": (255, 100, 100),       # Light blue
    "cup": (0, 0, 255),             # Red
    "bottle": (255, 0, 0),          # Blue
    "laptop": (0, 200, 0),          # Green
    "keyboard": (200, 0, 200),      # Purple
    "book": (0, 200, 200),          # Yellow
    "remote": (200, 200, 0),        # Cyan
    "scissors": (128, 0, 255),      # Pink
    "person": (0, 255, 0),          # Green
}
DEFAULT_COLOR = (200, 200, 200)     # Light gray


class YOLODetector:
    """
    YOLO-based object detector using ONNX Runtime.
    Detects and identifies common objects by name.
    """

    def __init__(self, model_path: str = None, confidence_threshold: float = 0.35,
                 nms_threshold: float = 0.45, input_size: int = 640):
        """
        Initialize YOLO detector.

        Args:
            model_path: Path to YOLOv5s ONNX model file
            confidence_threshold: Min confidence to accept a detection
            nms_threshold: Non-maximum suppression IoU threshold
            input_size: Input image size for the model (640 for YOLOv5s)
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.session = None
        self.input_name = None
        self.use_fp16 = False

        # Find model file
        if model_path is None:
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base_dir, "models", "yolov5s.onnx")

        if os.path.exists(model_path):
            self._load_model(model_path)
        else:
            print(f"[YOLO] Model not found at: {model_path}")
            print(f"[YOLO] Download it with:")
            print(f"  curl -L -o models/yolov5s.onnx https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.onnx")

    def _load_model(self, model_path: str):
        """Load the ONNX model."""
        try:
            # Use CPU provider (works everywhere)
            self.session = ort.InferenceSession(
                model_path,
                providers=['CPUExecutionProvider']
            )
            inp = self.session.get_inputs()[0]
            self.input_name = inp.name
            # Check if model expects float16
            self.use_fp16 = (inp.type == 'tensor(float16)')
            print(f"[YOLO] Loaded model: {os.path.basename(model_path)}")
            print(f"[YOLO]   Input: {inp.shape}, dtype={'fp16' if self.use_fp16 else 'fp32'}")
            print(f"[YOLO]   Classes: {len(COCO_CLASSES)} COCO classes")
        except Exception as e:
            print(f"[YOLO] Failed to load model: {e}")
            self.session = None

    @property
    def is_available(self) -> bool:
        """Check if YOLO model is loaded and ready."""
        return self.session is not None

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Detect objects in a frame.

        Args:
            frame: BGR image from camera

        Returns:
            List of detections, each with:
                - 'class_name': COCO class name (e.g., 'cell phone')
                - 'display_name': User-friendly name (e.g., 'Phone')
                - 'confidence': Detection confidence (0-1)
                - 'bbox': (x, y, w, h) bounding box in original image coords
                - 'center': (cx, cy) center point in original image coords
                - 'color_bgr': Suggested display color (BGR)
        """
        if not self.is_available:
            return []

        orig_h, orig_w = frame.shape[:2]

        # Preprocess: letterbox resize to 640x640
        blob, ratio, (dw, dh) = self._preprocess(frame)

        # Run inference
        dtype = np.float16 if self.use_fp16 else np.float32
        blob = blob.astype(dtype)
        outputs = self.session.run(None, {self.input_name: blob})
        predictions = outputs[0]  # shape: (1, 25200, 85)

        # Postprocess: extract detections
        detections = self._postprocess(
            predictions, orig_w, orig_h, ratio, dw, dh
        )

        return detections

    def _preprocess(self, frame: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Preprocess image for YOLO: letterbox resize + normalize.

        Returns:
            blob: (1, 3, 640, 640) normalized image
            ratio: resize ratio
            (dw, dh): padding offsets
        """
        h, w = frame.shape[:2]
        size = self.input_size

        # Calculate resize ratio (maintain aspect ratio)
        ratio = min(size / h, size / w)
        new_h, new_w = int(h * ratio), int(w * ratio)

        # Resize
        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

        # Pad to square
        dw = (size - new_w) // 2
        dh = (size - new_h) // 2
        padded = np.full((size, size, 3), 114, dtype=np.uint8)
        padded[dh:dh + new_h, dw:dw + new_w] = resized

        # BGR -> RGB, HWC -> CHW, normalize to 0-1
        blob = padded[:, :, ::-1].transpose(2, 0, 1)  # BGR->RGB, HWC->CHW
        blob = np.ascontiguousarray(blob, dtype=np.float32) / 255.0
        blob = blob[np.newaxis, ...]  # Add batch dimension

        return blob, ratio, (dw, dh)

    def _postprocess(self, predictions: np.ndarray,
                     orig_w: int, orig_h: int,
                     ratio: float, dw: int, dh: int) -> List[Dict]:
        """
        Post-process YOLO output: filter by confidence, apply NMS, scale boxes.
        """
        # predictions shape: (1, 25200, 85)
        # Format: [cx, cy, w, h, obj_conf, cls0, cls1, ..., cls79]
        preds = predictions[0].astype(np.float32)  # (25200, 85)

        # Filter by objectness confidence
        obj_conf = preds[:, 4]
        mask = obj_conf > self.confidence_threshold
        preds = preds[mask]

        if len(preds) == 0:
            return []

        # Get class scores = obj_conf * class_conf
        class_scores = preds[:, 5:] * preds[:, 4:5]  # (N, 80)
        class_ids = np.argmax(class_scores, axis=1)   # (N,)
        max_scores = class_scores[np.arange(len(class_scores)), class_ids]  # (N,)

        # Filter by class confidence
        conf_mask = max_scores > self.confidence_threshold
        preds = preds[conf_mask]
        class_ids = class_ids[conf_mask]
        max_scores = max_scores[conf_mask]

        if len(preds) == 0:
            return []

        # Convert center-format to corner-format boxes
        # and scale back to original image coordinates
        boxes = []
        for i, pred in enumerate(preds):
            cx, cy, bw, bh = pred[:4]
            # Remove letterbox padding and scale
            x1 = (cx - bw / 2 - dw) / ratio
            y1 = (cy - bh / 2 - dh) / ratio
            x2 = (cx + bw / 2 - dw) / ratio
            y2 = (cy + bh / 2 - dh) / ratio
            # Clip to image bounds
            x1 = max(0, min(x1, orig_w))
            y1 = max(0, min(y1, orig_h))
            x2 = max(0, min(x2, orig_w))
            y2 = max(0, min(y2, orig_h))
            boxes.append([x1, y1, x2 - x1, y2 - y1])  # x, y, w, h

        boxes = np.array(boxes, dtype=np.float32)
        scores = max_scores.astype(np.float32)

        # Apply NMS using OpenCV
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(), scores.tolist(),
            self.confidence_threshold, self.nms_threshold
        )

        detections = []
        if len(indices) > 0:
            # OpenCV NMS returns different formats depending on version
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()

            for idx in indices:
                x, y, w, h = boxes[idx]
                class_id = class_ids[idx]
                conf = float(scores[idx])
                class_name = COCO_CLASSES[class_id] if class_id < len(COCO_CLASSES) else "unknown"

                cx = int(x + w / 2)
                cy = int(y + h / 2)

                detections.append({
                    'class_name': class_name,
                    'display_name': DISPLAY_NAMES.get(class_name, class_name.title()),
                    'class_id': int(class_id),
                    'confidence': conf,
                    'bbox': (int(x), int(y), int(w), int(h)),
                    'center': (cx, cy),
                    'color_bgr': CLASS_COLORS.get(class_name, DEFAULT_COLOR),
                })

        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d['confidence'], reverse=True)
        return detections

    def draw_detections(self, frame: np.ndarray,
                        detections: List[Dict],
                        draw_labels: bool = True) -> np.ndarray:
        """
        Draw YOLO detections on frame.

        Args:
            frame: BGR image
            detections: List of detection dicts from detect()
            draw_labels: Whether to draw class name + confidence

        Returns:
            Frame with bounding boxes and labels drawn
        """
        output = frame.copy()

        for det in detections:
            x, y, w, h = det['bbox']
            color = det['color_bgr']
            name = det['display_name']
            conf = det['confidence']

            # Draw bounding box
            cv2.rectangle(output, (x, y), (x + w, y + h), color, 2)

            if draw_labels:
                # Draw label background
                label = f"{name} {conf:.0%}"
                (tw, th), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
                )
                cv2.rectangle(output, (x, y - th - 8), (x + tw + 4, y), color, -1)
                cv2.putText(output, label, (x + 2, y - 5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return output

    def find_object(self, detections: List[Dict], target_name: str) -> Optional[Dict]:
        """
        Find a specific object type in the detections.

        Args:
            detections: List of detection dicts
            target_name: Object name to search for (case-insensitive).
                         Checks both COCO class_name and display_name.

        Returns:
            Best matching detection dict, or None
        """
        target_lower = target_name.lower()
        matches = []
        for det in detections:
            if (det['class_name'].lower() == target_lower or
                det['display_name'].lower() == target_lower):
                matches.append(det)

        if matches:
            # Return highest confidence match
            return max(matches, key=lambda d: d['confidence'])
        return None

    def get_desk_objects(self, detections: List[Dict]) -> List[Dict]:
        """
        Filter detections to only include typical desk/table objects.

        Returns:
            Filtered list of detections
        """
        desk_classes = {
            'cup', 'bottle', 'wine glass', 'bowl', 'cell phone', 'mouse',
            'keyboard', 'laptop', 'remote', 'book', 'scissors', 'clock',
            'vase', 'potted plant', 'tv'
        }
        return [d for d in detections if d['class_name'] in desk_classes]
