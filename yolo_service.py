# yolo_service.py
from __future__ import annotations

import json
import threading
import time
from typing import Optional, Union, List, Dict, Any

import cv2
import numpy as np
from ultralytics import YOLO

# COCO class names for YOLO
COCO_CLASSES = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
    'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
    'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
    'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
    'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]

# -----------------------------
# Module-level state
# -----------------------------
_model: Optional[YOLO] = None
_cfg = {"imgsz": 640, "conf": 0.25, "device": 0}
_lock = threading.Lock()
_last_error: str = ""


def init_engine(
    engine_path: str,
    imgsz: int = 640,
    conf: float = 0.25,
    device: Union[int, str] = 0,
    do_warmup: bool = True,
) -> bool:
    """
    Initialize the TensorRT-backed YOLO model once. Returns True on success.
    """
    global _model, _cfg, _last_error
    try:
        with _lock:
            _cfg.update({"imgsz": int(imgsz), "conf": float(conf), "device": device})
            _model = YOLO(engine_path)  # load the .engine
            if do_warmup:
                # Quick warmup on dummy input (to allocate CUDA/TensorRT context)
                dummy = np.zeros((32, 32, 3), dtype=np.uint8)
                _model.predict(
                    source=dummy,
                    imgsz=_cfg["imgsz"],
                    conf=_cfg["conf"],
                    device=_cfg["device"],
                    verbose=False,
                )
        return True
    except Exception as e:
        _last_error = f"init_engine failed: {e}"
        _model = None
        return False


def get_last_error() -> str:
    """Return the last error message (empty if none)."""
    return _last_error


def _to_bgr_ndarray(image: Union[np.ndarray, bytes, bytearray, memoryview]) -> np.ndarray:
    """
    Accepts:
      - NumPy ndarray HxWx3 uint8, BGR
      - Encoded image bytes (JPEG/PNG) -> decoded to BGR ndarray
    """
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("NumPy array must be HxWx3 uint8 (BGR).")
        return np.ascontiguousarray(image)

    if isinstance(image, (bytes, bytearray, memoryview)):
        buf = np.frombuffer(image, dtype=np.uint8)
        frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Failed to decode image bytes.")
        return frame

    raise TypeError("image must be a NumPy array or encoded image bytes.")


def detect_image(image: Union[np.ndarray, bytes, bytearray, memoryview]) -> str:
    """
    Run detection on a single image (NumPy array or encoded bytes).
    Returns a JSON string with only the number of detected objects:
        {"ok": true, "count": <int>, "time_ms": <float>}
    On error:
        {"ok": false, "error": "<message>"}
    """
    global _last_error

    if _model is None:
        return json.dumps({"ok": False, "error": "model_not_initialized"})

    try:
        frame = _to_bgr_ndarray(image)
        t0 = time.time()
        with _lock:
            res = _model.predict(
                source=frame,
                imgsz=_cfg["imgsz"],
                conf=_cfg["conf"],
                device=_cfg["device"],
                verbose=False,
            )[0]
        t1 = time.time()

        count = int(len(res.boxes)) if getattr(res, "boxes", None) is not None else 0

        return json.dumps(
            {
                "ok": True,
                "count": count,
                "time_ms": round((t1 - t0) * 1000.0, 3),
            }
        )

    except Exception as e:
        _last_error = f"detect_image failed: {e}"
        return json.dumps({"ok": False, "error": _last_error})


def _rgba_to_bgr(rgba_data: Union[np.ndarray, bytes, bytearray, memoryview], width: int, height: int) -> np.ndarray:
    """
    Convert RGBA data to BGR format for YOLO processing.
    
    Args:
        rgba_data: Raw RGBA data as bytes or numpy array
        width: Image width
        height: Image height
    
    Returns:
        BGR numpy array (HxWx3 uint8)
    """
    if isinstance(rgba_data, (bytes, bytearray, memoryview)):
        # Convert bytes to numpy array
        rgba_array = np.frombuffer(rgba_data, dtype=np.uint8)
        rgba_array = rgba_array.reshape((height, width, 4))
    elif isinstance(rgba_data, np.ndarray):
        rgba_array = rgba_data.reshape((height, width, 4))
    else:
        raise TypeError("rgba_data must be bytes, bytearray, memoryview, or numpy array")
    
    # Convert RGBA to BGR (OpenCV format)
    bgr_array = cv2.cvtColor(rgba_array, cv2.COLOR_RGBA2BGR)
    return np.ascontiguousarray(bgr_array)


def _extract_detection_data(result) -> List[Dict[str, Any]]:
    """
    Extract detection data from YOLO result.
    
    Args:
        result: YOLO detection result
    
    Returns:
        List of detection dictionaries with bounding boxes and classifications
    """
    detections = []
    
    if result.boxes is None or len(result.boxes) == 0:
        return detections
    
    boxes = result.boxes
    for i in range(len(boxes)):
        # Get bounding box coordinates (xyxy format)
        box = boxes.xyxy[i].cpu().numpy()
        x1, y1, x2, y2 = box
        
        # Get confidence score
        conf = float(boxes.conf[i].cpu().numpy())
        
        # Get class ID and name
        cls_id = int(boxes.cls[i].cpu().numpy())
        cls_name = COCO_CLASSES[cls_id] if cls_id < len(COCO_CLASSES) else f"class_{cls_id}"
        
        detection = {
            "class_id": cls_id,
            "class_name": cls_name,
            "confidence": round(conf, 3),
            "bbox": {
                "x1": round(float(x1), 2),
                "y1": round(float(y1), 2),
                "x2": round(float(x2), 2),
                "y2": round(float(y2), 2),
                "width": round(float(x2 - x1), 2),
                "height": round(float(y2 - y1), 2)
            }
        }
        detections.append(detection)
    
    return detections


def detect_rgba_image(
    rgba_data: Union[np.ndarray, bytes, bytearray, memoryview], 
    width: int, 
    height: int
) -> str:
    """
    Run detection on raw RGBA image data.
    
    Args:
        rgba_data: Raw RGBA image data
        width: Image width
        height: Image height
    
    Returns:
        JSON string with detection results:
        {
            "ok": true,
            "count": <int>,
            "time_ms": <float>,
            "detections": [
                {
                    "class_id": <int>,
                    "class_name": <str>,
                    "confidence": <float>,
                    "bbox": {
                        "x1": <float>, "y1": <float>, "x2": <float>, "y2": <float>,
                        "width": <float>, "height": <float>
                    }
                }
            ]
        }
    """
    global _last_error

    if _model is None:
        return json.dumps({"ok": False, "error": "model_not_initialized"})

    try:
        # Convert RGBA to BGR
        frame = _rgba_to_bgr(rgba_data, width, height)
        
        # Run detection
        t0 = time.time()
        with _lock:
            res = _model.predict(
                source=frame,
                imgsz=_cfg["imgsz"],
                conf=_cfg["conf"],
                device=_cfg["device"],
                verbose=False,
            )[0]
        t1 = time.time()

        # Extract detection data
        detections = _extract_detection_data(res)
        count = len(detections)

        return json.dumps(
            {
                "ok": True,
                "count": count,
                "time_ms": round((t1 - t0) * 1000.0, 3),
                "detections": detections
            },
            indent=2
        )

    except Exception as e:
        _last_error = f"detect_rgba_image failed: {e}"
        return json.dumps({"ok": False, "error": _last_error})


def shutdown() -> None:
    """
    Explicitly clear the model reference.
    This helps free CUDA/TensorRT resources before process exit.
    """
    global _model
    with _lock:
        _model = None
