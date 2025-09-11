# yolo_service.py
from __future__ import annotations

import json
import threading
import time
from typing import Optional, Union

import cv2
import numpy as np
from ultralytics import YOLO

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


def shutdown() -> None:
    """
    Explicitly clear the model reference.
    This helps free CUDA/TensorRT resources before process exit.
    """
    global _model
    with _lock:
        _model = None
