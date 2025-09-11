YOLO Service (Python ↔ C++ Bridge)

This module provides a lightweight Python interface for running YOLO inference on images using a TensorRT-optimized engine.
It is designed to be called from a C++ application (via pybind11 or the Python C API).

Features

One-time initialization: load the YOLO TensorRT engine only once.

Fast detection calls: run inference on NumPy arrays (raw frames) or encoded image bytes (e.g., JPEG/PNG).

Thread-safe: guarded by a lock around model inference.

Minimal JSON output: returns only the number of detected objects and processing time.

Graceful shutdown: explicitly release model and GPU resources.

Installation

Install dependencies:

pip install ultralytics opencv-python numpy


Ensure you have TensorRT and CUDA drivers installed for your target platform.

Place the file yolo_service.py in your project.

Python API
from yolo_service import init_engine, detect_image, shutdown, get_last_error

# 1. Initialize model once
ok = init_engine("/root/yolo/yolo11n_fp16.engine", imgsz=640, conf=0.25, device=0)
if not ok:
    print("Init failed:", get_last_error())
    exit(1)

# 2. Run detection on an OpenCV image
import cv2
img = cv2.imread("test.jpg")
result_json = detect_image(img)
print(result_json)   # e.g. {"ok": true, "count": 3, "time_ms": 7.532}

# 3. Run detection on encoded image bytes
with open("test.jpg", "rb") as f:
    img_bytes = f.read()
result_json = detect_image(img_bytes)

# 4. Shutdown when done
shutdown()

JSON Output Format

On success:

{
  "ok": true,
  "count": 3,
  "time_ms": 7.532
}


On error:

{
  "ok": false,
  "error": "model_not_initialized"
}

C++ Integration (via pybind11)

Example wrapper class:

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <string>
namespace py = pybind11;

struct YoloService {
    py::scoped_interpreter guard{};
    py::object mod, init_fn, detect_fn, shutdown_fn, err_fn;

    YoloService() {
        mod = py::module_::import("yolo_service");
        init_fn = mod.attr("init_engine");
        detect_fn = mod.attr("detect_image");
        shutdown_fn = mod.attr("shutdown");
        err_fn = mod.attr("get_last_error");
    }

    bool init(const std::string& engine) {
        return init_fn(engine, 640, 0.25, 0).cast<bool>();
    }

    std::string detect(uint8_t* data, int h, int w, int stride_bytes) {
        ssize_t shape[3] = {h, w, 3};
        ssize_t strides[3] = {stride_bytes, 3, 1};
        py::capsule free_when_done(nullptr, [](void*){});
        py::array_t<uint8_t> arr({shape[0], shape[1], shape[2]}, strides, data, free_when_done);
        return detect_fn(arr).cast<std::string>();
    }

    void shutdown() { shutdown_fn(); }

    std::string last_error() { return err_fn().cast<std::string>(); }
};

Typical Flow (C++ or Python)

Call init_engine() once at startup.

Repeatedly call detect_image() with images.

When finished, call shutdown().

Notes

Only NumPy arrays (HxWx3 uint8, BGR) or encoded image bytes are accepted.

Engine must be compiled with TensorRT (e.g., yolo export model=yolov8n.pt format=engine).

For multi-threaded use: Python GIL + internal lock ensures safety but serializes inference. For parallelism, consider multiple processes each with its own model.

Do you want me to also add a section about running it as a microservice (FastAPI/Flask) so your C++ could just send HTTP requests instead of embedding Python?
