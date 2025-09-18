YOLO Service (Python ↔ C++ Bridge)

A comprehensive YOLO object detection service that provides both Python and C++ interfaces for running YOLO inference on images and videos using TensorRT-optimized engines. The service supports multiple input formats and provides detailed detection results with bounding boxes and classifications.

## Project Structure

- **Python Service** (`yolo_service.py`): Core YOLO service with Python API
- **C++ Applications**: Three different C++ executables for various use cases
- **Python Examples**: Standalone and video processing examples
- **Build System**: CMake configuration for C++ compilation

## Features

- **One-time initialization**: Load the YOLO TensorRT engine only once
- **Multiple input formats**: Support for NumPy arrays, encoded image bytes, and RGBA data
- **Detailed detection results**: Returns bounding boxes, class names, and confidence scores
- **Thread-safe**: Guarded by locks around model inference
- **C++ Integration**: Full pybind11 integration for C++ applications
- **Video processing**: Support for both real-time and batch video processing
- **Headless operation**: C++ applications can run without GUI dependencies
- **Graceful shutdown**: Explicitly release model and GPU resources

## Installation

### Prerequisites

1. **Python Dependencies**:
   ```bash
   pip install ultralytics opencv-python numpy
   ```

2. **System Requirements**:
   - CUDA drivers installed for your target platform
   - TensorRT library
   - OpenCV development libraries
   - CMake 3.12 or higher
   - C++17 compatible compiler

3. **YOLO Engine File**:
   - Generate a TensorRT engine file: `yolo export model=yolov8n.pt format=engine`
   - Place the `.engine` file in your project directory

### Building C++ Applications

1. **Navigate to C++ directory**:
   ```bash
   cd c++/
   ```

2. **Create build directory**:
   ```bash
   mkdir build && cd build
   ```

3. **Configure with CMake**:
   ```bash
   cmake ..
   ```

4. **Build the project**:
   ```bash
   make
   ```

5. **Copy Python service**:
   ```bash
   cp ../yolo_service.py .
   ```

This will create three executables:
- `yolo_test`: Interactive test with camera/image input
- `yolo_test_headless`: Batch processing for images/videos
- `yolo_image_processor`: RGBA image processing with detailed results

## Python API

### Basic Usage

```python
from yolo_service import init_engine, detect_image, detect_rgba_image, shutdown, get_last_error

# 1. Initialize model once
ok = init_engine("./yolo11n.engine", imgsz=640, conf=0.25, device=0)
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

# 4. Run detection on RGBA data with detailed results
rgba_image = cv2.cvtColor(img, cv2.COLOR_BGR2RGBA)
rgba_flat = rgba_image.flatten()
result_json = detect_rgba_image(rgba_flat, rgba_image.shape[1], rgba_image.shape[0])
print(result_json)   # Detailed results with bounding boxes

# 5. Shutdown when done
shutdown()
```

### Available Functions

- `init_engine(engine_path, imgsz=640, conf=0.25, device=0, do_warmup=True)`: Initialize the YOLO model
- `detect_image(image)`: Basic detection returning count and timing
- `detect_rgba_image(rgba_data, width, height)`: Detailed detection with bounding boxes
- `get_last_error()`: Get the last error message
- `shutdown()`: Clean up resources

## JSON Output Format

### Basic Detection (`detect_image`)

**On success:**
```json
{
  "ok": true,
  "count": 3,
  "time_ms": 7.532
}
```

**On error:**
```json
{
  "ok": false,
  "error": "model_not_initialized"
}
```

### Detailed Detection (`detect_rgba_image`)

**On success:**
```json
{
  "ok": true,
  "count": 2,
  "time_ms": 8.245,
  "detections": [
    {
      "class_id": 0,
      "class_name": "person",
      "confidence": 0.856,
      "bbox": {
        "x1": 100.5,
        "y1": 50.2,
        "x2": 200.8,
        "y2": 300.1,
        "width": 100.3,
        "height": 249.9
      }
    },
    {
      "class_id": 2,
      "class_name": "car",
      "confidence": 0.734,
      "bbox": {
        "x1": 300.0,
        "y1": 150.0,
        "x2": 500.0,
        "y2": 400.0,
        "width": 200.0,
        "height": 250.0
      }
    }
  ]
}
```

## C++ Applications

The project includes three C++ executables, each designed for different use cases:

### 1. Interactive Test (`yolo_test`)

**Purpose**: Interactive testing with camera or image input
**Usage**: `./yolo_test`

```cpp
// Basic usage example
YoloService yolo;
yolo.init("./yolo11n.engine", 640, 0.25, 0);

cv::Mat frame = cv::imread("test.jpg");
std::string result = yolo.detect_cv_image(frame);
std::cout << "Detection result: " << result << std::endl;
```

**Features**:
- Real-time camera input
- Image file processing
- Interactive controls (press 'q' to quit, 's' to save frame)
- Performance timing

### 2. Headless Batch Processor (`yolo_test_headless`)

**Purpose**: Batch processing of images and videos without GUI
**Usage**: 
```bash
./yolo_test_headless <input_path> [output_dir] [save_annotated] [max_frames]
```

**Examples**:
```bash
# Process single image
./yolo_test_headless image.jpg output true

# Process video with frame limit
./yolo_test_headless video.mp4 output true 100

# Process all frames in video
./yolo_test_headless video.mp4 output false
```

**Features**:
- Batch image processing
- Video processing with progress tracking
- Optional annotated frame saving
- Frame count limiting
- Performance statistics

### 3. RGBA Image Processor (`yolo_image_processor`)

**Purpose**: Advanced image processing with detailed detection results
**Usage**: 
```bash
./yolo_image_processor <input_image> <output_image> [engine_path] [output_dir]
```

**Example**:
```bash
./yolo_image_processor input.jpg result.jpg ./yolo11n.engine output
```

**Features**:
- RGBA image support
- Detailed bounding box results
- JSON output for debugging
- Automatic image format conversion
- Comprehensive detection data

### C++ Integration (via pybind11)

**Basic wrapper class**:

```cpp
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <opencv2/opencv.hpp>
#include <string>

namespace py = pybind11;

class YoloService {
private:
    py::scoped_interpreter guard{};
    py::object mod, init_fn, detect_fn, shutdown_fn, err_fn;
    bool initialized = false;

public:
    YoloService() {
        mod = py::module_::import("yolo_service");
        init_fn = mod.attr("init_engine");
        detect_fn = mod.attr("detect_image");
        shutdown_fn = mod.attr("shutdown");
        err_fn = mod.attr("get_last_error");
    }

    bool init(const std::string& engine, int imgsz = 640, float conf = 0.25, int device = 0) {
        bool result = init_fn(engine, imgsz, conf, device, true).cast<bool>();
        initialized = result;
        return result;
    }

    std::string detect_cv_image(cv::Mat& image) {
        if (!initialized) {
            return R"({"ok": false, "error": "not_initialized"})";
        }
        
        py::array_t<uint8_t> arr(
            {image.rows, image.cols, image.channels()},
            image.data
        );
        
        return detect_fn(arr).cast<std::string>();
    }

    void shutdown() {
        if (initialized) {
            shutdown_fn();
            initialized = false;
        }
    }
};
```

## Python Examples

### 1. Standalone Image Processing (`run_yolo_standalone.py`)

**Purpose**: Complete image processing workflow with RGBA support
**Usage**: `python run_yolo_standalone.py`

**Features**:
- Load and convert images to RGBA
- Run YOLO detection with detailed results
- Render bounding boxes on images
- Save annotated results and JSON data

### 2. Video Processing (`run_yolo_video.py`)

**Purpose**: Real-time video processing with GUI controls
**Usage**: `python run_yolo_video.py`

**Features**:
- Video file processing
- Real-time FPS display
- Interactive controls (pause, restart, save frames)
- Headless mode support
- Progress tracking

### 3. Example Usage (`example_usage.py`)

**Purpose**: Demonstrates complete workflow with RGBA detection
**Usage**: `python example_usage.py`

**Features**:
- Step-by-step image processing
- RGBA conversion and detection
- Bounding box rendering
- JSON result saving

## Typical Usage Flow

### Python
1. Call `init_engine()` once at startup
2. Repeatedly call `detect_image()` or `detect_rgba_image()` with images
3. Parse JSON results for detection data
4. Call `shutdown()` when finished

### C++
1. Initialize `YoloService` object
2. Call `init()` method with engine path
3. Repeatedly call `detect_cv_image()` with OpenCV Mat objects
4. Parse JSON results
5. Call `shutdown()` when finished

## Important Notes

- **Input Formats**: Supports NumPy arrays (HxWx3 uint8, BGR), encoded image bytes, and RGBA data
- **Engine Requirements**: Must be compiled with TensorRT (e.g., `yolo export model=yolov8n.pt format=engine`)
- **Thread Safety**: Python GIL + internal lock ensures safety but serializes inference
- **Performance**: For parallelism, consider multiple processes each with its own model
- **Memory Management**: Always call `shutdown()` to properly release GPU resources

## Troubleshooting

### Common Issues

1. **"model_not_initialized" error**: Ensure `init_engine()` was called successfully
2. **CUDA/TensorRT errors**: Verify CUDA drivers and TensorRT installation
3. **Import errors**: Ensure all Python dependencies are installed
4. **Build errors**: Check CMake configuration and compiler compatibility

### Performance Tips

- Use TensorRT engines for optimal performance
- Consider batch processing for multiple images
- Monitor GPU memory usage during processing
- Use appropriate confidence thresholds for your use case
