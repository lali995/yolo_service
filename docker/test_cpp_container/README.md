# YOLO C++ Test Container

This Docker container allows you to run the C++ YOLO test (`test_yolo_headless.cpp`) in a containerized environment.

## Features

- Multi-stage build for optimized container size
- Pre-built C++ executables with pybind11 integration
- OpenCV and Python dependencies included
- Support for both image and video processing
- Volume mounts for input/output files

## Quick Start

1. **Build the container:**
   ```bash
   cd docker/test_cpp_container
   ./build.sh
   ```

2. **Run with default test image:**
   ```bash
   docker-compose up yolo-cpp-test
   ```

3. **Run with custom image:**
   ```bash
   docker-compose run --rm yolo-cpp-test /app/yolo_test_headless /app/input/your_image.jpg /app/output false -1
   ```

## Usage Examples

### Image Processing
```bash
# Process a single image
docker-compose run --rm yolo-cpp-test /app/yolo_test_headless /app/input/test_image.jpg /app/output false -1

# Process image with annotations saved
docker-compose run --rm yolo-cpp-test /app/yolo_test_headless /app/input/test_image.jpg /app/output true -1
```

### Video Processing
```bash
# Process video (first 100 frames)
docker-compose --profile video up yolo-cpp-video

# Process video with custom parameters
docker-compose run --rm yolo-cpp-test /app/yolo_test_headless /app/input/test_video.mp4 /app/output true 50
```

### Interactive Shell
```bash
# Get an interactive shell in the container
docker-compose run --rm yolo-cpp-test /bin/bash
```

## Command Line Arguments

The `yolo_test_headless` executable accepts the following arguments:

1. `input_path` - Path to input image or video file
2. `output_dir` - Directory to save output files
3. `save_annotated` - "true"/"1" to save annotated frames, "false"/"0" otherwise
4. `max_frames` - Maximum number of frames to process (-1 for all frames)

## Directory Structure

```
docker/test_cpp_container/
├── Dockerfile              # Multi-stage Docker build
├── docker-compose.yml      # Docker Compose configuration
├── build.sh               # Build script
├── README.md              # This file
├── input/                 # Mount point for input files
│   └── test_image.jpg     # Sample test image
└── output/                # Mount point for output files
```

## Requirements

- Docker and Docker Compose
- YOLO engine file (`yolo11n.engine`) in the project root
- Input images/videos in the `input/` directory

## Notes

- The container expects the YOLO engine file to be available at `/app/yolo11n.engine`
- Input files should be placed in the `input/` directory
- Output files will be saved to the `output/` directory
- The container uses a dummy YOLO engine for testing - replace with your actual engine file

## Troubleshooting

1. **Permission issues:** Make sure the build script is executable:
   ```bash
   chmod +x build.sh
   ```

2. **Missing engine file:** Ensure `yolo11n.engine` exists in the project root

3. **Input file not found:** Check that your input file is in the `input/` directory

4. **Build failures:** Check Docker logs for detailed error messages
