#!/bin/bash

# Build script for YOLO C++ test container
set -e

echo "=== Building YOLO C++ Test Container ==="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Project root: $PROJECT_ROOT"
echo "Script directory: $SCRIPT_DIR"

# Create necessary directories
mkdir -p "$SCRIPT_DIR/input"
mkdir -p "$SCRIPT_DIR/output"

# Create a sample test image if it doesn't exist
if [ ! -f "$SCRIPT_DIR/input/test_image.jpg" ]; then
    echo "Creating sample test image..."
    # Create a simple test image using ImageMagick (if available) or Python
    if command -v convert &> /dev/null; then
        convert -size 640x480 xc:white -pointsize 24 -fill black -annotate +50+240 "Test Image for YOLO" "$SCRIPT_DIR/input/test_image.jpg"
    else
        python3 -c "
import cv2
import numpy as np
img = np.ones((480, 640, 3), dtype=np.uint8) * 255
cv2.putText(img, 'Test Image for YOLO', (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
cv2.imwrite('$SCRIPT_DIR/input/test_image.jpg', img)
print('Created test image')
"
    fi
fi

# Build the Docker image
echo "Building Docker image..."
cd "$PROJECT_ROOT"
docker build -f docker/test_cpp_container/Dockerfile -t yolo-cpp-test:latest .

echo "=== Build completed successfully! ==="
echo ""
echo "Usage examples:"
echo "1. Run with default test image:"
echo "   docker-compose -f docker/test_cpp_container/docker-compose.yml up yolo-cpp-test"
echo ""
echo "2. Run with custom image:"
echo "   docker-compose -f docker/test_cpp_container/docker-compose.yml run --rm yolo-cpp-test /app/yolo_test_headless /app/input/your_image.jpg /app/output false -1"
echo ""
echo "3. Run with video processing:"
echo "   docker-compose -f docker/test_cpp_container/docker-compose.yml --profile video up yolo-cpp-video"
echo ""
echo "4. Interactive shell:"
echo "   docker-compose -f docker/test_cpp_container/docker-compose.yml run --rm yolo-cpp-test /bin/bash"
