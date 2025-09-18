#!/bin/bash

echo "Building YOLO Headless C++ Test..."

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Run cmake
echo "Running cmake..."
cmake ..

# Build the headless version
echo "Building headless executable..."
make yolo_test_headless

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Headless executable: ./yolo_test_headless"
    echo ""
    echo "Usage examples:"
    echo "  ./yolo_test_headless                                    # Process test_image.jpg"
    echo "  ./yolo_test_headless video.mp4                          # Process video file"
    echo "  ./yolo_test_headless image.jpg output_dir               # Process image, save to output_dir"
    echo "  ./yolo_test_headless video.mp4 output_dir true 100      # Process 100 frames, save annotated frames"
    echo ""
    echo "Arguments:"
    echo "  [input_path]     - Image or video file to process (default: test_image.jpg)"
    echo "  [output_dir]     - Output directory for results (default: output)"
    echo "  [save_annotated] - Save annotated frames (true/false, default: false)"
    echo "  [max_frames]     - Maximum frames to process for video (default: all)"
else
    echo "Build failed!"
    exit 1
fi
