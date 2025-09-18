#!/bin/bash

echo "Building YOLO Image Processor..."

# Create build directory if it doesn't exist
mkdir -p build
cd build

# Run cmake
echo "Running cmake..."
cmake ..

# Build the image processor
echo "Building image processor executable..."
make yolo_image_processor

if [ $? -eq 0 ]; then
    echo "Build successful!"
    echo "Image processor executable: ./yolo_image_processor"
    echo ""
    echo "Usage:"
    echo "  ./yolo_image_processor <input_image> <output_image> [engine_path] [output_dir]"
    echo ""
    echo "Examples:"
    echo "  ./yolo_image_processor test.jpg result.jpg"
    echo "  ./yolo_image_processor input.png output.png ./yolo11n.engine output"
    echo ""
    echo "Arguments:"
    echo "  input_image  - Path to input JPG/PNG image"
    echo "  output_image - Path to save annotated result"
    echo "  engine_path  - Path to YOLO engine file (default: ./yolo11n.engine)"
    echo "  output_dir   - Output directory for additional files (default: output)"
else
    echo "Build failed!"
    exit 1
fi
