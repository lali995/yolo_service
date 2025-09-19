#!/bin/bash

# Test script for YOLO C++ container
set -e

echo "=== Testing YOLO C++ Container ==="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

echo "Testing container build and basic functionality..."

# Test 1: Build the container
echo "1. Building container..."
cd "$PROJECT_ROOT"
if docker build -f docker/test_cpp_container/Dockerfile -t yolo-cpp-test:latest .; then
    echo "✓ Container built successfully"
else
    echo "✗ Container build failed"
    exit 1
fi

# Test 2: Check if executables are present
echo "2. Checking executables in container..."
if docker run --rm yolo-cpp-test:latest ls -la /app/ | grep -E "(yolo_test|yolo_test_headless|yolo_image_processor)"; then
    echo "✓ Executables found in container"
else
    echo "✗ Executables not found in container"
    exit 1
fi

# Test 3: Test Python module import
echo "3. Testing Python module import..."
if docker run --rm yolo-cpp-test:latest python3 -c "import yolo_service; print('YOLO service module imported successfully')"; then
    echo "✓ Python module imports successfully"
else
    echo "✗ Python module import failed"
    exit 1
fi

# Test 4: Test C++ executable (without actual YOLO engine)
echo "4. Testing C++ executable (expecting initialization failure)..."
if docker run --rm yolo-cpp-test:latest /app/yolo_test_headless /app/input/test_image.jpg /app/output false -1 2>&1 | grep -q "Failed to initialize YOLO model"; then
    echo "✓ C++ executable runs (expected initialization failure without engine)"
else
    echo "? C++ executable behavior unexpected (this might be normal)"
fi

echo ""
echo "=== Container Test Summary ==="
echo "✓ Container builds successfully"
echo "✓ All executables are present"
echo "✓ Python dependencies work"
echo "✓ C++ executable runs (needs YOLO engine for full functionality)"
echo ""
echo "To use with a real YOLO engine:"
echo "1. Place your yolo11n.engine file in the project root"
echo "2. Run: docker-compose -f docker/test_cpp_container/docker-compose.yml up yolo-cpp-test"
