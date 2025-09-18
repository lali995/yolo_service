cmake_minimum_required(VERSION 3.12)
project(yolo_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Python and pybind11
find_package(pybind11 REQUIRED)

# Add executable
add_executable(yolo_test test_yolo.cpp)

# Link libraries
target_link_libraries(yolo_test 
    ${OpenCV_LIBS}
    pybind11::embed
)

# Include directories
target_include_directories(yolo_test PRIVATE ${OpenCV_INCLUDE_DIRS})

# Set Python executable
set(PYTHON_EXECUTABLE /usr/bin/python3)
