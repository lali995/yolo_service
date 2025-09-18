cmake_minimum_required(VERSION 3.12)
project(yolo_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Find Python and pybind11
find_package(pybind11 REQUIRED)

# Add executables
add_executable(yolo_test test_yolo.cpp)
add_executable(yolo_test_headless test_yolo_headless.cpp)
add_executable(yolo_image_processor yolo_image_processor.cpp)

# Link libraries
target_link_libraries(yolo_test 
    ${OpenCV_LIBS}
    pybind11::embed
)

target_link_libraries(yolo_test_headless 
    ${OpenCV_LIBS}
    pybind11::embed
)

target_link_libraries(yolo_image_processor 
    ${OpenCV_LIBS}
    pybind11::embed
)

# Include directories
target_include_directories(yolo_test PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(yolo_test_headless PRIVATE ${OpenCV_INCLUDE_DIRS})
target_include_directories(yolo_image_processor PRIVATE ${OpenCV_INCLUDE_DIRS})

# Set Python executable
set(PYTHON_EXECUTABLE /usr/bin/python3)
