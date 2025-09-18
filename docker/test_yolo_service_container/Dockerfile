# Use Ubuntu as the base image
FROM ubuntu:20.04

# Install dependencies (OpenCV, curl, build tools)
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libcurl4-openssl-dev \
    git \
    g++ \
    && apt-get clean

# Create a directory for the app
WORKDIR /app

# Copy the C++ code into the container
COPY test_caller.cpp /app/test_caller.cpp

# Build the C++ app
RUN g++ test_caller.cpp -o test_caller `pkg-config --cflags --libs opencv4` -lcurl

# Set the default command to run the C++ program
CMD ["./test_caller", "test_image.jpg"]  # Replace with your image/video path
