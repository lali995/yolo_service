# Multi-stage build for C++ YOLO test container
FROM ubuntu:22.04 as builder

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    pkg-config \
    libopencv-dev \
    python3 \
    python3-pip \
    python3-dev \
    python3-numpy \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    pybind11 \
    opencv-python \
    ultralytics \
    numpy

# Set working directory
WORKDIR /app

# Copy source files
COPY c++/ /app/c++/
COPY yolo_service.py /app/

# Build the C++ application
WORKDIR /app/c++
RUN mkdir -p build && cd build && \
    cmake .. && \
    make -j$(nproc)

# Runtime stage
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    libopencv-dev \
    python3 \
    python3-pip \
    python3-numpy \
    libpython3-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
RUN pip3 install --no-cache-dir \
    pybind11 \
    opencv-python \
    ultralytics \
    numpy

# Create app directory
WORKDIR /app

# Copy Python service and built binaries
COPY yolo_service.py /app/
COPY --from=builder /app/c++/build/yolo_test_headless /app/
COPY --from=builder /app/c++/build/yolo_test /app/
COPY --from=builder /app/c++/build/yolo_image_processor /app/

# Create directories for input/output
RUN mkdir -p /app/input /app/output

# Set permissions
RUN chmod +x /app/yolo_test_headless /app/yolo_test /app/yolo_image_processor

# Default command
CMD ["/app/yolo_test_headless", "test_image.jpg", "output", "false", "-1"]
