# Apple Counting: Technical Documentation

This document provides an overview of the technical architecture, API endpoints, and Jetson-specific optimizations implemented for the Apple Counting project.

## 1. System Architecture

The system is designed as a modular high-performance vision pipeline running on **NVIDIA Jetson Thor (Blackwell)**. It utilizes TensorRT for ultra-low latency inference.

### Core Components:
- **`app.py`**: FastAPI server handling REST endpoints for detection and segmentation.
- **`utils/detector.py`**: Direct abstraction for YOLO-World and FastSAM inference using TensorRT.
- **`utils/trt_worker.py`**: Low-level TensorRT engine management and CUDA memory handling.
- **`utils/tracker.py`**: IOU and Centroid-based object tracking.
- **`utils/config.py`**: Centralized configuration management via `.env`.

---

## 2. API Endpoints

### `POST /detect`
Used by the detection node in Node-RED.
- **Input**: Multipart form-data with `image` (JPEG/PNG) and optional `classes`/`conf_threshold`.
- **Output**: JSON containing bounding boxes, visual tracking IDs, and current counts.
- **Optimization**: Utilizes a **ResultCache** based on image hashing to synchronise data with subsequent segmentation calls.

### `POST /segment`
Used by the segmentation node in Node-RED.
- **Input**: Multipart form-data with `image` (JPEG/PNG).
- **Output**: JSON containing segmentation polygons matched to the tracker IDs.
- **Logic**: Filters for confirmed apples and candidates meeting the `SEGMENTATION_MIN_CONF` threshold.

### `GET /reset`
Resets the object tracker state.

---

## 3. Jetson Thor Optimizations

- **TensorRT 10.x**: Native integration using `execute_async_v3` for Blackwell GPU efficiency.
- **FP16 Precision**: All models are quantized to Half-Precision (FP16) for a 2-3x speedup on Jetson hardware with minimal accuracy loss.
- **Zero-Copy Memory**: Managed CUDA buffer allocations for input/output to minimize host-to-device latency.
- **Image Hashing**: Optimized `xxhash` to prevent redundant re-detections during segmentation requests.

---

## 4. Operational Guide

### Requirements
- JetPack 6.x+ (with TensorRT 10+)
- Python 3.12+
- CUDA Driver & Toolkit

### Startup
Run the optimized server using the provided shell script:
```bash
./start_optimized.sh
```

### Configuration
Adjust sensitivity and class targets in the `.env` file. Currently optimized for **Red Apples** with **1280px resolution**.
