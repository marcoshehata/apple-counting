# 🍎 Apple Counting System

A professional-grade, real-time object detection and tracking system designed for high-throughput orchard analysis. This project combines a **FastAPI** backend leveraging **YOLO-World** with a **Node-RED** dashboard for interactive visualization.

## 🏗️ Architecture Overview

The system is split into two primary layers:

1.  **AI Analysis Layer (Python/FastAPI)**:
    -   **Detection**: Zero-shot object detection using YOLO-World (Small/Medium/Large).
    -   **Segmentation**: High-speed refiner using **FastSAM**, prompted by bounding boxes for 1:1 synchronization.
    -   **Tracking**: Multi-object tracking via a custom 2-stage association engine (IOU + Centroid Fallback).
    -   **Quantification**: Cumulative analysis that counts the total number of unique objects across the entire video duration.
2.  **Visualization & Control Layer (Node-RED)**:
    -   **Flow Pipeline**: Manages high-speed frame capture and concurrent network dispatch.
    -   **Pipelining**: Implements asynchronous frame pipelining (concurrency: 3) to maximize throughput and minimize latency.
    -   **UI**: Real-time canvas rendering of analyzed frames and performance KPIs (FPS, Inference Latency).

## 🛠️ Installation Guide

### 1. Prerequisites
- Python 3.10+
- Node.js & Node-RED
- (Optional) CUDA-capable GPU

### 2. Backend Setup
```bash
# Clone and enter the repository
cd apple_counting

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Environment Configuration
Update the `.env` file in the root directory:
```env
MODEL_ID=yolo_world/s
PRIMARY_CLASSES=apple,red apple,ripe apple,fruit
CONFIDENCE_THRESHOLD=0.05
IOU_THRESHOLD=0.3
MAX_AGE=60
MIN_HITS_FOR_CONFIRMATION=4
FILL_ALPHA=0.5
CANDIDATE_VISIBLE_THRESHOLD=0.05
PORT=8000
HOST=0.0.0.0
```

### 4. Running the System
```bash
# 1. Start Node-RED
node-red --userDir ./nodered

# 2. Start Python Backend (in a separate terminal inside venv)
source venv/bin/activate
python3 app.py
```
-   Node-RED Editor: [http://127.0.0.1:1880](http://127.0.0.1:1880)
-   Apple Counting App: [http://127.0.0.1:1880/app](http://127.0.0.1:1880/app)

## 🧩 Modular Node-RED Components

This project introduces a set of reusable subflows located in `nodered/flows.json`.

### 1. `Video Object Detection`
- **Description**: Universal detector using YOLO-World.
- **Configurable Parameters**:
    - `API_URL`: Path to the detection endpoint.
    - `CONFIDENCE`: Detection sensitivity (0.05 - 1.0).
    - `CLASSES`: Dynamic search terms (e.g., `apple, fruit, branch`).

### 2. `Video Segmentation`
- **Description**: High-fidelity refiner using FastSAM.
- **Logic**: Use the output of the detection node to "prompt" this node for pixel-perfect masks.

## 📈 Methodology & Logic

### Detection (Zero-Shot)
We use **YOLO-World** for blazing fast zero-shot detection. The model is specifically tuned to recognize fruit without project-specific training, using descriptors like `red apple` and `ripe apple`.

### Tracking (Robust 2-Stage Association)
The `Tracker` (in `utils/tracker.py`) implements a robust association logic:
- **Stage 1: IoU Matching**: Standard overlap-based association.
- **Stage 2: Centroid Fallback**: Matches objects using centroid proximity if the bounding box shape changes significantly (e.g., during 360° rotation).
- **Promotion Rule**: Objects must persist for **4 frames** (`MIN_HITS_FOR_CONFIRMATION`) before being officially counted, filtering out noise.

### Segmentation (FastSAM Refiner)
To provide the "WOW" effect, we use a **Prompted Segmentation** architecture:
- **Logic**: Instead of running detection and segmentation in parallel (which can lead to mismatched IDs), we use the YOLO-World bounding boxes as "prompts" for the **FastSAM** model.
- **Accuracy**: FastSAM isolates the exact pixels of the apple within the prompted region, ensuring the third view is 100% synchronized with the analysis.
- **Latency**: By only segmenting objects that are actively being tracked, we maintain high throughput (~20 FPS).

## 🔍 API Overview
- `GET /health`: Returns service status and active track count.
- `POST /infer`: Primary inference endpoint. Accepts binary images and returns localized bounding boxes.
- `GET /reset`: Clears internal tracker memory for a fresh start.
