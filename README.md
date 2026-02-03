# 🍎 Apple Counting System (GPU Optimized)

A professional-grade, real-time object detection and tracking system designed for high-throughput orchard analysis. Optimized for the **NVIDIA Jetson Thor (Blackwell)** with **TensorRT 10.x**.

## 🏗️ Technical Architecture

The system implements a decoupled **Detection-then-Segmentation** architecture to ensure 100% synchronization and real-time performance.

1.  **Detection & Tracking**: YOLO-World handles spatial localization and identification.
2.  **Segmentation**: FastSAM refines bounding boxes into pixel-perfect masks using prototype reconstruction.
3.  **Precision Layer**: Post-processing filters (Hue, Circularity) eliminate environmental noise (branches, leaves).

---

## 🧠 Mathematical Framework & AI Models

### 1. YOLO-World (Open-Vocabulary Detection)
Unlike traditional YOLO models confined to fixed datasets (like COCO), **YOLO-World** leverages a **Vision-Language Model (VLM)** approach.
-   **Concept**: It encodes textual prompts (e.g., "red apple") into a latent space using a language encoder (CLIP-based).
-   **Mechanism**: The vision backbone extracts image features, which are then cross-attended with the language embeddings. This allows the model to detect *any* object described in natural language without retraining.
-   **Precision Post-Processing**: We utilize **negative prompting** (e.g., "branch") to force the vision-language head to distinguish between target fruit and background noise.

### 2. FastSAM (Segment Anything Model - Fast)
**FastSAM** is an accelerated implementation of various **Segment Anything Model (SAM)** principles, optimized for edge devices.
-   **Prompted Segmentation**: Instead of segmenting the whole image (slow), we use YOLO-World's bounding boxes as **visual prompts**.
-   **Prototype Reconstruction**: The model generates a set of binary "prototypes" and a set of masks coefficients. The final mask ($M$) is reconstructed via:
    $$M = \sigma(C \cdot P)$$
    where $C$ are the coefficients for the specific box and $P$ are the global prototypes.

### 3. Precision-First Filtering (Data Science Layer)
To achieve "Diamond Standard" precision, we apply three distinct mathematical filters:
-   **Chromatic Hue Filter**: Converts crops to HSV space and validates the median hue. Non-red objects (brown branches, green leaves) are rejected:
    $$\text{Valid} \iff \text{Hue} \in [0, 15] \cup [160, 180]$$
-   **Circularity Scoring ($C$):** Rejects elongated or amorphous objects:
    $$C = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2} > 0.65$$
-   **CLAHE (Contrast Enhancement):** Selectively recovers detail in shadowed regions by limiting histogram amplification in tiles, revealing occluded apples before detection.

---

## 🛠️ Operational Guide

### 1. Environment Configuration
Tunable parameters in `.env`:
- `PRIMARY_CLASSES`: Natural language descriptors for the AI.
- `CONFIDENCE_THRESHOLD`: Base sensitivity (default: `0.015`).
- `USE_CHROMA_FILTER`: Enables/disables Hue-based rejection.

### 2. Running the System
```bash
./start_optimized.sh
```

---

## 🔍 API Reference
- `POST /detect`: High-speed identification and tracking (33+ FPS).
- `POST /segment`: High-fidelity mask generation (synchronized with track IDs).
- `GET /reset`: Clears tracker memory and frame cache.

For deep technical details, see the **[Mathematical Logic Guide](file:///home/jetson/Desktop/fiera_rimini_2026/test/ai/apple_counting/docs/mathematical_logic.md)**.
