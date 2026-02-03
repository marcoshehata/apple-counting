# Apple Counting: Mathematical & Informatic Logic

This document details the mathematical transformations and algorithmic logic used in the detection, segmentation, and tracking pipeline.

## 1. Geometric Preprocessing: Letterbox Transform

To preserve the aspect ratio of the 1080p camera feed when feeding into the 1280x1280 square model input, we utilize a **Letterbox transformation**.

### Forward Transform:
Given an input image of size $H \times W$ and model size $S \times S$:
1.  **Scale Ratio ($r$):** $r = \min(\frac{S}{H}, \frac{S}{W})$
2.  **Unpadded Shape:** $(H', W') = (r \cdot H, r \cdot W)$
3.  **Padding:** $(dw, dh) = (\frac{S - W'}{2}, \frac{S - H'}{2})$

### Inverse Coordinate Transform:
For a detection box $[x_1, y_1, x_2, y_2]$ in model space, the original coordinates $[\hat{x}, \hat{y}]$ are:
$$\hat{x} = \frac{x - dw}{r}$$
$$\hat{y} = \frac{y - dh}{r}$$

---

## 2. Segmentation: Proto-Mask Reconstruction

The optimized FastSAM implementation uses a prototype-based segmentation approach to remain efficient on the edge.

### Mathematical Rejection:
For each detected box $B_{det}$:
1.  **IOU Matching**: We find all FastSAM internal segments $S_i$ where $IOU(B_{det}, B_{S_i}) > \tau_{min}$.
2.  **Centroid Constraint**: The segment $S_i$ is only considered valid if its centroid $C_i$ lies within $B_{det} \pm \delta$.

### Mask Reconstruction:
The mask for a selected prototype set $P$ and coefficients $C$ is computed via:
$$M = \text{Sigmoid}(C \cdot P)$$
We then apply an inverse Letterbox transform to $M$ and crop it strictly to $B_{det}$ to eliminate leakage into background objects (wood/leaves).

---

## 3. Object Tracking: Sorted Centroid-IOU

The `Tracker` maintains object identity across frames using a cost-matrix approach.

### Association Logic:
1.  **Cost Function**: A combined metric of Centroid Euclidean Distance and Bounding Box IOU.
2.  **Confirmation**: A track is promoted from **Candidate** to **Confirmed** only after it has been detected in $N$ consecutive frames (`MIN_HITS_FOR_CONFIRMATION`).
3.  **Persistence**: Confirmed tracks persist for `MAX_AGE` frames if the object is temporarily occluded.

---

## 4. Identification Synchronization

To ensure consistent IDs between detection and segmentation requests on the same frame, we implement a **Deterministic Frame Hashing**:
$$\text{ID}_{object}(F) = \text{Cache}(\text{Hash}(F), \text{TrackerID})$$
This prevents "visual jumps" in the Node-RED UI caused by asynchronous network calls.

---

## 5. Expert Data Science Logic (Final Pass)

### CLAHE Contrast Enhancement
To recover occluded apples in shadowed regions, we apply **Contrast Limited Adaptive Histogram Equalization** to the Value (V) channel of the HSV color space:
$$V_{out} = \text{CLAHE}(V_{in}, \text{tile}=8, \text{limit}=2.0)$$
This ensures local structural details are amplified for the detector without blowing out global highlights.

### Test Time Augmentation (TTA)
To increase geometric robustness, we ensemble results from two viewpoints:
- $P_{orig} = \text{Inference}(\text{Frame})$
- $P_{flip} = \text{Inference}(\text{FlipHorizontal}(\text{Frame}))$
- $P_{final} = \text{NMS}(P_{orig} \cup \text{InverseFlip}(P_{flip}))$

### Circularity Heuristic
We reject non-fruit segments (branches/leaves) using a mathematical "Roundness" score $C$:
$$C = \frac{4\pi \cdot \text{Area}}{\text{Perimeter}^2}$$
Objects are discarded if $C < 0.6$, ensuring only spherical fruits are promoted to the dashboard.
