from typing import Optional, Any, List
import numpy as np
import cv2
import supervision as sv
import torch
import os
from .config import config

# Import optimized TRT worker
try:
    from .trt_worker import TRTModel
    from ultralytics.utils import ops
    HAS_TRT = True
except ImportError:
    print("[!] TRT Worker or dependencies missing. Falling back to CPU.")
    HAS_TRT = False

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114)):
    """Resizes and pads image to new_shape with letterbox."""
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return img, (r, r), (dw, dh)

class AppleDetector:
    """
    Optimized Apple Detector using TensorRT on Jetson GPU.
    Uses Letterbox preprocessing for maximum accuracy.
    """
    def __init__(self):
        self.use_trt = config.USE_TENSORRT and HAS_TRT
        
        if self.use_trt and os.path.exists(config.ENGINE_PATH):
            print(f"[*] Initializing Optimized Detector (TRT): {config.ENGINE_PATH}")
            self.trt_model = TRTModel(config.ENGINE_PATH)
            self.imgsz = config.DETECTIONS_IMGSZ
        else:
            print(f"[*] Initializing Baseline YOLO Detector (CPU): {config.MODEL_ID}")
            from inference.models.yolo_world.yolo_world import YOLOWorld
            self.model = YOLOWorld(model_id=config.MODEL_ID)
            self.use_trt = False

    def infer(self, frame: np.ndarray, confidence: Optional[float] = None) -> sv.Detections:
        conf_thres = confidence if confidence is not None else config.CONFIDENCE_THRESHOLD
        
        if not self.use_trt:
            results = self.model.infer(frame, text=config.TARGET_CLASSES, confidence=conf_thres)
            return sv.Detections.from_inference(results)

        # --- Expert Data Science Preprocessing ---
        proc_frame = frame.copy()
        if config.USE_CLAHE:
            hsv = cv2.cvtColor(proc_frame, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            v = clahe.apply(v)
            hsv = cv2.merge((h, s, v))
            proc_frame = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        def run_pass(img_in, flip=False):
            if flip:
                img_in = cv2.flip(img_in, 1)
            
            # Preprocess
            input_img, ratio, pad = letterbox(img_in, (self.imgsz, self.imgsz))
            input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
            input_img = input_img.astype(np.float32) / 255.0
            input_img = np.expand_dims(input_img, axis=0)

            # Inference
            outputs = self.trt_model.infer([input_img])
            output0 = outputs[0][0] 
            if output0.shape[0] < output0.shape[1]:
                output0 = output0.T 
            
            # cxcywh -> xyxy and scale
            boxes = output0[:, :4]
            scores = output0[:, 4:]
            
            x1 = (boxes[:, 0] - boxes[:, 2]/2 - pad[0]) / ratio[0]
            y1 = (boxes[:, 1] - boxes[:, 3]/2 - pad[1]) / ratio[1]
            x2 = (boxes[:, 0] + boxes[:, 2]/2 - pad[0]) / ratio[0]
            y2 = (boxes[:, 1] + boxes[:, 3]/2 - pad[1]) / ratio[1]
            xyxy = np.stack([x1, y1, x2, y2], axis=1)
            
            if flip:
                # Flip x coordinates back
                final_x1 = w - xyxy[:, 2]
                final_x2 = w - xyxy[:, 0]
                xyxy[:, 0] = final_x1
                xyxy[:, 2] = final_x2
                
            return xyxy, scores

        h, w = frame.shape[:2]
        
        # Pass 1: Original (with CLAHE if enabled)
        boxes1, scores1 = run_pass(proc_frame, flip=False)
        
        # Pass 2: TTA (Flip)
        if config.USE_TTA:
            boxes2, scores2 = run_pass(proc_frame, flip=True)
            res_boxes = np.concatenate([boxes1, boxes2])
            res_scores = np.concatenate([scores1, scores2])
        else:
            res_boxes, res_scores = boxes1, scores1
            
        class_indices = np.argmax(res_scores, axis=1)
        confidences = np.max(res_scores, axis=1)
        
        # --- Expert Data Science Refinement Layer ---
        # 1. Negative Class Rejection
        valid_mask = (class_indices < config.POSITIVE_CLASSES_COUNT)
        
        # 2. Tiered Confidence
        # shadowed apple, hidden red apple, small red fruit, tex skin, round red, occluded (approx mapping)
        vague_indices = [2, 3, 4, 5, 6, 7] 
        tiered_mask = np.ones_like(confidences, dtype=bool)
        for idx in vague_indices:
            if idx < config.POSITIVE_CLASSES_COUNT:
                tiered_mask &= ~((class_indices == idx) & (confidences < config.VAGUE_PROMPT_THRESHOLD))
        
        # 3. Base Mask
        active_mask = (confidences > conf_thres) & valid_mask & tiered_mask
        if not np.any(active_mask):
            return sv.Detections.empty()
            
        res_boxes = res_boxes[active_mask]
        res_confidences = confidences[active_mask]
        res_class_ids = class_indices[active_mask]
        
        # 4. Geometric Heuristics (Strict Precision Pass)
        # Stricter limits to eliminate erratic detections
        box_w = res_boxes[:, 2] - res_boxes[:, 0]
        box_h = res_boxes[:, 3] - res_boxes[:, 1]
        aspect_ratio = np.maximum(box_w / (box_h + 1e-6), box_h / (box_w + 1e-6))
        area = box_w * box_h
        
        # AR < 2.2 Rejects most branches. Area > 250 Rejects small distant noise.
        geo_mask = (aspect_ratio < 2.2) & (area > 250)
        
        if not np.any(geo_mask):
            return sv.Detections.empty()
            
        res_boxes = res_boxes[geo_mask]
        res_confidences = res_confidences[geo_mask]
        res_class_ids = res_class_ids[geo_mask]

        # 5. Chromatic Filtering (Optional)
        # Rejects objects that are not 'Red' enough in HSV space
        if config.USE_CHROMA_FILTER:
            chroma_mask = []
            for box in res_boxes:
                # Extract crop
                bx1, by1, bx2, by2 = map(int, [max(0, box[0]), max(0, box[1]), min(w, box[2]), min(h, box[3])])
                if (bx2-bx1) < 2 or (by2-by1) < 2:
                    chroma_mask.append(False)
                    continue
                crop = frame[by1:by2, bx1:bx2]
                hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                median_hue = np.median(hsv_crop[:, :, 0])
                # Red Hue ranges: [0, 15] and [160, 180]
                is_red = (median_hue <= 15) or (median_hue >= 160)
                chroma_mask.append(is_red)
            
            chroma_mask = np.array(chroma_mask)
            if not np.any(chroma_mask):
                return sv.Detections.empty()
                
            res_boxes = res_boxes[chroma_mask]
            res_confidences = res_confidences[chroma_mask]
            res_class_ids = res_class_ids[chroma_mask]
        
        # 6. Non-Maximum Suppression (NMS)
        indices = cv2.dnn.NMSBoxes(res_boxes.tolist(), res_confidences.tolist(), conf_thres, 0.45)
        if len(indices) > 0:
            indices = np.array(indices).flatten()
            return sv.Detections(
                xyxy=res_boxes[indices].astype(np.float32),
                confidence=res_confidences[indices].astype(np.float32),
                class_id=res_class_ids[indices].astype(int)
            )
        return sv.Detections.empty()

class AppleSegmenter:
    """
    Optimized FastSAM Segmenter with Letterbox support.
    """
    def __init__(self):
        self.use_trt = config.USE_TENSORRT and HAS_TRT
        
        if self.use_trt and os.path.exists(config.SEGMENTATION_ENGINE_PATH):
            print(f"[*] Initializing Optimized FastSAM Segmenter (TRT): {config.SEGMENTATION_ENGINE_PATH}")
            self.trt_model = TRTModel(config.SEGMENTATION_ENGINE_PATH)
            self.imgsz = config.SEGMENTATION_IMGSZ
        else:
            print(f"[*] Initializing Baseline FastSAM Segmenter (CPU): {config.SEGMENTATION_MODEL_ID}")
            from ultralytics import FastSAM
            self.model = FastSAM(config.SEGMENTATION_MODEL_ID)
            self.use_trt = False

    def segment_boxes(self, frame: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        if len(bboxes) == 0:
            return []
            
        if not self.use_trt:
            results = self.model(frame, bboxes=bboxes, conf=0.1, retina_masks=False, imgsz=config.SEGMENTATION_IMGSZ, verbose=False)[0]
            if hasattr(results, 'masks') and results.masks is not None:
                return results.masks.xy
            return []

        # --- TRT Optimized Path for FastSAM with Letterbox ---
        oh, ow = frame.shape[:2]
        input_img, ratio, pad = letterbox(frame, (self.imgsz, self.imgsz))
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)
        input_img = input_img.astype(np.float32) / 255.0
        input_img = np.expand_dims(input_img, axis=0)

        # Inference
        outputs = self.trt_model.infer([input_img])
        output0 = outputs[0][0] # [N, 37] or [37, N]
        protos = outputs[1][0]  # [32, 64, 64]
        
        if output0.shape[0] < output0.shape[1]:
            output0 = output0.T
        
        pred_boxes = output0[:, :4]
        pred_scores = output0[:, 4]
        pred_coeffs = output0[:, 5:]
        
        indices = cv2.dnn.NMSBoxes(pred_boxes.tolist(), pred_scores.tolist(), 0.1, 0.4)
        if len(indices) == 0: return [np.array([]) for _ in bboxes]
        indices = np.array(indices).flatten()
        
        f_pred_boxes = pred_boxes[indices]
        f_pred_coeffs = pred_coeffs[indices]
        
        # Scale internal boxes to original frame
        px1 = (f_pred_boxes[:, 0] - f_pred_boxes[:, 2]/2 - pad[0]) / ratio[0]
        py1 = (f_pred_boxes[:, 1] - f_pred_boxes[:, 3]/2 - pad[1]) / ratio[1]
        px2 = (f_pred_boxes[:, 0] + f_pred_boxes[:, 2]/2 - pad[0]) / ratio[0]
        py2 = (f_pred_boxes[:, 1] + f_pred_boxes[:, 3]/2 - pad[1]) / ratio[1]
        pred_xyxy = np.stack([px1, py1, px2, py2], axis=1)
        
        final_polygons = []
        for bbox in bboxes:
            cious = self._compute_ious(bbox, pred_xyxy)
            centers = np.stack([(pred_xyxy[:, 0] + pred_xyxy[:, 2])/2, 
                               (pred_xyxy[:, 1] + pred_xyxy[:, 3])/2], axis=1)
            
            in_bbox = (centers[:, 0] >= bbox[0] - 5) & (centers[:, 0] <= bbox[2] + 5) & \
                      (centers[:, 1] >= bbox[1] - 5) & (centers[:, 1] <= bbox[3] + 5)
            
            cious = cious * in_bbox 
            best_match_idx = np.argmax(cious)
            
            if cious[best_match_idx] < 0.1:
                final_polygons.append(np.array([]))
                continue
            
            # Reconstruct Mask
            c = f_pred_coeffs[best_match_idx]
            mask_raw = (c @ protos.reshape(32, -1)).reshape(64, 64)
            mask_raw = 1 / (1 + np.exp(-mask_raw)) # Sigmoid
            
            # Letterbox Inverse Resize for mask
            mask_upscaled = cv2.resize(mask_raw, (self.imgsz, self.imgsz))
            
            t, l = int(np.floor(pad[1])), int(np.floor(pad[0]))
            b, r = int(np.ceil(self.imgsz - pad[1])), int(np.ceil(self.imgsz - pad[0]))
            mask_cropped = mask_upscaled[t:b, l:r]
            
            mask_full = cv2.resize(mask_cropped, (ow, oh))
            
            # Constraint mask to bbox
            mask_refined = np.zeros_like(mask_full)
            bx1, by1, bx2, by2 = map(int, [max(0, bbox[0]), max(0, bbox[1]), min(ow, bbox[2]), min(oh, bbox[3])])
            mask_refined[by1:by2, bx1:bx2] = mask_full[by1:by2, bx1:bx2]
            
            mask_bool = (mask_refined > 0.45).astype(np.uint8)
            contours, _ = cv2.findContours(mask_bool, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                cnt = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(cnt)
                perimeter = cv2.arcLength(cnt, True)
                
                # 3. Expert Circularity Heuristic
                # C = 4 * PI * Area / Perimeter^2
                if perimeter > 0:
                    circularity = (4 * np.pi * area) / (perimeter ** 2)
                    if circularity < config.CIRCULARITY_THRESHOLD:
                        final_polygons.append(np.array([]))
                        continue
                
                poly = cnt.reshape(-1, 2)
                final_polygons.append(poly.astype(float))
            else:
                final_polygons.append(np.array([]))
                
        return final_polygons

    def _compute_ious(self, box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        x1 = np.maximum(box[0], boxes[:, 0])
        y1 = np.maximum(box[1], boxes[:, 1])
        x2 = np.minimum(box[2], boxes[:, 2])
        y2 = np.minimum(box[3], boxes[:, 3])
        inter = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area1 = (box[2] - box[0]) * (box[3] - box[1])
        area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        union = area1 + area2 - inter + 1e-6
        return inter / union
