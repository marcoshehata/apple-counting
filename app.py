import os
import cv2
import numpy as np
import time
import logging
import traceback
import json
from typing import List, Dict, Any, Optional, Tuple
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
import supervision as sv

# Modular imports
from utils.config import config
from utils.detector import AppleDetector, AppleSegmenter
from utils.tracker import Tracker

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("StandaloneVisionAPI")

app = FastAPI(title="Standalone Vision API", version="2.3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
detector = AppleDetector()
segmenter = AppleSegmenter()
# Initialize models
detector = AppleDetector()
segmenter = AppleSegmenter()
tracker = Tracker()

def decode_image(contents: bytes) -> np.ndarray:
    # Use IMREAD_COLOR (1) explicitly, but optimize if possible
    nparr = np.frombuffer(contents, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

def process_detection(frame: np.ndarray, conf_threshold: Optional[float] = None, classes: Optional[str] = None):
    """
    Direct detection processing without caching.
    """
    h, w, _ = frame.shape
    if classes:
        detector.target_classes = [c.strip() for c in classes.split(",")]
    
    t0 = time.time()
    detections = detector.infer(frame, confidence=conf_threshold)
    t1 = time.time()
    logger.info(f"Detector Inference: {(t1-t0)*1000:.2f}ms")
    
    detections, visual_ids, statuses = tracker.update(detections)
    
    return detections, visual_ids, statuses, w, h

@app.get("/health")
async def health():
    return {
        "status": "healthy",
        "model": config.MODEL_ID,
        "count": tracker.confirmed_count,
        "active_tracks": len(tracker.tracks)
    }

@app.post("/detect")
async def detect(
    image: UploadFile = File(...),
    conf_threshold: Optional[float] = Form(None),
    classes: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """
    Video Object Detection Node Endpoint.
    """
    start_time = time.time()
    try:
        contents = await image.read()
        t_read = time.time()
        
        frame = decode_image(contents)
        if frame is None: return {"error": "Invalid image"}
        t_decode = time.time()
        
        detections, visual_ids, statuses, w, h = process_detection(frame, conf_threshold, classes)
        t_infer = time.time()
        
        results = []
        if detections.xyxy is not None:
            for i in range(len(detections.xyxy)):
                results.append({
                    "bbox": detections.xyxy[i].tolist(),
                    "confidence": round(float(detections.confidence[i]), 3),
                    "tracker_id": int(visual_ids[i]),
                    "status": statuses[i]
                })
        
        total_ms = int((time.time() - start_time) * 1000)
        logger.info(f"DETECT: Read={(t_read-start_time)*1000:.1f}ms Decode={(t_decode-t_read)*1000:.1f}ms Infer={(t_infer-t_decode)*1000:.1f}ms Total={total_ms}ms")
        
        return {
            "width": w, "height": h,
            "detections": results,
            "count": tracker.confirmed_count,
            "performance": {"total_ms": total_ms}
        }
    except Exception as e:
        logger.error(f"Detection Error: {str(e)}")
        # traceback.print_exc()
        return {"error": str(e)}

@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    conf_threshold: Optional[float] = Form(None),
    classes: Optional[str] = Form(None)
) -> Dict[str, Any]:
    """
    Video Segmentation Node Endpoint.
    """
    start_time = time.time()
    try:
        contents = await image.read()
        t_read = time.time()
        
        frame = decode_image(contents)
        if frame is None: return {"error": "Invalid image"}
        t_decode = time.time()
        
        # 1. Detection & Tracking
        detections, visual_ids, statuses, w, h = process_detection(frame, conf_threshold, classes)
        t_infer_det = time.time()
        
        # 2. FastSAM Refiner
        # Only segment confirmed or high-confidence objects to save time?
        # Optimization: Filter boxes BEFORE segmentation to save latency!
        # This addresses user's concern about efficiency.
        
        valid_indices = []
        valid_boxes = []
        
        if detections.xyxy is not None:
             for i in range(len(detections.xyxy)):
                conf = float(detections.confidence[i])
                status = statuses[i]
                # Optimization: Only segment confirmed tracks or high-confidence candidates
                if status == 'confirmed':
                     valid_indices.append(i)
                     valid_boxes.append(detections.xyxy[i])
                elif status == 'candidate' and conf >= config.SEGMENTATION_MIN_CONF:
                     valid_indices.append(i)
                     valid_boxes.append(detections.xyxy[i])

        
        polygons_filtered = []
        if len(valid_boxes) > 0:
            polygons_filtered = segmenter.segment_boxes(frame, np.array(valid_boxes))
        
        t_infer_seg = time.time()
        
        results = []
        for idx, original_idx in enumerate(valid_indices):
             results.append({
                "tracker_id": int(visual_ids[original_idx]),
                "polygon": polygons_filtered[idx].tolist() if idx < len(polygons_filtered) else []
            })
        
        total_ms = int((time.time() - start_time) * 1000)
        logger.info(f"SEGMENT: Read={(t_read-start_time)*1000:.1f}ms Decode={(t_decode-t_read)*1000:.1f}ms Det={(t_infer_det-t_decode)*1000:.1f}ms Seg={(t_infer_seg-t_infer_det)*1000:.1f}ms Total={total_ms}ms")
                
        return {
            "width": w, "height": h,
            "segments": results,
            "performance": {"total_ms": total_ms}
        }
    except Exception as e:
        logger.error(f"Segmentation Error: {str(e)}")
        return {"error": str(e)}

@app.get("/reset")
async def reset():
    global tracker
    tracker = Tracker()
    # No cache to clear
    return {"status": "success"}

@app.post("/infer")
async def infer_legacy(image: UploadFile = File(...), conf_threshold: Optional[float] = Form(None), frame_id: int = Form(0)):
    # Keep legacy for compatibility if needed, using new optimize logic
    start_time = time.time()
    try:
        contents = await image.read()
        frame = decode_image(contents)
        if frame is None: return {"error": "Invalid image", "frame_id": frame_id}
        
        detections, visual_ids, statuses, w, h = process_detection(frame, conf_threshold)
        
        # FastSAM on all (legacy behavior)
        polygons_raw = segmenter.segment_boxes(frame, detections.xyxy if detections.xyxy is not None else np.array([]))
        
        formatted = []
        confidences = []
        confirmed_in_frame = 0
        
        if detections.xyxy is not None:
            for i in range(len(detections.xyxy)):
                status = statuses[i]
                conf = float(detections.confidence[i])
                if status == 'confirmed': 
                    confidences.append(conf)
                    confirmed_in_frame += 1
                if status == 'candidate' and conf < config.CANDIDATE_VISIBLE_THRESHOLD: 
                    continue
                formatted.append({
                    "bbox": detections.xyxy[i].tolist(),
                    "confidence": round(conf, 3),
                    "tracker_id": int(visual_ids[i]),
                    "status": status,
                    "polygon": polygons_raw[i].tolist() if i < len(polygons_raw) else []
                })
        
        return {
            "frame_id": frame_id, "width": w, "height": h,
            "detections": formatted,
            "count": tracker.confirmed_count,
            "current_count": confirmed_in_frame,
            "avg_confidence": round(sum(confidences)/len(confidences), 2) if confidences else 0.0,
            "active_tracks": len(tracker.tracks),
            "performance": {"total_ms": int((time.time() - start_time) * 1000)}
        }
    except Exception as e:
        logger.error(f"Infer error: {str(e)}")
        return {"error": str(e), "frame_id": frame_id}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=config.HOST, port=config.PORT)
