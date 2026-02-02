from typing import Optional, Any, List
import numpy as np
import supervision as sv
from ultralytics import YOLO, FastSAM
from inference.models.yolo_world.yolo_world import YOLOWorld
from .config import config

class AppleDetector:
    """
    High-level wrapper for the YOLO-World Zero-Shot detector.
    """
    def __init__(self):
        print(f"[*] Initializing YOLO-World Detector: {config.MODEL_ID}")
        self.model = YOLOWorld(model_id=config.MODEL_ID)
        self.target_classes = config.TARGET_CLASSES

    def infer(self, frame: np.ndarray, confidence: Optional[float] = None) -> sv.Detections:
        conf = confidence if confidence is not None else config.CONFIDENCE_THRESHOLD
        results = self.model.infer(frame, text=self.target_classes, confidence=conf)
        return sv.Detections.from_inference(results)

class AppleSegmenter:
    """
    Refined segmenter using FastSAM.
    Generates masks specifically for bounding boxes provided by the primary detector.
    """
    def __init__(self):
        print(f"[*] Initializing FastSAM Segmenter: {config.SEGMENTATION_MODEL_ID}")
        self.model = FastSAM(config.SEGMENTATION_MODEL_ID)

    def segment_boxes(self, frame: np.ndarray, bboxes: np.ndarray) -> List[np.ndarray]:
        """
        Takes a frame and a list of bounding boxes (xyxy).
        Returns a list of polygons synchronized with the input boxes.
        """
        if len(bboxes) == 0:
            return []
            
        # FastSAM prompted segmentation
        # Using retina_masks=True to ensure coordinates match the original frame exactly
        # optimization: enforce imgsz=1024 or lower if needed, but 1024 is standard for FastSAM
        # Setting imgsz=640 to trade off some precision for speed if latency is high
        results = self.model(frame, bboxes=bboxes, conf=0.1, retina_masks=False, imgsz=config.SEGMENTATION_IMGSZ, verbose=False)[0]
        
        if hasattr(results, 'masks') and results.masks is not None:
            # results.masks.xy contains the polygons in original resolution
            return results.masks.xy
        return []
