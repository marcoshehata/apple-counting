import numpy as np
import logging
from typing import List, Dict, Any, Optional, Tuple, Set
from .config import config

# --- Logging Configuration ---
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("AppleTracker")

class Tracker:
    """
    Robust Simple IOU Tracker with Centroid Fallback.
    
    This tracker is designed for high-stability apple counting. It uses a 2-stage 
    association process (IOU followed by Centroid distance) to handle object 
    deformation and 360-degree rotations.
    """
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 30):
        """
        Initializes the tracker.
        
        Args:
            iou_threshold: Minimum overlap required for Stage 1 (IOU) matching.
            max_age: Number of consecutive frames an object can be lost before purging.
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.confirmed_count = 0
        self.tracks: Dict[int, Dict[str, Any]] = {}
        
        # Spatial sensitivity for rotation fallback (Stage 2)
        self.max_distance_threshold = 150 
        
        logger.info(f"Tracker initialized (IOU Threshold: {iou_threshold})")

    def _compute_iou(self, box1: np.ndarray, box2: np.ndarray) -> float:
        """Computes the Intersection over Union (IoU) of two bounding boxes."""
        x1, y1, x2, y2 = (
            max(box1[0], box2[0]), 
            max(box1[1], box2[1]), 
            min(box1[2], box2[2]), 
            min(box1[3], box2[3])
        )
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        return intersection / union if union > 0 else 0

    def _get_centroid(self, box: np.ndarray) -> np.ndarray:
        """Calculates the center point (x, y) of a bounding box."""
        return np.array([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2])

    @property
    def track_data(self) -> Dict[int, Dict[str, Any]]:
        """Returns the internal tracks dictionary."""
        return self.tracks

    def update(self, detections: Any) -> Tuple[Any, List[int], List[str]]:
        """
        Main update method called on every frame.
        
        Args:
            detections: A supervision.Detections object from the detector.
            
        Returns:
            A tuple of (original detections, visual IDs, track statuses).
        """
        # Note: We do not age tracks globally here to avoid double-aging logic errors.
        # Tracks are aged selectively during the cleanup phase at the end of this method.
        
        if len(detections) == 0:
            self._cleanup_old_tracks(used_ids=set())
            return detections, [], []

        matched_ids: List[Optional[int]] = [None] * len(detections)
        used_track_ids: Set[int] = set()
        
        # --- Stage 1: IOU Matching (Overlap based) ---
        for i, det_box in enumerate(detections.xyxy):
            best_iou, best_tid = 0.0, None
            for tid, track in self.tracks.items():
                if tid in used_track_ids: continue
                iou = self._compute_iou(det_box, track['bbox'])
                if iou >= self.iou_threshold and iou > best_iou: 
                    best_iou, best_tid = iou, tid
            if best_tid is not None: 
                matched_ids[i] = best_tid
                used_track_ids.add(best_tid)

        # --- Stage 2: Centroid Fallback (Proximity based) ---
        # Used when IOU fails (e.g., fast movement or object rotation/deformation)
        for i, det_box in enumerate(detections.xyxy):
            if matched_ids[i] is not None: continue
            
            det_centroid = self._get_centroid(det_box)
            min_dist, best_tid = self.max_distance_threshold, None
            
            for tid, track in self.tracks.items():
                if tid in used_track_ids: continue
                dist = np.linalg.norm(det_centroid - self._get_centroid(track['bbox']))
                if dist < min_dist: 
                    min_dist, best_tid = dist, tid
            
            if best_tid is not None: 
                matched_ids[i] = best_tid
                used_track_ids.add(best_tid)

        # --- Stage 3: Promotion and State Transitions ---
        final_visual_ids: List[int] = []
        statuses: List[str] = []
        
        for i, tid in enumerate(matched_ids):
            det_box = detections.xyxy[i]
            
            if tid is not None:
                # Existing Track Found
                track = self.tracks[tid]
                track.update({
                    'bbox': det_box, 
                    'age': 0, 
                    'hits': track['hits'] + 1
                })
                
                # Promotion Logic: Promote "Candidate" to "Confirmed"
                if (track['status'] == 'candidate' and 
                    track['hits'] >= config.MIN_HITS_FOR_CONFIRMATION):
                    
                    track['status'] = 'confirmed'
                    self.confirmed_count += 1
                    track['confirmed_id'] = self.confirmed_count
                    logger.info(f"[PROMOTION] Track {tid} confirmed as Apple #{self.confirmed_count}")
                
                final_visual_ids.append(track['confirmed_id'] if track['status'] == 'confirmed' else tid)
                statuses.append(track['status'])
            else:
                # New Track Found (Initialize as Candidate)
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    'bbox': det_box, 
                    'age': 0, 
                    'hits': 1, 
                    'status': 'candidate', 
                    'confirmed_id': -1
                }
                final_visual_ids.append(new_id)
                statuses.append('candidate')

        # Cleanup lost tracks
        self._cleanup_old_tracks(used_ids=used_track_ids)
        
        return detections, final_visual_ids, statuses

    def _cleanup_old_tracks(self, used_ids: Set[int]):
        """Increments age of unmatched tracks and purges those exceeding max_age."""
        for tid in list(self.tracks.keys()):
            if tid not in used_ids:
                self.tracks[tid]['age'] += 1
                if self.tracks[tid]['age'] > self.max_age:
                    logger.info(f"[PURGE] Track {tid} lost for {self.max_age} frames. Removing.")
                    del self.tracks[tid]
