import os
from typing import List
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Centralized configuration management for the Apple Counting system.
    Loads settings from environment variables with safe defaults.
    """
    
    # --- Model Settings ---
    # Options: yolo_world/s, yolo_world/m, yolo_world/l
    # Optimized: Use local YOLO-World v2 Small engine at high resolution (1280) for Jetson Thor GPU
    MODEL_ID: str = os.getenv("MODEL_ID", "models/yolov8s-worldv2.pt")
    ENGINE_PATH: str = os.getenv("ENGINE_PATH", "models/yolov8s-worldv2.engine")
    
    # Segmentation for the "WOW" view (Prompted via Boxes)
    SEGMENTATION_MODEL_ID: str = os.getenv("SEGMENTATION_MODEL_ID", "models/FastSAM-s.pt")
    SEGMENTATION_ENGINE_PATH: str = os.getenv("SEGMENTATION_ENGINE_PATH", "models/FastSAM-s.engine")
    SEGMENTATION_IMGSZ: int = int(os.getenv("SEGMENTATION_IMGSZ", 256))

    # Inference Mode
    USE_TENSORRT: bool = os.getenv("USE_TENSORRT", "true").lower() == "true"
    USE_CACHE: bool = os.getenv("USE_CACHE", "true").lower() == "true"
    DETECTIONS_IMGSZ: int = int(os.getenv("DETECTIONS_IMGSZ", 1280))
    USE_CLAHE: bool = os.getenv("USE_CLAHE", "true").lower() == "true"
    USE_TTA: bool = os.getenv("USE_TTA", "true").lower() == "true"
    USE_CHROMA_FILTER: bool = os.getenv("USE_CHROMA_FILTER", "true").lower() == "true"
    CIRCULARITY_THRESHOLD: float = float(os.getenv("CIRCULARITY_THRESHOLD", 0.6))
    
    # Classes for zero-shot detection (comma-separated in .env)
    PRIMARY_CLASSES: List[str] = os.getenv(
        "PRIMARY_CLASSES", 
        "apple,red apple,ripe apple,fruit"
    ).split(",")
    TARGET_CLASSES: List[str] = PRIMARY_CLASSES
    POSITIVE_CLASSES_COUNT: int = int(os.getenv("POSITIVE_CLASSES_COUNT", len(PRIMARY_CLASSES)))
    
    # --- Processing Thresholds ---
    # Global detection confidence
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.05))
    
    # UI Visibility (Hide low-confidence candidate detections)
    CANDIDATE_VISIBLE_THRESHOLD: float = float(os.getenv("CANDIDATE_VISIBLE_THRESHOLD", 0.05))
    SEGMENTATION_MIN_CONF: float = float(os.getenv("SEGMENTATION_MIN_CONF", 0.01))
    VAGUE_PROMPT_THRESHOLD: float = float(os.getenv("VAGUE_PROMPT_THRESHOLD", 0.05))
    
    # --- Tracker Settings ---
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", 0.3))
    
    # Number of frames to persist a lost track
    MAX_AGE: int = int(os.getenv("MAX_AGE", 60))
    
    # Minimum detection hits required to promote a Candidate to a Confirmed Apple
    MIN_HITS_FOR_CONFIRMATION: int = int(os.getenv("MIN_HITS_FOR_CONFIRMATION", 3))
    
    # --- Visualization Settings ---
    FILL_ALPHA: float = float(os.getenv("FILL_ALPHA", 0.5))
    
    # --- Server Settings ---
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = os.getenv("HOST", "0.0.0.0")

# Singleton instance for consistent access across modules
config = Config()
