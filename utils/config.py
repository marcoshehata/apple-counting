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
    MODEL_ID: str = os.getenv("MODEL_ID", "yolo_world/s")
    
    # Segmentation for the "WOW" view (Prompted via Boxes)
    SEGMENTATION_MODEL_ID: str = os.getenv("SEGMENTATION_MODEL_ID", "FastSAM-s.pt")
    SEGMENTATION_IMGSZ: int = int(os.getenv("SEGMENTATION_IMGSZ", 256))
    
    # Classes for zero-shot detection (comma-separated in .env)
    PRIMARY_CLASSES: List[str] = os.getenv(
        "PRIMARY_CLASSES", 
        "apple,red apple,ripe apple,fruit"
    ).split(",")
    TARGET_CLASSES: List[str] = PRIMARY_CLASSES
    
    # --- Processing Thresholds ---
    # Global detection confidence
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.05))
    
    # UI Visibility (Hide low-confidence candidate detections)
    CANDIDATE_VISIBLE_THRESHOLD: float = float(os.getenv("CANDIDATE_VISIBLE_THRESHOLD", 0.05))
    SEGMENTATION_MIN_CONF: float = float(os.getenv("SEGMENTATION_MIN_CONF", 0.25))
    
    # --- Tracker Settings ---
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", 0.3))
    
    # Number of frames to persist a lost track
    MAX_AGE: int = int(os.getenv("MAX_AGE", 60))
    
    # Minimum detection hits required to promote a Candidate to a Confirmed Apple
    MIN_HITS_FOR_CONFIRMATION: int = int(os.getenv("MIN_HITS_FOR_CONFIRMATION", 4))
    
    # --- Visualization Settings ---
    FILL_ALPHA: float = float(os.getenv("FILL_ALPHA", 0.5))
    
    # --- Server Settings ---
    PORT: int = int(os.getenv("PORT", 8000))
    HOST: str = os.getenv("HOST", "0.0.0.0")

# Singleton instance for consistent access across modules
config = Config()
