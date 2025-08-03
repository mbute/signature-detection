#!/usr/bin/env python3
"""
Train YOLO model for signature detection.
"""

import os
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO
from src.utils.config import get_config


def train_model():
    """Train YOLO model for signature detection."""
    project_root = Path(__file__).parent.parent
    
    # Get configuration
    config = get_config()
    model_config = config.get_model_config()
    
    # Define paths
    dataset_yaml = project_root / "data" / "dataset" / "dataset.yaml"
    models_dir = project_root / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting YOLO model training...")
    print(f"📋 Dataset: {dataset_yaml}")
    print(f"📁 Models will be saved to: {models_dir}")
    
    # Check if dataset exists
    if not dataset_yaml.exists():
        print("❌ Dataset configuration not found. Run prepare_training_data.py first.")
        return
    
    # Load dataset configuration
    with open(dataset_yaml, 'r') as f:
        dataset_config = yaml.safe_load(f)
    
    print(f"📊 Classes: {dataset_config['names']}")
    print(f"📊 Number of classes: {dataset_config['nc']}")
    
    # Initialize model
    model = YOLO('yolov8n.pt')  # Start with nano model for faster training
    
    # Training parameters
    training_params = {
        'data': str(dataset_yaml),
        'epochs': 100,
        'imgsz': 640,
        'batch': 8,
        'device': 'cpu',  # Change to '0' for GPU if available
        'project': str(models_dir),
        'name': 'signature_detector',
        'save': True,
        'save_period': 10,
        'patience': 20,
        'verbose': True
    }
    
    print(f"🎯 Training parameters:")
    for key, value in training_params.items():
        print(f"   {key}: {value}")
    
    # Start training
    try:
        results = model.train(**training_params)
        
        print("\n🎉 Training completed!")
        print(f"📁 Best model saved to: {models_dir}/signature_detector/weights/best.pt")
        print(f"📁 Last model saved to: {models_dir}/signature_detector/weights/last.pt")
        
        # Validate the model
        print("\n🔍 Validating model...")
        val_results = model.val()
        
        print(f"📊 Validation Results:")
        print(f"   mAP50: {val_results.box.map50:.4f}")
        print(f"   mAP50-95: {val_results.box.map:.4f}")
        print(f"   Precision: {val_results.box.mp:.4f}")
        print(f"   Recall: {val_results.box.mr:.4f}")
        
        return model
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return None


if __name__ == "__main__":
    train_model() 