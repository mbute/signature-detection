#!/usr/bin/env python3
"""
Fixed improved training script for signature detection with correct YOLO parameters.
"""

import os
import sys
from pathlib import Path
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO


def train_improved_fixed():
    """Train YOLO model with improved parameters for small datasets."""
    project_root = Path(__file__).parent.parent
    
    # Define paths
    dataset_yaml = project_root / "data" / "dataset" / "dataset.yaml"
    models_dir = project_root / "data" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Starting improved YOLO training (fixed parameters)...")
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
    model = YOLO('yolov8n.pt')  # Start with nano model
    
    # Fixed training parameters (removed invalid ones)
    training_params = {
        'data': str(dataset_yaml),
        'epochs': 200,  # More epochs
        'imgsz': 640,
        'batch': 4,  # Smaller batch size
        'device': 'cpu',
        'project': str(models_dir),
        'name': 'signature_detector_improved_fixed',
        'save': True,
        'save_period': 25,  # Save every 25 epochs
        'patience': 50,  # More patience
        'verbose': True,
        'lr0': 0.001,  # Lower learning rate
        'lrf': 0.1,  # Lower final learning rate
        'momentum': 0.937,
        'weight_decay': 0.0005,
        'warmup_epochs': 5,  # More warmup
        'warmup_momentum': 0.8,
        'warmup_bias_lr': 0.1,
        'box': 7.5,  # Box loss gain
        'cls': 0.5,  # Class loss gain
        'dfl': 1.5,  # DFL loss gain
        'label_smoothing': 0.0,  # Label smoothing epsilon
        'nbs': 64,  # Nominal batch size
        'overlap_mask': True,
        'mask_ratio': 4,
        'dropout': 0.0,  # Use dropout for regularization
        'val': True,
        'plots': True,
        'save_txt': False,
        'save_conf': False,
        'save_crop': False,
        'save_json': False,
        'single_cls': False,
        'augment': True,  # Enable augmentation
        'mosaic': 1.0,
        'mixup': 0.0,
        'copy_paste': 0.0,
        'degrees': 0.0,  # No rotation
        'translate': 0.1,  # Small translation
        'scale': 0.5,  # Scale augmentation
        'shear': 0.0,
        'perspective': 0.0,
        'flipud': 0.0,
        'fliplr': 0.5,  # Horizontal flip
        'hsv_h': 0.015,  # HSV-Hue augmentation
        'hsv_s': 0.7,  # HSV-Saturation augmentation
        'hsv_v': 0.4,  # HSV-Value augmentation
    }
    
    print(f"🎯 Improved training parameters:")
    for key, value in training_params.items():
        print(f"   {key}: {value}")
    
    # Start training
    try:
        results = model.train(**training_params)
        print("\n🎉 Improved training completed!")
        print(f"📁 Best model saved to: {models_dir}/signature_detector_improved_fixed/weights/best.pt")
        print(f"📁 Last model saved to: {models_dir}/signature_detector_improved_fixed/weights/last.pt")
        
        # Validate the model
        print("\n🔍 Validating improved model...")
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
    train_improved_fixed() 