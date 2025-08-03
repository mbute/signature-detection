#!/usr/bin/env python3
"""
Diagnostic script to understand why the model isn't detecting signatures.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO


def diagnose_model():
    """Diagnose why the model isn't detecting signatures."""
    project_root = Path(__file__).parent.parent
    
    print("🔍 Model Diagnosis")
    print("=" * 50)
    
    # 1. Check model file
    model_path = project_root / "data" / "models" / "signature_detector" / "weights" / "best.pt"
    print(f"📁 Model path: {model_path}")
    print(f"📁 Model exists: {model_path.exists()}")
    if model_path.exists():
        print(f"📁 Model size: {model_path.stat().st_size / 1024 / 1024:.1f} MB")
    
    # 2. Load model and check info
    print("\n🔄 Loading model...")
    try:
        model = YOLO(str(model_path))
        print(f"✅ Model loaded successfully")
        print(f"📊 Model info: {model.info()}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # 3. Check dataset
    dataset_yaml = project_root / "data" / "dataset" / "dataset.yaml"
    print(f"\n📋 Dataset config: {dataset_yaml}")
    print(f"📋 Dataset exists: {dataset_yaml.exists()}")
    
    if dataset_yaml.exists():
        import yaml
        with open(dataset_yaml, 'r') as f:
            dataset_config = yaml.safe_load(f)
        print(f"📊 Dataset classes: {dataset_config.get('names', 'Not found')}")
        print(f"📊 Number of classes: {dataset_config.get('nc', 'Not found')}")
    
    # 4. Check annotations
    labels_dir = project_root / "data" / "labels"
    images_dir = project_root / "data" / "processed" / "images"
    
    print(f"\n📄 Labels directory: {labels_dir}")
    print(f"📄 Images directory: {images_dir}")
    
    # Count annotations
    label_files = list(labels_dir.glob("*.txt"))
    image_files = list(images_dir.glob("*.png"))
    
    print(f"📊 Label files: {len(label_files)}")
    print(f"📊 Image files: {len(image_files)}")
    
    # 5. Check annotation format
    if label_files:
        print(f"\n📋 Sample annotation (first label file):")
        sample_label = label_files[0]
        print(f"📄 File: {sample_label.name}")
        with open(sample_label, 'r') as f:
            content = f.read().strip()
            print(f"📄 Content: {content}")
            
            # Parse annotation
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if line.strip():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        class_id = int(parts[0])
                        x_center = float(parts[1])
                        y_center = float(parts[2])
                        width = float(parts[3])
                        height = float(parts[4])
                        print(f"   Annotation {i+1}: Class {class_id}, Center ({x_center:.3f}, {y_center:.3f}), Size ({width:.3f}, {height:.3f})")
    
    # 6. Test on annotated image
    print(f"\n🧪 Testing on annotated image...")
    if label_files and image_files:
        # Find matching image
        sample_label = label_files[0]
        sample_image = images_dir / f"{sample_label.stem}.png"
        
        if sample_image.exists():
            print(f"📄 Testing on: {sample_image.name}")
            
            # Load image
            image = cv2.imread(str(sample_image))
            if image is not None:
                print(f"📊 Image shape: {image.shape}")
                
                # Test with different confidence thresholds
                for conf_threshold in [0.01, 0.05, 0.1, 0.25, 0.5]:
                    print(f"   🔍 Testing confidence threshold: {conf_threshold}")
                    results = model(str(sample_image), conf=conf_threshold)
                    
                    detections = []
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                cls = int(box.cls[0].cpu().numpy())
                                conf = float(box.conf[0].cpu().numpy())
                                detections.append({
                                    'class': cls,
                                    'confidence': conf,
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                                })
                    
                    if detections:
                        print(f"      ✅ Found {len(detections)} detection(s):")
                        for det in detections:
                            print(f"         Class {det['class']}: {det['confidence']:.3f}")
                    else:
                        print(f"      ❌ No detections")
                
                # Test with raw prediction (no confidence threshold)
                print(f"   🔍 Testing raw predictions (no threshold)...")
                results = model(str(sample_image), conf=0.0)
                
                raw_detections = []
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy())
                            conf = float(box.conf[0].cpu().numpy())
                            raw_detections.append({
                                'class': cls,
                                'confidence': conf,
                                'bbox': [int(x1), int(y1), int(x2), int(y2)]
                            })
                
                if raw_detections:
                    print(f"      📊 Raw predictions (before threshold):")
                    for det in raw_detections:
                        print(f"         Class {det['class']}: {det['confidence']:.6f}")
                else:
                    print(f"      ❌ No raw predictions at all")
            else:
                print(f"❌ Could not load image: {sample_image}")
        else:
            print(f"❌ No matching image found for: {sample_label.stem}")
    
    # 7. Check model validation results
    print(f"\n📊 Model validation results:")
    val_results_dir = project_root / "data" / "models" / "signature_detector"
    if val_results_dir.exists():
        results_files = list(val_results_dir.glob("results*.png"))
        if results_files:
            print(f"📄 Validation plots found: {len(results_files)}")
            for f in results_files:
                print(f"   📄 {f.name}")
    
    print(f"\n🎯 Diagnosis complete!")
    print(f"\n💡 Recommendations:")
    print(f"   1. Check if model was trained on the correct dataset")
    print(f"   2. Verify annotation format matches YOLO expectations")
    print(f"   3. Try training with more data and better parameters")
    print(f"   4. Check if image preprocessing is consistent")


if __name__ == "__main__":
    diagnose_model() 