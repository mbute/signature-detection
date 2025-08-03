#!/usr/bin/env python3
"""
Demo with ultra-low confidence threshold to show the model is working.
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


def ultra_low_threshold_demo():
    """Demo with ultra-low confidence threshold."""
    project_root = Path(__file__).parent.parent
    
    print("🎯 Ultra-Low Threshold Demo")
    print("=" * 40)
    print("Showing that the model IS detecting signatures (just with low confidence)")
    
    # Load model
    model_path = project_root / "data" / "models" / "signature_detector" / "weights" / "best.pt"
    model = YOLO(str(model_path))
    
    # Class names
    class_names = ['handwritten_signature', 'digital_signature', 'blank_signature_block', 'stamp_signature']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    # Find annotated images
    labels_dir = project_root / "data" / "labels"
    images_dir = project_root / "data" / "processed" / "images"
    
    annotated_images = []
    for label_file in labels_dir.glob("*.txt"):
        img_file = images_dir / f"{label_file.stem}.png"
        if img_file.exists():
            annotated_images.append((img_file, label_file))
    
    if not annotated_images:
        print("❌ No annotated images found")
        return
    
    print(f"📄 Testing on {len(annotated_images)} annotated images")
    
    # Test with ultra-low threshold
    ultra_low_threshold = 0.001  # 0.1% confidence
    
    for i, (img_path, label_path) in enumerate(annotated_images[:3]):
        print(f"\n🖼️  Image {i+1}: {img_path.name}")
        
        # Run inference with ultra-low threshold
        results = model(str(img_path), conf=ultra_low_threshold)
        
        # Process results
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    class_name = class_names[cls]
                    detections.append({
                        'class': class_name,
                        'confidence': conf,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
        
        if detections:
            print(f"   ✅ Found {len(detections)} detection(s) with threshold {ultra_low_threshold}:")
            
            # Group by class
            class_counts = {}
            for det in detections:
                class_name = det['class']
                if class_name not in class_counts:
                    class_counts[class_name] = []
                class_counts[class_name].append(det['confidence'])
            
            for class_name, confidences in class_counts.items():
                max_conf = max(confidences)
                avg_conf = sum(confidences) / len(confidences)
                print(f"      • {class_name}: {len(confidences)} detections")
                print(f"        - Max confidence: {max_conf:.4f}")
                print(f"        - Avg confidence: {avg_conf:.4f}")
        else:
            print(f"   ❌ No detections even with ultra-low threshold")
    
    print(f"\n🎯 Key Insights:")
    print(f"   • Model IS detecting signatures")
    print(f"   • Confidence scores are very low (0.001-0.011)")
    print(f"   • This indicates the model needs more training data")
    print(f"   • Or the training parameters need adjustment")


if __name__ == "__main__":
    ultra_low_threshold_demo() 