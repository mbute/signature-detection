#!/usr/bin/env python3
"""
Quick demo script for signature detection - perfect for presentations!
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO


def quick_demo():
    """Quick demo showing signature detection in action."""
    project_root = Path(__file__).parent.parent
    
    print("🎯 Signature Detection Quick Demo")
    print("=" * 40)
    
    # Find model
    model_path = project_root / "data" / "models" / "signature_detector" / "weights" / "best.pt"
    if not model_path.exists():
        print("❌ No trained model found. Please run training first.")
        return
    
    # Load model
    print("🔄 Loading signature detection model...")
    model = YOLO(str(model_path))
    
    # Class names
    class_names = ['handwritten_signature', 'digital_signature', 'blank_signature_block', 'stamp_signature']
    
    # Find test images
    images_dir = project_root / "data" / "processed" / "images"
    image_files = list(images_dir.glob("*.png"))
    
    if not image_files:
        print("❌ No test images found")
        return
    
    print(f"📄 Found {len(image_files)} test images")
    print("\n🔍 Running signature detection...")
    print("-" * 40)
    
    # Test on first 3 images
    for i, img_path in enumerate(image_files[:3]):
        print(f"\n🖼️  Image {i+1}: {img_path.name}")
        
        # Run inference
        results = model(str(img_path), conf=0.1)
        
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
            print(f"   ✅ Found {len(detections)} signature(s):")
            for det in detections:
                print(f"      • {det['class']}: {det['confidence']:.2f}")
        else:
            print("   ❌ No signatures detected")
    
    print("\n" + "=" * 40)
    print("🎉 Demo complete!")
    print("\n💡 For visual results, run:")
    print("   python -m scripts.demo_inference")
    print("   python -m scripts.interactive_demo")


if __name__ == "__main__":
    quick_demo() 