#!/usr/bin/env python3
"""
Test the current trained signature detection model.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO


def test_current_model():
    """Test the current trained model."""
    project_root = Path(__file__).parent.parent
    
    # Define paths - use the correct model path
    model_path = project_root / "data" / "models" / "signature_detector" / "weights" / "best.pt"
    test_images_dir = project_root / "data" / "processed" / "images"
    output_dir = project_root / "output" / "current_test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Testing current signature detection model...")
    print(f"📁 Model: {model_path}")
    print(f"📁 Test images: {test_images_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Check if model exists
    if not model_path.exists():
        print("❌ Model not found. Looking for alternative paths...")
        # Try alternative paths
        alternative_paths = [
            project_root / "data" / "models" / "signature_detector2" / "weights" / "best.pt",
            project_root / "data" / "models" / "signature_detector_improved" / "weights" / "best.pt"
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                model_path = alt_path
                print(f"✅ Found model at: {model_path}")
                break
        else:
            print("❌ No model found. Run training first.")
            return
    
    # Load model
    model = YOLO(str(model_path))
    
    # Class names
    class_names = ['handwritten_signature', 'digital_signature', 'blank_signature_block', 'stamp_signature']
    
    # Test on annotated images first
    labels_dir = project_root / "data" / "labels"
    annotated_images = []
    for label_file in labels_dir.glob("*.txt"):
        img_file = test_images_dir / f"{label_file.stem}.png"
        if img_file.exists():
            annotated_images.append((img_file, label_file))
    
    print(f"📄 Found {len(annotated_images)} annotated images")
    
    # Test on first 5 annotated images
    for img_path, label_path in annotated_images[:5]:
        print(f"\n🖼️  Testing: {img_path.name}")
        
        # Run inference with different confidence thresholds
        for conf_threshold in [0.1, 0.25, 0.5]:
            results = model(str(img_path), conf=conf_threshold)
            
            # Process results
            detections_found = False
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        # Get box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        
                        # Get class and confidence
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        class_name = class_names[cls]
                        
                        print(f"   📍 {class_name}: {conf:.3f} (threshold: {conf_threshold})")
                        detections_found = True
            
            if detections_found:
                break
        
        if not detections_found:
            print("   ❌ No signatures detected at any threshold")
        
        # Save visualization with lowest threshold
        results = model(str(img_path), conf=0.1)
        image = cv2.imread(str(img_path))
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    class_name = class_names[cls]
                    
                    color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        output_path = output_dir / f"test_{img_path.name}"
        cv2.imwrite(str(output_path), image)
        print(f"   💾 Result saved: {output_path.name}")


if __name__ == "__main__":
    test_current_model() 