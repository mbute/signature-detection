#!/usr/bin/env python3
"""
Test the trained model on images that were annotated during training.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO


def test_annotated_images():
    """Test the model on images that were annotated."""
    project_root = Path(__file__).parent.parent
    
    # Define paths
    model_path = project_root / "data" / "models" / "signature_detector2" / "weights" / "best.pt"
    labels_dir = project_root / "data" / "labels"
    images_dir = project_root / "data" / "processed" / "images"
    output_dir = project_root / "output" / "annotated_test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Testing model on annotated images...")
    print(f"📁 Model: {model_path}")
    print(f"📁 Output: {output_dir}")
    
    # Check if model exists
    if not model_path.exists():
        print("❌ Model not found. Run training first.")
        return
    
    # Load model
    model = YOLO(str(model_path))
    
    # Class names
    class_names = ['handwritten_signature', 'digital_signature', 'blank_signature_block', 'stamp_signature']
    
    # Find annotated images (those with corresponding label files)
    annotated_images = []
    for label_file in labels_dir.glob("*.txt"):
        img_file = images_dir / f"{label_file.stem}.png"
        if img_file.exists():
            annotated_images.append((img_file, label_file))
    
    print(f"📄 Found {len(annotated_images)} annotated images")
    
    # Test on annotated images
    for img_path, label_path in annotated_images[:5]:  # Test first 5
        print(f"\n🖼️  Testing: {img_path.name}")
        
        # Run inference with very low confidence threshold
        results = model(str(img_path), conf=0.1)  # Very low threshold for testing
        
        # Load image for visualization
        image = cv2.imread(str(img_path))
        
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
                    
                    print(f"   📍 {class_name}: {conf:.3f} at ({x1}, {y1}, {x2}, {y2})")
                    detections_found = True
                    
                    # Draw bounding box
                    color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        if not detections_found:
            print("   ❌ No signatures detected (even with low threshold)")
        
        # Save result
        output_path = output_dir / f"test_{img_path.name}"
        cv2.imwrite(str(output_path), image)
        print(f"   💾 Result saved: {output_path.name}")


if __name__ == "__main__":
    test_annotated_images() 