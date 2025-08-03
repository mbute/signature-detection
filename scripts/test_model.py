#!/usr/bin/env python3
"""
Test the trained signature detection model.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO


def test_model():
    """Test the trained model on sample images."""
    project_root = Path(__file__).parent.parent
    
    # Define paths
    model_path = project_root / "data" / "models" / "signature_detector2" / "weights" / "best.pt"
    test_images_dir = project_root / "data" / "processed" / "images"
    output_dir = project_root / "output" / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔍 Testing signature detection model...")
    print(f"📁 Model: {model_path}")
    print(f"📁 Test images: {test_images_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Check if model exists
    if not model_path.exists():
        print("❌ Model not found. Run training first.")
        return
    
    # Load model
    model = YOLO(str(model_path))
    
    # Class names
    class_names = ['handwritten_signature', 'digital_signature', 'blank_signature_block', 'stamp_signature']
    
    # Test on a few sample images
    test_images = list(test_images_dir.glob("*.png"))[:5]  # Test first 5 images
    
    for img_path in test_images:
        print(f"\n🖼️  Testing: {img_path.name}")
        
        # Run inference
        results = model(str(img_path), conf=0.25)  # Lower confidence threshold for testing
        
        # Load image for visualization
        image = cv2.imread(str(img_path))
        
        # Process results
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
                    
                    # Draw bounding box
                    color = (0, 255, 0) if conf > 0.5 else (0, 255, 255)  # Green if confident, yellow if not
                    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                    
                    # Add label
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                print("   ❌ No signatures detected")
        
        # Save result
        output_path = output_dir / f"test_{img_path.name}"
        cv2.imwrite(str(output_path), image)
        print(f"   💾 Result saved: {output_path.name}")


if __name__ == "__main__":
    test_model() 