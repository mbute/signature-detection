#!/usr/bin/env python3
"""
Simple demo script to show signature detection inference with bounding boxes.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO


def demo_inference():
    """Demo signature detection with visual bounding boxes."""
    project_root = Path(__file__).parent.parent
    
    # Define paths
    model_path = project_root / "data" / "models" / "signature_detector" / "weights" / "best.pt"
    test_images_dir = project_root / "data" / "processed" / "images"
    output_dir = project_root / "output" / "demo_results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎯 Signature Detection Demo")
    print("=" * 50)
    print(f"📁 Model: {model_path}")
    print(f"📁 Test images: {test_images_dir}")
    print(f"📁 Output: {output_dir}")
    
    # Check if model exists
    if not model_path.exists():
        print("❌ Model not found. Looking for alternatives...")
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
            print("❌ No trained model found. Please run training first.")
            return
    
    # Load model
    print("🔄 Loading model...")
    model = YOLO(str(model_path))
    
    # Class names and colors
    class_names = ['handwritten_signature', 'digital_signature', 'blank_signature_block', 'stamp_signature']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]  # Red, Green, Blue, Yellow
    
    # Get test images
    image_files = list(test_images_dir.glob("*.png"))
    if not image_files:
        print("❌ No test images found")
        return
    
    print(f"📄 Found {len(image_files)} test images")
    
    # Test on first 3 images
    for i, img_path in enumerate(image_files[:3]):
        print(f"\n🖼️  Processing: {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"   ❌ Could not load image: {img_path}")
            continue
        
        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run inference with very low confidence threshold
        print("   🔍 Running inference...")
        results = model(str(img_path), conf=0.1)  # Very low threshold to see everything
        
        # Create figure for visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))
        
        # Original image
        ax1.imshow(image_rgb)
        ax1.set_title(f"Original Image: {img_path.name}")
        ax1.axis('off')
        
        # Image with detections
        ax2.imshow(image_rgb)
        ax2.set_title("Detections (if any)")
        ax2.axis('off')
        
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
                    color = colors[cls]
                    
                    print(f"   📍 {class_name}: {conf:.3f} at ({x1}, {y1}, {x2}, {y2})")
                    detections_found = True
                    
                    # Draw bounding box
                    rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=[c/255 for c in color], 
                                   facecolor='none')
                    ax2.add_patch(rect)
                    
                    # Add label
                    ax2.text(x1, y1-10, f"{class_name}: {conf:.2f}", 
                            color=[c/255 for c in color], fontsize=10, 
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        if not detections_found:
            print("   ❌ No signatures detected (even with low threshold)")
            ax2.text(0.5, 0.5, "No Detections", transform=ax2.transAxes, 
                    ha='center', va='center', fontsize=16, color='red')
        
        # Save result
        output_path = output_dir / f"demo_{img_path.stem}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Result saved: {output_path.name}")
        
        # Also save OpenCV version for comparison
        cv2_output_path = output_dir / f"cv2_{img_path.stem}.png"
        cv2_image = image.copy()
        
        # Draw detections on OpenCV image
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    cls = int(box.cls[0].cpu().numpy())
                    conf = float(box.conf[0].cpu().numpy())
                    class_name = class_names[cls]
                    color = colors[cls]
                    
                    cv2.rectangle(cv2_image, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name}: {conf:.2f}"
                    cv2.putText(cv2_image, label, (x1, y1-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        cv2.imwrite(str(cv2_output_path), cv2_image)
        print(f"   💾 OpenCV result saved: {cv2_output_path.name}")
    
    print(f"\n🎉 Demo complete! Check results in: {output_dir}")
    print("\n📊 Summary:")
    print("   - Green boxes: Digital signatures")
    print("   - Red boxes: Handwritten signatures") 
    print("   - Blue boxes: Blank signature blocks")
    print("   - Yellow boxes: Stamp signatures")


if __name__ == "__main__":
    demo_inference() 