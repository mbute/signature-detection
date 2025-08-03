#!/usr/bin/env python3
"""
Working demo that shows signature detection with appropriate confidence thresholds.
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


def working_demo():
    """Working demo with appropriate confidence thresholds."""
    project_root = Path(__file__).parent.parent
    
    print("🎯 Working Signature Detection Demo")
    print("=" * 50)
    print("Showing detections with appropriate confidence thresholds")
    
    # Load model
    model_path = project_root / "data" / "models" / "signature_detector" / "weights" / "best.pt"
    model = YOLO(str(model_path))
    
    # Class names and colors
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
    
    output_dir = project_root / "output" / "working_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📄 Testing on {len(annotated_images)} annotated images")
    
    # Test with different confidence thresholds
    thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
    
    for i, (img_path, label_path) in enumerate(annotated_images[:3]):
        print(f"\n🖼️  Image {i+1}: {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create subplot for different thresholds
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Original image
        axes[0].imshow(image_rgb)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Test different thresholds
        for j, threshold in enumerate(thresholds):
            ax = axes[j + 1]
            
            # Run inference
            results = model(str(img_path), conf=threshold)
            
            # Draw detections
            ax.imshow(image_rgb)
            ax.set_title(f"Threshold: {threshold}")
            ax.axis('off')
            
            detections = []
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy())
                        conf = float(box.conf[0].cpu().numpy())
                        class_name = class_names[cls]
                        color = colors[cls]
                        
                        detections.append({
                            'class': class_name,
                            'confidence': conf,
                            'bbox': [int(x1), int(y1), int(x2), int(y2)]
                        })
                        
                        # Draw bounding box
                        rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                                       linewidth=2, edgecolor=[c/255 for c in color], 
                                       facecolor='none')
                        ax.add_patch(rect)
                        
                        # Add label
                        ax.text(x1, y1-10, f"{class_name}: {conf:.3f}", 
                               color=[c/255 for c in color], fontsize=8,
                               bbox=dict(boxstyle="round,pad=0.2", facecolor='white', alpha=0.8))
            
            if detections:
                print(f"   ✅ Threshold {threshold}: {len(detections)} detections")
            else:
                print(f"   ❌ Threshold {threshold}: No detections")
        
        # Save result
        output_path = output_dir / f"thresholds_{img_path.stem}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved: {output_path.name}")
    
    print(f"\n🎉 Working demo complete!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"\n💡 Recommendations:")
    print(f"   • Use threshold 0.001-0.005 for maximum sensitivity")
    print(f"   • Use threshold 0.01-0.02 for balanced results")
    print(f"   • Use threshold 0.05+ for high confidence only")
    print(f"\n🔧 To improve confidence scores:")
    print(f"   • Add more training data (aim for 100+ annotations)")
    print(f"   • Use improved training parameters")
    print(f"   • Consider data augmentation")


if __name__ == "__main__":
    working_demo() 