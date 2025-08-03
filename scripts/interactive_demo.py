#!/usr/bin/env python3
"""
Interactive demo showing model predictions vs ground truth annotations.
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


def interactive_demo():
    """Interactive demo showing predictions vs ground truth."""
    project_root = Path(__file__).parent.parent
    
    # Define paths
    model_path = project_root / "data" / "models" / "signature_detector" / "weights" / "best.pt"
    images_dir = project_root / "data" / "processed" / "images"
    labels_dir = project_root / "data" / "labels"
    output_dir = project_root / "output" / "interactive_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🎯 Interactive Signature Detection Demo")
    print("=" * 50)
    print("This demo shows model predictions vs ground truth annotations")
    
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
            print("❌ No trained model found.")
            return
    
    # Load model
    print("🔄 Loading model...")
    model = YOLO(str(model_path))
    
    # Class names and colors
    class_names = ['handwritten_signature', 'digital_signature', 'blank_signature_block', 'stamp_signature']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    # Find annotated images
    annotated_images = []
    for label_file in labels_dir.glob("*.txt"):
        img_file = images_dir / f"{label_file.stem}.png"
        if img_file.exists():
            annotated_images.append((img_file, label_file))
    
    if not annotated_images:
        print("❌ No annotated images found")
        return
    
    print(f"📄 Found {len(annotated_images)} annotated images")
    
    # Process first 5 annotated images
    for i, (img_path, label_path) in enumerate(annotated_images[:5]):
        print(f"\n🖼️  Image {i+1}: {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        height, width = image.shape[:2]
        
        # Load ground truth annotations
        ground_truth = []
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 5:
                    class_id = int(parts[0])
                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    w = float(parts[3]) * width
                    h = float(parts[4]) * height
                    
                    x1 = int(x_center - w/2)
                    y1 = int(y_center - h/2)
                    x2 = int(x_center + w/2)
                    y2 = int(y_center + h/2)
                    
                    ground_truth.append({
                        'class_id': class_id,
                        'class_name': class_names[class_id],
                        'bbox': [x1, y1, x2, y2]
                    })
        
        # Run model inference
        results = model(str(img_path), conf=0.01)
        
        # Create visualization
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 8))
        
        # Original image
        ax1.imshow(image_rgb)
        ax1.set_title("Original Image")
        ax1.axis('off')
        
        # Ground truth
        ax2.imshow(image_rgb)
        ax2.set_title(f"Ground Truth ({len(ground_truth)} annotations)")
        ax2.axis('off')
        
        # Model predictions
        ax3.imshow(image_rgb)
        ax3.set_title("Model Predictions")
        ax3.axis('off')
        
        # Draw ground truth
        for gt in ground_truth:
            x1, y1, x2, y2 = gt['bbox']
            color = colors[gt['class_id']]
            rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                           linewidth=2, edgecolor=[c/255 for c in color], 
                           facecolor='none', linestyle='-')
            ax2.add_patch(rect)
            ax2.text(x1, y1-10, gt['class_name'], 
                    color=[c/255 for c in color], fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Draw model predictions
        predictions_found = False
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
                    
                    rect = Rectangle((x1, y1), x2-x1, y2-y1, 
                                   linewidth=2, edgecolor=[c/255 for c in color], 
                                   facecolor='none', linestyle='--')
                    ax3.add_patch(rect)
                    ax3.text(x1, y1-10, f"{class_name}: {conf:.2f}", 
                            color=[c/255 for c in color], fontsize=10,
                            bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
                    predictions_found = True
        
        if not predictions_found:
            ax3.text(0.5, 0.5, "No Predictions", transform=ax3.transAxes, 
                    ha='center', va='center', fontsize=16, color='red')
        
        # Save result
        output_path = output_dir / f"comparison_{img_path.stem}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   📊 Ground truth: {len(ground_truth)} annotations")
        print(f"   🎯 Model predictions: {'Found' if predictions_found else 'None'}")
        print(f"   💾 Saved: {output_path.name}")
    
    print(f"\n🎉 Interactive demo complete!")
    print(f"📁 Results saved in: {output_dir}")
    print("\n📋 Legend:")
    print("   - Solid lines: Ground truth annotations")
    print("   - Dashed lines: Model predictions")
    print("   - Green: Digital signatures")
    print("   - Red: Handwritten signatures")
    print("   - Blue: Blank signature blocks")
    print("   - Yellow: Stamp signatures")


if __name__ == "__main__":
    interactive_demo() 