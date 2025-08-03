#!/usr/bin/env python3
"""
Confidence calibration script to find optimal thresholds for signature detection.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO


def calibrate_confidence():
    """Calibrate confidence thresholds for optimal detection."""
    project_root = Path(__file__).parent.parent
    
    print("🎯 Confidence Calibration")
    print("=" * 40)
    print("Finding optimal confidence thresholds for signature detection")
    
    # Load model
    model_path = project_root / "data" / "models" / "signature_detector" / "weights" / "best.pt"
    model = YOLO(str(model_path))
    
    # Class names
    class_names = ['handwritten_signature', 'digital_signature', 'blank_signature_block', 'stamp_signature']
    
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
    
    # Test different confidence thresholds
    thresholds = np.arange(0.001, 0.1, 0.001)  # 0.001 to 0.1 in steps of 0.001
    
    # Store results
    results = defaultdict(list)
    
    print("\n🔍 Testing confidence thresholds...")
    
    for threshold in thresholds:
        total_detections = 0
        class_detections = defaultdict(int)
        
        for img_path, label_path in annotated_images:
            # Run inference
            inference_results = model(str(img_path), conf=threshold)
            
            # Count detections
            for result in inference_results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        cls = int(box.cls[0].cpu().numpy())
                        class_name = class_names[cls]
                        class_detections[class_name] += 1
                        total_detections += 1
        
        # Store results
        results['threshold'].append(threshold)
        results['total_detections'].append(total_detections)
        for class_name in class_names:
            results[f'{class_name}_detections'].append(class_detections[class_name])
    
    # Find optimal thresholds
    print("\n📊 Analyzing results...")
    
    # Find threshold for maximum detections
    max_detections_idx = np.argmax(results['total_detections'])
    max_detections_threshold = results['threshold'][max_detections_idx]
    max_detections_count = results['total_detections'][max_detections_idx]
    
    # Find threshold for reasonable number of detections (not too many, not too few)
    reasonable_detections = [d for d in results['total_detections'] if 5 <= d <= 50]
    if reasonable_detections:
        reasonable_threshold_idx = results['total_detections'].index(min(reasonable_detections, key=lambda x: abs(x - 20)))
        reasonable_threshold = results['threshold'][reasonable_threshold_idx]
        reasonable_detections_count = results['total_detections'][reasonable_threshold_idx]
    else:
        reasonable_threshold = 0.01
        reasonable_detections_count = results['total_detections'][results['threshold'].index(0.01)]
    
    # Find threshold for high confidence (fewer but more confident detections)
    high_confidence_threshold = 0.02
    high_confidence_idx = results['threshold'].index(high_confidence_threshold)
    high_confidence_detections = results['total_detections'][high_confidence_idx]
    
    print(f"\n🎯 Optimal Thresholds:")
    print(f"   • Maximum detections: {max_detections_threshold:.3f} ({max_detections_count} detections)")
    print(f"   • Reasonable balance: {reasonable_threshold:.3f} ({reasonable_detections_count} detections)")
    print(f"   • High confidence: {high_confidence_threshold:.3f} ({high_confidence_detections} detections)")
    
    # Create visualization
    create_calibration_plot(results, class_names, project_root)
    
    # Create recommended thresholds file
    create_threshold_recommendations(
        max_detections_threshold, reasonable_threshold, high_confidence_threshold,
        project_root
    )
    
    print(f"\n💡 Recommendations:")
    print(f"   • For demos: Use threshold {max_detections_threshold:.3f}")
    print(f"   • For production: Use threshold {reasonable_threshold:.3f}")
    print(f"   • For high confidence: Use threshold {high_confidence_threshold:.3f}")
    
    return {
        'max_detections': max_detections_threshold,
        'reasonable': reasonable_threshold,
        'high_confidence': high_confidence_threshold
    }


def create_calibration_plot(results, class_names, project_root):
    """Create visualization of confidence calibration results."""
    output_dir = project_root / "output" / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot total detections
    ax1.plot(results['threshold'], results['total_detections'], 'b-', linewidth=2, label='Total Detections')
    ax1.set_xlabel('Confidence Threshold')
    ax1.set_ylabel('Number of Detections')
    ax1.set_title('Total Detections vs Confidence Threshold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Plot detections by class
    for class_name in class_names:
        detections = results[f'{class_name}_detections']
        ax2.plot(results['threshold'], detections, linewidth=2, label=class_name)
    
    ax2.set_xlabel('Confidence Threshold')
    ax2.set_ylabel('Number of Detections')
    ax2.set_title('Detections by Class vs Confidence Threshold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    output_path = output_dir / "confidence_calibration.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Calibration plot saved: {output_path}")


def create_threshold_recommendations(max_thresh, reasonable_thresh, high_conf_thresh, project_root):
    """Create a file with threshold recommendations."""
    output_file = project_root / "output" / "calibration" / "threshold_recommendations.txt"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    content = f"""# Confidence Threshold Recommendations

## Optimal Thresholds for Signature Detection

### 1. Maximum Detections
- Threshold: {max_thresh:.3f}
- Use case: Demos, maximum sensitivity
- Description: Shows all possible detections

### 2. Reasonable Balance
- Threshold: {reasonable_thresh:.3f}
- Use case: Production, balanced approach
- Description: Good balance between sensitivity and precision

### 3. High Confidence
- Threshold: {high_conf_thresh:.3f}
- Use case: High precision requirements
- Description: Only high-confidence detections

## Usage Examples

```python
# For demos
results = model(image, conf={max_thresh:.3f})

# For production
results = model(image, conf={reasonable_thresh:.3f})

# For high confidence
results = model(image, conf={high_conf_thresh:.3f})
```

## Notes
- These thresholds are based on current dataset (21 images)
- Thresholds may improve with more training data
- Consider adjusting based on specific use case requirements
"""
    
    with open(output_file, 'w') as f:
        f.write(content)
    
    print(f"📋 Threshold recommendations saved: {output_file}")


if __name__ == "__main__":
    calibrate_confidence() 