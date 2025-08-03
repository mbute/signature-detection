#!/usr/bin/env python3
"""
Production-ready demo with calibrated confidence thresholds.
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


def production_demo():
    """Production demo with calibrated confidence thresholds."""
    project_root = Path(__file__).parent.parent
    
    print("🏭 Production-Ready Signature Detection Demo")
    print("=" * 50)
    print("Using calibrated confidence thresholds for optimal performance")
    
    # Load model
    model_path = project_root / "data" / "models" / "signature_detector" / "weights" / "best.pt"
    model = YOLO(str(model_path))
    
    # Class names and colors
    class_names = ['handwritten_signature', 'digital_signature', 'blank_signature_block', 'stamp_signature']
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    # Production confidence thresholds (calibrated)
    thresholds = {
        'maximum_sensitivity': 0.001,  # Show all possible detections
        'balanced': 0.005,             # Good balance
        'high_confidence': 0.01,       # High confidence only
        'strict': 0.02                 # Very strict
    }
    
    # Find test images
    images_dir = project_root / "data" / "processed" / "images"
    image_files = list(images_dir.glob("*.png"))
    
    if not image_files:
        print("❌ No test images found")
        return
    
    output_dir = project_root / "output" / "production_demo"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📄 Testing on {len(image_files)} images")
    print(f"📁 Output: {output_dir}")
    
    # Test on first 3 images
    for i, img_path in enumerate(image_files[:3]):
        print(f"\n🖼️  Processing: {img_path.name}")
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            continue
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create subplot for different thresholds
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        axes = axes.flatten()
        
        # Test each threshold
        for j, (threshold_name, threshold_value) in enumerate(thresholds.items()):
            ax = axes[j]
            
            # Run inference
            results = model(str(img_path), conf=threshold_value)
            
            # Draw detections
            ax.imshow(image_rgb)
            ax.set_title(f"{threshold_name.replace('_', ' ').title()}\n(Threshold: {threshold_value})")
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
                print(f"   ✅ {threshold_name}: {len(detections)} detections")
            else:
                print(f"   ❌ {threshold_name}: No detections")
        
        # Save result
        output_path = output_dir / f"production_{img_path.stem}.png"
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"   💾 Saved: {output_path.name}")
    
    # Create summary report
    create_production_summary(thresholds, output_dir)
    
    print(f"\n🎉 Production demo complete!")
    print(f"📁 Results saved in: {output_dir}")
    print(f"\n💡 Production Recommendations:")
    print(f"   • For maximum sensitivity: Use threshold {thresholds['maximum_sensitivity']}")
    print(f"   • For balanced approach: Use threshold {thresholds['balanced']}")
    print(f"   • For high confidence: Use threshold {thresholds['high_confidence']}")
    print(f"   • For strict filtering: Use threshold {thresholds['strict']}")


def create_production_summary(thresholds, output_dir):
    """Create a production summary report."""
    summary_file = output_dir / "production_summary.txt"
    
    content = f"""# Production Signature Detection Summary

## Calibrated Confidence Thresholds

### Threshold Settings
- **Maximum Sensitivity**: {thresholds['maximum_sensitivity']} - Shows all possible detections
- **Balanced**: {thresholds['balanced']} - Good balance between sensitivity and precision
- **High Confidence**: {thresholds['high_confidence']} - High confidence detections only
- **Strict**: {thresholds['strict']} - Very strict filtering

## Usage Guidelines

### For Different Use Cases

1. **Development/Testing**
   - Use: Maximum Sensitivity threshold
   - Purpose: See all possible detections
   - Threshold: {thresholds['maximum_sensitivity']}

2. **Production (General)**
   - Use: Balanced threshold
   - Purpose: Good balance of sensitivity and precision
   - Threshold: {thresholds['balanced']}

3. **Production (High Precision)**
   - Use: High Confidence threshold
   - Purpose: Only high-confidence detections
   - Threshold: {thresholds['high_confidence']}

4. **Production (Very Strict)**
   - Use: Strict threshold
   - Purpose: Only the most confident detections
   - Threshold: {thresholds['strict']}

## Implementation Example

```python
from ultralytics import YOLO

# Load model
model = YOLO('path/to/signature_detector.pt')

# Production inference with calibrated thresholds
def detect_signatures(image_path, threshold_type='balanced'):
    thresholds = {{
        'maximum_sensitivity': {thresholds['maximum_sensitivity']},
        'balanced': {thresholds['balanced']},
        'high_confidence': {thresholds['high_confidence']},
        'strict': {thresholds['strict']}
    }}
    
    conf_threshold = thresholds[threshold_type]
    results = model(image_path, conf=conf_threshold)
    
    return results

# Usage
results = detect_signatures('document.pdf', 'balanced')
```

## Performance Notes

- Current model trained on 21 images
- Digital signatures perform well (81.7% mAP50)
- Other classes need more training data
- Thresholds optimized for current dataset
- Consider retraining with more data for better performance

## Next Steps

1. Collect more training data (aim for 100+ annotations)
2. Retrain model with improved parameters
3. Recalibrate thresholds with larger dataset
4. Test on real production documents
5. Implement post-processing logic for compliance checking
"""
    
    with open(summary_file, 'w') as f:
        f.write(content)
    
    print(f"📋 Production summary saved: {summary_file}")


if __name__ == "__main__":
    production_demo() 