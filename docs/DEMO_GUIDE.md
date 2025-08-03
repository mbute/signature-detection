# Signature Detection Demo Guide

This guide shows you how to run the signature detection demos and understand the results.

## 🚀 Quick Start

### 1. Simple Text Demo
```bash
python -m scripts.quick_demo
```
Shows a quick text-based demo of signature detection on test images.

### 2. Visual Demo with Bounding Boxes
```bash
python -m scripts.demo_inference
```
Creates visual results showing detected signatures with colored bounding boxes.

### 3. Interactive Demo (Ground Truth vs Predictions)
```bash
python -m scripts.interactive_demo
```
Shows side-by-side comparison of ground truth annotations vs model predictions.

## 📊 Understanding the Results

### Color Coding
- **🟢 Green**: Digital signatures
- **🔴 Red**: Handwritten signatures  
- **🔵 Blue**: Blank signature blocks
- **🟡 Yellow**: Stamp signatures

### Demo Outputs
- **Text Results**: Shows detection confidence scores
- **Visual Results**: Images with bounding boxes saved to `output/` directory
- **Comparison Results**: Side-by-side ground truth vs predictions

## 📁 Output Files

### Demo Results Location
```
output/
├── demo_results/          # Basic inference results
├── interactive_demo/      # Ground truth vs predictions
├── current_test_results/  # Current model testing
└── annotated_test_results/ # Testing on annotated images
```

### File Naming
- `demo_*.png`: Basic detection results
- `comparison_*.png`: Ground truth vs predictions
- `cv2_*.png`: OpenCV visualization
- `test_*.png`: Testing results

## 🔍 Current Model Performance

### Training Results
- **Overall mAP50**: 27.4%
- **Digital Signatures**: 81.7% mAP50 ✅
- **Handwritten Signatures**: 0.5% mAP50 ⚠️
- **Blank Signature Blocks**: 0% mAP50 ⚠️

### Key Insights
1. **Digital signatures are working well** - 81.7% accuracy
2. **Other classes need more training data** - small dataset limitation
3. **Model is detecting but with low confidence** - may need threshold adjustment

## 🛠️ Troubleshooting

### No Detections Found
If the model isn't detecting signatures:
1. **Check confidence threshold** - try lower values (0.1, 0.05)
2. **Verify model path** - ensure `best.pt` exists
3. **Check image format** - ensure images are PNG format
4. **Dataset size** - current dataset is small (21 images)

### Model Not Found
```bash
# Check if model exists
ls -la data/models/signature_detector/weights/
```

### No Test Images
```bash
# Check if images exist
ls -la data/processed/images/
```

## 🎯 Demo Scripts Summary

| Script | Purpose | Output |
|--------|---------|--------|
| `quick_demo.py` | Text-based demo | Console output |
| `demo_inference.py` | Visual detection | Images with boxes |
| `interactive_demo.py` | Compare predictions | Side-by-side images |

## 💡 Next Steps

1. **Expand Dataset**: Add more annotated signatures (aim for 100+)
2. **Improve Training**: Use `scripts/train_improved.py`
3. **Test on Real Documents**: Use actual PDFs for inference
4. **Fine-tune Parameters**: Adjust confidence thresholds

## 📞 Support

For issues or questions:
1. Check the main README.md
2. Review training logs in `data/models/`
3. Examine output images for visual debugging 