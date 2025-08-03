# LabelImg Guide for Signature Detection

## 🎨 What is LabelImg?

LabelImg is a graphical image annotation tool for labeling objects in images. It's perfect for creating training data for your signature detection model.

## 🚀 Quick Start

### 1. Prepare Your Data
```bash
# Convert PDFs to images for annotation
python scripts/prepare_annotation_data.py
```

### 2. Start LabelImg
```bash
# Use the provided script (recommended)
./scripts/annotation/start_labelimg.sh

# Or start manually
labelimg data/processed/images data/labels --format yolo
```

## 📋 Annotation Workflow

### Step 1: Open LabelImg
- Run the start script or command above
- LabelImg will open with your images loaded

### Step 2: Set Up Classes
1. Click **View → Auto Save** (recommended)
2. Click **View → Display Labels** (to see your annotations)
3. **Important**: Set up your signature classes first!

### Step 3: Define Signature Classes
1. Go to **Edit → Edit Labels**
2. Add these classes (one per line):
   ```
   handwritten_signature
   digital_signature
   blank_signature_block
   stamp_signature
   ```
3. Click **OK**

### Step 4: Annotate Signatures
1. **Navigate**: Use `A` (previous) and `D` (next) to move between images
2. **Create Box**: Press `W` or click **Create RectBox**
3. **Draw Rectangle**: Click and drag to create bounding box around signature
4. **Select Class**: Choose the appropriate signature type from dropdown
5. **Save**: Press `Ctrl+S` or click **Save**

### Step 5: Quality Control
- **Review**: Go through all images to ensure consistent labeling
- **Verify**: Check that all signatures are properly bounded
- **Export**: Labels are automatically saved in YOLO format

## 🎯 Annotation Guidelines

### What to Label
- ✅ **Handwritten signatures**: Pen/ink signatures
- ✅ **Digital signatures**: Electronic signatures, typed names
- ✅ **Blank signature blocks**: Empty signature lines
- ✅ **Stamp signatures**: Official stamps, seals
- ✅ **Signature lines**: Lines where signatures should go

### What NOT to Label
- ❌ **Text that's not a signature**: Regular document text
- ❌ **Logos or graphics**: Company logos, decorative elements
- ❌ **Page numbers or headers**: Document metadata

### Bounding Box Tips
- **Tight fit**: Make boxes just large enough to contain the signature
- **Include context**: Include nearby text that identifies the signer
- **Consistent size**: Try to be consistent with box sizes for similar signatures
- **Multiple signatures**: Label each signature separately

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `W` | Create bounding box |
| `A` | Previous image |
| `D` | Next image |
| `Ctrl+S` | Save |
| `Ctrl+O` | Open image |
| `Ctrl+N` | Open directory |
| `Del` | Delete selected box |
| `Ctrl+D` | Duplicate selected box |

## 📁 File Organization

After annotation, your directory structure will look like:

```
data/
├── processed/
│   └── images/           # PNG images from PDFs
│       ├── doc1_page_001.png
│       ├── doc1_page_002.png
│       └── ...
└── labels/              # YOLO format labels
    ├── doc1_page_001.txt
    ├── doc1_page_002.txt
    └── ...
```

## 📊 YOLO Format

Each label file contains one line per signature:
```
class_id center_x center_y width height
```

Example:
```
0 0.5 0.3 0.2 0.1    # handwritten_signature
1 0.7 0.8 0.15 0.05  # digital_signature
```

## 🔧 Troubleshooting

### LabelImg Won't Start
```bash
# Try installing with conda
conda install -c conda-forge labelimg

# Or install from source
git clone https://github.com/tzutalin/labelImg.git
cd labelImg
pip install -r requirements/requirements-linux.txt
python labelImg.py
```

### Images Not Loading
- Check that images are in PNG/JPG format
- Ensure images directory path is correct
- Verify file permissions

### Labels Not Saving
- Check write permissions on labels directory
- Ensure "Auto Save" is enabled
- Verify disk space

## �� Best Practices

### 1. Start Small
- Begin with 10-20 images to get familiar with the tool
- Establish consistent annotation patterns
- Review and refine your approach

### 2. Quality Over Quantity
- Take time to create accurate bounding boxes
- Be consistent with class assignments
- Review your annotations regularly

### 3. Document Your Process
- Keep notes on annotation decisions
- Document any special cases or edge cases
- Create guidelines for your team

### 4. Regular Backups
- Save your work frequently
- Back up your labels directory
- Version control your annotation guidelines

## 🎯 Next Steps

After completing annotation:

1. **Review annotations** for consistency
2. **Split data** into train/validation sets
3. **Train your model** using the annotated data
4. **Validate results** on test images
5. **Iterate and improve** based on results

## 📞 Getting Help

- **LabelImg Issues**: Check the [LabelImg GitHub](https://github.com/tzutalin/labelImg)
- **Annotation Questions**: Review this guide or ask your team
- **Technical Problems**: Check the troubleshooting section above
