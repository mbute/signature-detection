# Quick Start Guide

This guide will help you get the Signature Detection & Compliance Checker up and running quickly.

## Prerequisites

- Python 3.8 or higher
- macOS, Linux, or Windows
- Git (for cloning the repository)

## Installation

### 1. Clone the Repository

```bash
git clone <repository-url>
cd signature_detection
```

### 2. Set Up Environment

**Option A: Automated Setup (Recommended)**
```bash
python scripts/setup_environment.py
```

**Option B: Manual Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install system dependencies (macOS)
brew install tesseract poppler

# Install system dependencies (Ubuntu/Debian)
sudo apt-get install tesseract-ocr tesseract-ocr-eng poppler-utils
```

### 3. Configure the System

```bash
# Copy example configuration
cp config/config.example.yaml config/config.yaml

# Edit configuration if needed
nano config/config.yaml
```

## Basic Usage

### 1. Detect Signatures in a PDF

```bash
# Basic detection
python -m src.main detect path/to/your/document.pdf

# Save results to output directory
python -m src.main detect path/to/your/document.pdf --output results/

# Save processed images
python -m src.main detect path/to/your/document.pdf --output results/ --save-images
```

### 2. Check Compliance

```bash
# Check compliance (auto-detect document type)
python -m src.main check path/to/your/document.pdf

# Specify document type
python -m src.main check path/to/your/document.pdf --document-type award_decision

# Save compliance report
python -m src.main check path/to/your/document.pdf --output results/
```

### 3. Batch Processing

```bash
# Process all PDFs in a directory
python -m src.main batch path/to/pdf/directory/

# Save batch results
python -m src.main batch path/to/pdf/directory/ --output results/
```

### 4. System Information

```bash
# Check system status and available components
python -m src.main info
```

## Example Workflow

1. **Prepare your documents:**
   ```bash
   # Copy PDFs to the data directory
   cp your_documents/*.pdf data/raw/
   ```

2. **Run detection:**
   ```bash
   python -m src.main detect data/raw/document1.pdf --output results/
   ```

3. **Check compliance:**
   ```bash
   python -m src.main check data/raw/document1.pdf --output results/
   ```

4. **Review results:**
   ```bash
   # View detection results
   cat results/detection_results.json
   
   # View compliance report
   cat results/compliance_report.json
   ```

## Configuration

The system uses a YAML configuration file (`config/config.yaml`) with the following key sections:

- **Model**: YOLO model settings and signature types
- **OCR**: Text extraction engine and settings
- **PDF**: Processing parameters (DPI, preprocessing)
- **Compliance**: Document types and required signatures
- **Data**: File paths and training settings
- **Logging**: Log levels and output settings

### Key Configuration Options

```yaml
# Model settings
model:
  yolo:
    model_size: "yolov8n"  # Model size: yolov8n, yolov8s, yolov8m, yolov8l, yolov8x
    confidence_threshold: 0.5  # Detection confidence threshold

# OCR settings
ocr:
  engine: "paddleocr"  # OCR engine: paddleocr, tesseract, easyocr
  text_extraction:
    min_confidence: 0.6  # Minimum text confidence

# PDF processing
pdf:
  dpi: 300  # Image resolution
  preprocessing:
    deskew: true  # Auto-rotate skewed pages
    denoise: true  # Remove noise
```

## Troubleshooting

### Common Issues

1. **"No OCR engines available"**
   - Install PaddleOCR: `pip install paddleocr`
   - Or install Tesseract: `brew install tesseract` (macOS) / `sudo apt-get install tesseract-ocr` (Ubuntu)

2. **"PDF conversion failed"**
   - Install Poppler: `brew install poppler` (macOS) / `sudo apt-get install poppler-utils` (Ubuntu)

3. **"YOLO model not found"**
   - The system will automatically download the default YOLO model on first use
   - Or specify a custom model: `--model path/to/your/model.pt`

4. **"Configuration file not found"**
   - Copy the example config: `cp config/config.example.yaml config/config.yaml`

### Getting Help

- Check system status: `python -m src.main info`
- Enable verbose logging: `python -m src.main --verbose detect document.pdf`
- Review logs: `tail -f logs/signature_detection.log`

## Next Steps

- **Training Custom Models**: See `docs/TRAINING.md`
- **Advanced Configuration**: See `docs/CONFIGURATION.md`
- **API Usage**: See `docs/API.md`
- **Contributing**: See `CONTRIBUTING.md`

## Support

For issues and questions:
1. Check the troubleshooting section above
2. Review the logs in `logs/signature_detection.log`
3. Open an issue on the project repository
4. Check the documentation in the `docs/` directory 