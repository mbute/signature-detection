#!/bin/bash

# LabelImg launcher script for signature detection project
# This script sets up LabelImg with the correct directories and settings

echo "🎨 Starting LabelImg for Signature Detection Annotation"
echo "======================================================"

# Get the project root directory
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Set paths
IMAGES_DIR="$PROJECT_ROOT/data/processed/images"
LABELS_DIR="$PROJECT_ROOT/data/labels"
CLASSES_FILE="$PROJECT_ROOT/data/labels/signature_classes.txt"

echo "📁 Images directory: $IMAGES_DIR"
echo "📁 Labels directory: $LABELS_DIR"
echo "📋 Classes file: $CLASSES_FILE"
echo ""

# Check if directories exist
if [ ! -d "$IMAGES_DIR" ]; then
    echo "❌ Images directory not found: $IMAGES_DIR"
    echo "💡 Run the preparation script first: python -m scripts.prepare_annotation_data"
    exit 1
fi

if [ ! -f "$CLASSES_FILE" ]; then
    echo "❌ Classes file not found: $CLASSES_FILE"
    exit 1
fi

echo "🚀 Starting LabelImg..."
echo "📋 Annotation format: YOLO"
echo "💡 Tips:"
echo "   - Press 'W' to create a bounding box"
echo "   - Press 'D' to go to next image"
echo "   - Press 'A' to go to previous image"
echo "   - Press 'Ctrl+S' to save"
echo ""

# Start LabelImg with correct arguments
labelimg "$IMAGES_DIR" "$CLASSES_FILE" "$LABELS_DIR"
