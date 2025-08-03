#!/usr/bin/env python3
"""
Prepare PDF documents for annotation with LabelImg.
Converts PDFs to images and organizes them for easy annotation.
"""

import os
import sys
from pathlib import Path
import shutil

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.preprocessing.pdf_processor import PDFProcessor
from src.utils.config import get_config


def prepare_annotation_data():
    """Prepare PDF documents for annotation."""
    print("🔄 Preparing annotation data...")
    
    # Get paths
    project_root = Path(__file__).parent.parent
    raw_dir = project_root / "data" / "raw"
    processed_dir = project_root / "data" / "processed"
    images_dir = processed_dir / "images"
    labels_dir = project_root / "data" / "labels"
    
    # Create directories
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    
    # Find PDF files
    pdf_files = list(raw_dir.glob("*.pdf"))
    
    if not pdf_files:
        print("❌ No PDF files found in data/raw/")
        print("💡 Please place your PDF documents in data/raw/ first")
        return
    
    print(f"📄 Found {len(pdf_files)} PDF files")
    
    # Initialize PDF processor
    processor = PDFProcessor()
    
    total_images = 0
    
    for pdf_file in pdf_files:
        print(f"\n🔄 Processing {pdf_file.name}...")
        
        try:
            # Convert PDF to images
            images = processor.convert_pdf_to_images(str(pdf_file))
            
            # Save images
            pdf_stem = pdf_file.stem
            for i, image in enumerate(images):
                image_filename = f"{pdf_stem}_page_{i+1:03d}.png"
                image_path = images_dir / image_filename
                
                # Convert numpy array to PIL Image and save
                from PIL import Image
                if len(image.shape) == 3:
                    pil_image = Image.fromarray(image, mode='RGB')
                else:
                    pil_image = Image.fromarray(image, mode='L').convert('RGB')
                
                pil_image.save(image_path)
                total_images += 1
                
                print(f"  ✅ Saved: {image_filename}")
            
        except Exception as e:
            print(f"  ❌ Error processing {pdf_file.name}: {e}")
    
    print(f"\n🎉 Preparation complete!")
    print(f"📊 Total images prepared: {total_images}")
    print(f"📁 Images saved to: {images_dir}")
    print(f"📁 Labels will be saved to: {labels_dir}")
    print(f"\n🚀 To start annotation, run:")
    print(f"   ./scripts/annotation/start_labelimg.sh")


if __name__ == "__main__":
    prepare_annotation_data()
