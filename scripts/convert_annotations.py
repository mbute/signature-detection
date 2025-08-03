#!/usr/bin/env python3
"""
Convert Pascal VOC XML annotations to YOLO format.
"""

import os
import xml.etree.ElementTree as ET
from pathlib import Path


def convert_voc_to_yolo(xml_path, classes_file, output_dir):
    """Convert a single XML file to YOLO format."""
    # Read classes
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    # Parse XML
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Get image dimensions
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    
    # Find all objects
    yolo_lines = []
    for obj in root.findall('object'):
        class_name = obj.find('name').text
        if class_name in classes:
            class_id = classes.index(class_name)
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            # Convert to YOLO format (normalized coordinates)
            x_center = (xmin + xmax) / 2.0 / width
            y_center = (ymin + ymax) / 2.0 / height
            w = (xmax - xmin) / width
            h = (ymax - ymin) / height
            
            yolo_lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}")
    
    return yolo_lines


def main():
    """Convert all XML annotations to YOLO format."""
    project_root = Path(__file__).parent.parent
    labels_dir = project_root / "data" / "labels"
    classes_file = labels_dir / "signature_classes.txt"
    
    print("🔄 Converting Pascal VOC annotations to YOLO format...")
    
    # Find all XML files
    xml_files = list(labels_dir.glob("*.xml"))
    
    if not xml_files:
        print("❌ No XML annotation files found")
        return
    
    print(f"📄 Found {len(xml_files)} XML annotation files")
    
    converted_count = 0
    
    for xml_file in xml_files:
        try:
            # Convert to YOLO format
            yolo_lines = convert_voc_to_yolo(xml_file, classes_file, labels_dir)
            
            # Write YOLO format file
            yolo_file = xml_file.with_suffix('.txt')
            with open(yolo_file, 'w') as f:
                f.write('\n'.join(yolo_lines))
            
            converted_count += 1
            print(f"  ✅ Converted: {xml_file.name} -> {yolo_file.name} ({len(yolo_lines)} annotations)")
            
        except Exception as e:
            print(f"  ❌ Error converting {xml_file.name}: {e}")
    
    print(f"\n🎉 Conversion complete!")
    print(f"📊 Converted {converted_count} files")
    print(f"📁 YOLO annotations saved to: {labels_dir}")


if __name__ == "__main__":
    main() 