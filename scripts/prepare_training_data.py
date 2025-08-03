#!/usr/bin/env python3
"""
Prepare training dataset by splitting images and annotations into train/val sets.
"""

import os
import shutil
import random
from pathlib import Path


def prepare_training_data():
    """Prepare training dataset with train/val split."""
    project_root = Path(__file__).parent.parent
    
    # Define paths
    images_dir = project_root / "data" / "processed" / "images"
    labels_dir = project_root / "data" / "labels"
    dataset_dir = project_root / "data" / "dataset"
    
    # Create dataset directories
    train_images_dir = dataset_dir / "train" / "images"
    train_labels_dir = dataset_dir / "train" / "labels"
    val_images_dir = dataset_dir / "val" / "images"
    val_labels_dir = dataset_dir / "val" / "labels"
    
    for dir_path in [train_images_dir, train_labels_dir, val_images_dir, val_labels_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Preparing training dataset...")
    
    # Find all annotated images (those with corresponding label files)
    annotated_images = []
    for img_file in images_dir.glob("*.png"):
        label_file = labels_dir / f"{img_file.stem}.txt"
        if label_file.exists():
            annotated_images.append((img_file, label_file))
    
    if not annotated_images:
        print("❌ No annotated images found")
        return
    
    print(f"📄 Found {len(annotated_images)} annotated images")
    
    # Shuffle and split
    random.shuffle(annotated_images)
    split_idx = int(len(annotated_images) * 0.8)  # 80% train, 20% val
    
    train_data = annotated_images[:split_idx]
    val_data = annotated_images[split_idx:]
    
    print(f"📊 Train set: {len(train_data)} images")
    print(f"📊 Val set: {len(val_data)} images")
    
    # Copy training data
    for img_file, label_file in train_data:
        shutil.copy2(img_file, train_images_dir / img_file.name)
        shutil.copy2(label_file, train_labels_dir / label_file.name)
    
    # Copy validation data
    for img_file, label_file in val_data:
        shutil.copy2(img_file, val_images_dir / img_file.name)
        shutil.copy2(label_file, val_labels_dir / label_file.name)
    
    # Create dataset.yaml file
    dataset_yaml = dataset_dir / "dataset.yaml"
    classes_file = labels_dir / "signature_classes.txt"
    
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    yaml_content = f"""# Dataset configuration for YOLO training
path: {dataset_dir.absolute()}
train: train/images
val: val/images

# Number of classes
nc: {len(classes)}

# Class names
names:
"""
    
    for i, class_name in enumerate(classes):
        yaml_content += f"  {i}: {class_name}\n"
    
    with open(dataset_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"\n🎉 Dataset preparation complete!")
    print(f"📁 Dataset saved to: {dataset_dir}")
    print(f"📋 Configuration: {dataset_yaml}")
    print(f"📊 Classes: {classes}")
    
    return dataset_yaml


if __name__ == "__main__":
    prepare_training_data() 