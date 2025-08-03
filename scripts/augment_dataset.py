#!/usr/bin/env python3
"""
Data augmentation script to expand the signature detection dataset.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np
import shutil
import random

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def augment_dataset():
    """Augment the existing dataset to create more training examples."""
    project_root = Path(__file__).parent.parent
    
    # Define paths
    images_dir = project_root / "data" / "processed" / "images"
    labels_dir = project_root / "data" / "labels"
    augmented_dir = project_root / "data" / "augmented"
    
    # Create augmented directory
    augmented_images_dir = augmented_dir / "images"
    augmented_labels_dir = augmented_dir / "labels"
    augmented_images_dir.mkdir(parents=True, exist_ok=True)
    augmented_labels_dir.mkdir(parents=True, exist_ok=True)
    
    print("🔄 Starting dataset augmentation...")
    print(f"📁 Source images: {images_dir}")
    print(f"📁 Source labels: {labels_dir}")
    print(f"📁 Augmented output: {augmented_dir}")
    
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
    
    # Copy original files
    print("\n📋 Copying original files...")
    for img_file, label_file in annotated_images:
        # Copy image
        shutil.copy2(img_file, augmented_images_dir / img_file.name)
        # Copy label
        shutil.copy2(label_file, augmented_labels_dir / label_file.name)
    
    print(f"✅ Copied {len(annotated_images)} original files")
    
    # Augmentation parameters
    augmentations = [
        {'name': 'flip_horizontal', 'func': flip_horizontal},
        {'name': 'brightness_up', 'func': adjust_brightness, 'params': {'factor': 1.2}},
        {'name': 'brightness_down', 'func': adjust_brightness, 'params': {'factor': 0.8}},
        {'name': 'contrast_up', 'func': adjust_contrast, 'params': {'factor': 1.3}},
        {'name': 'contrast_down', 'func': adjust_contrast, 'params': {'factor': 0.7}},
        {'name': 'noise', 'func': add_noise, 'params': {'intensity': 0.02}},
        {'name': 'blur', 'func': add_blur, 'params': {'kernel_size': 3}},
    ]
    
    # Apply augmentations
    total_augmented = 0
    for img_file, label_file in annotated_images:
        print(f"\n🖼️  Augmenting: {img_file.name}")
        
        # Load image and labels
        image = cv2.imread(str(img_file))
        if image is None:
            continue
        
        # Read labels
        with open(label_file, 'r') as f:
            labels = f.read().strip().split('\n')
        
        # Apply each augmentation
        for aug in augmentations:
            try:
                aug_name = aug['name']
                aug_func = aug['func']
                aug_params = aug.get('params', {})
                
                # Apply augmentation
                if aug_params:
                    aug_image = aug_func(image.copy(), **aug_params)
                else:
                    aug_image = aug_func(image.copy())
                
                # Generate filename
                base_name = img_file.stem
                aug_filename = f"{base_name}_{aug_name}.png"
                aug_image_path = augmented_images_dir / aug_filename
                
                # Save augmented image
                cv2.imwrite(str(aug_image_path), aug_image)
                
                # Copy labels (for most augmentations, labels remain the same)
                aug_label_path = augmented_labels_dir / f"{base_name}_{aug_name}.txt"
                shutil.copy2(label_file, aug_label_path)
                
                # Special handling for horizontal flip
                if aug_name == 'flip_horizontal':
                    flip_labels(labels, aug_label_path, image.shape[1])
                
                total_augmented += 1
                print(f"   ✅ {aug_name}")
                
            except Exception as e:
                print(f"   ❌ {aug_name}: {e}")
    
    print(f"\n🎉 Augmentation complete!")
    print(f"📊 Original images: {len(annotated_images)}")
    print(f"📊 Augmented images: {total_augmented}")
    print(f"📊 Total images: {len(annotated_images) + total_augmented}")
    print(f"📁 Augmented files saved to: {augmented_dir}")
    
    # Create augmented dataset config
    create_augmented_dataset_config(augmented_dir, len(annotated_images) + total_augmented)


def flip_horizontal(image):
    """Flip image horizontally."""
    return cv2.flip(image, 1)


def adjust_brightness(image, factor=1.2):
    """Adjust image brightness."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def adjust_contrast(image, factor=1.3):
    """Adjust image contrast."""
    return cv2.convertScaleAbs(image, alpha=factor, beta=0)


def add_noise(image, intensity=0.02):
    """Add noise to image."""
    noise = np.random.normal(0, intensity * 255, image.shape).astype(np.uint8)
    noisy_image = cv2.add(image, noise)
    return np.clip(noisy_image, 0, 255).astype(np.uint8)


def add_blur(image, kernel_size=3):
    """Add slight blur to image."""
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def flip_labels(labels, output_path, image_width):
    """Flip YOLO labels horizontally."""
    flipped_labels = []
    
    for label in labels:
        if label.strip():
            parts = label.strip().split()
            if len(parts) == 5:
                class_id = parts[0]
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                
                # Flip x coordinate
                x_center_flipped = 1.0 - x_center
                
                flipped_labels.append(f"{class_id} {x_center_flipped:.6f} {y_center:.6f} {width:.6f} {height:.6f}")
    
    # Write flipped labels
    with open(output_path, 'w') as f:
        f.write('\n'.join(flipped_labels))


def create_augmented_dataset_config(augmented_dir, total_images):
    """Create dataset configuration for augmented dataset."""
    dataset_yaml = augmented_dir / "dataset.yaml"
    
    # Read class names from original
    classes_file = Path(__file__).parent.parent / "data" / "labels" / "signature_classes.txt"
    with open(classes_file, 'r') as f:
        classes = [line.strip() for line in f.readlines()]
    
    yaml_content = f"""# Augmented dataset configuration for YOLO training
path: {augmented_dir.absolute()}
train: images
val: images  # Use same images for validation in small dataset

# Number of classes
nc: {len(classes)}

# Class names
names:
"""
    
    for i, class_name in enumerate(classes):
        yaml_content += f"  {i}: {class_name}\n"
    
    with open(dataset_yaml, 'w') as f:
        f.write(yaml_content)
    
    print(f"📋 Created dataset config: {dataset_yaml}")
    print(f"📊 Total images: {total_images}")
    print(f"📊 Classes: {classes}")


if __name__ == "__main__":
    augment_dataset() 