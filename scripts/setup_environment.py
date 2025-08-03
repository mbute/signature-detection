#!/usr/bin/env python3
"""
Setup script for Signature Detection & Compliance Checker.
"""

import subprocess
import sys
import os
import time
from pathlib import Path


def run_command_with_progress(command, description, show_output=True):
    """Run a command and show real-time progress."""
    print(f"🔄 {description}...")
    
    if show_output:
        # Run with real-time output for pip installs
        try:
            process = subprocess.Popen(
                command, 
                shell=True, 
                stdout=subprocess.PIPE, 
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1
            )
            
            # Print output in real-time
            for line in process.stdout:
                print(line.rstrip())
            
            process.wait()
            
            if process.returncode == 0:
                print(f"✅ {description} completed successfully")
                return True
            else:
                print(f"❌ {description} failed with return code {process.returncode}")
                return False
                
        except Exception as e:
            print(f"❌ {description} failed: {e}")
            return False
    else:
        # Run silently for quick commands
        try:
            result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            print(f"✅ {description} completed successfully")
            return True
        except subprocess.CalledProcessError as e:
            print(f"❌ {description} failed: {e}")
            print(f"Error output: {e.stderr}")
            return False


def check_python_version():
    """Check if Python version is compatible."""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True


def install_dependencies():
    """Install Python dependencies with progress tracking."""
    print("\n📦 Installing Python dependencies...")
    
    # Upgrade pip (quick, no need for verbose output)
    print("🔄 Upgrading pip...")
    if not run_command_with_progress("pip install --upgrade pip", "Upgrading pip", show_output=False):
        return False
    
    # Install requirements with verbose output
    requirements_file = Path(__file__).parent.parent / "requirements.txt"
    print(f"🔄 Installing requirements from {requirements_file}...")
    print("📋 This may take several minutes for large packages like PyTorch and PaddlePaddle...")
    print("⏳ You'll see real-time progress below:\n")
    
    if not run_command_with_progress(
        f"pip install -r {requirements_file} -v --progress-bar on", 
        "Installing requirements", 
        show_output=True
    ):
        return False
    
    return True


def setup_configuration():
    """Set up configuration files."""
    print("\n⚙️  Setting up configuration...")
    
    config_dir = Path(__file__).parent.parent / "config"
    example_config = config_dir / "config.example.yaml"
    config_file = config_dir / "config.yaml"
    
    if not config_file.exists() and example_config.exists():
        import shutil
        shutil.copy(example_config, config_file)
        print(f"✅ Configuration file created: {config_file}")
    else:
        print(f"ℹ️  Configuration file already exists: {config_file}")
    
    return True


def create_directories():
    """Create necessary directories."""
    print("\n📁 Creating directories...")
    
    base_dir = Path(__file__).parent.parent
    directories = [
        "data/raw",
        "data/processed",
        "data/models",
        "data/labels",
        "output",
        "logs"
    ]
    
    for dir_path in directories:
        full_path = base_dir / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created directory: {dir_path}")
    
    return True


def check_system_dependencies():
    """Check for system dependencies."""
    print("\n🔍 Checking system dependencies...")
    
    # Check for Tesseract
    try:
        result = subprocess.run(["tesseract", "--version"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Tesseract is installed")
        else:
            print("⚠️  Tesseract not found - install with: brew install tesseract (macOS) or apt-get install tesseract-ocr (Ubuntu)")
    except FileNotFoundError:
        print("⚠️  Tesseract not found - install with: brew install tesseract (macOS) or apt-get install tesseract-ocr (Ubuntu)")
    
    # Check for Poppler (for pdf2image)
    try:
        result = subprocess.run(["pdftoppm", "-h"], capture_output=True, text=True)
        if result.returncode == 0:
            print("✅ Poppler is installed")
        else:
            print("⚠️  Poppler not found - install with: brew install poppler (macOS) or apt-get install poppler-utils (Ubuntu)")
    except FileNotFoundError:
        print("⚠️  Poppler not found - install with: brew install poppler (macOS) or apt-get install poppler-utils (Ubuntu)")
    
    return True


def test_installation():
    """Test the installation."""
    print("\n🧪 Testing installation...")
    
    try:
        # Test imports
        import cv2
        print("✅ OpenCV imported successfully")
        
        import numpy as np
        print("✅ NumPy imported successfully")
        
        import fitz
        print("✅ PyMuPDF imported successfully")
        
        # Test configuration
        sys.path.append(str(Path(__file__).parent.parent))
        from src.utils.config import get_config
        config = get_config()
        print("✅ Configuration loaded successfully")
        
        print("✅ Installation test completed successfully")
        return True
        
    except ImportError as e:
        print(f"❌ Import test failed: {e}")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    """Main setup function."""
    print("🚀 Setting up Signature Detection & Compliance Checker")
    print("=" * 60)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create directories
    if not create_directories():
        sys.exit(1)
    
    # Setup configuration
    if not setup_configuration():
        sys.exit(1)
    
    # Install dependencies
    if not install_dependencies():
        print("\n❌ Failed to install dependencies. Please check the error messages above.")
        sys.exit(1)
    
    # Check system dependencies
    check_system_dependencies()
    
    # Test installation
    if not test_installation():
        print("\n❌ Installation test failed. Please check the error messages above.")
        sys.exit(1)
    
    print("\n🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("  1. Copy your PDF documents to data/raw/")
    print("  2. Run: python -m src.main detect data/raw/your_document.pdf")
    print("  3. Run: python -m src.main check data/raw/your_document.pdf")
    print("  4. Check the README.md for more information")
    
    print("\n📚 For training custom models:")
    print("  1. Prepare training data in data/labels/")
    print("  2. Use LabelImg or Roboflow for annotation")
    print("  3. Train with: python -m src.detection.trainer")


if __name__ == "__main__":
    main()
