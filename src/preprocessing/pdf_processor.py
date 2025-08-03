"""
PDF processing utilities for signature detection.
"""

import fitz  # PyMuPDF
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from PIL import Image, ImageEnhance
import pdf2image
from loguru import logger

from ..utils.config import get_config


class PDFProcessor:
    """Handles PDF processing and image conversion for signature detection."""
    
    def __init__(self):
        """Initialize the PDF processor with configuration."""
        self.config = get_config().get_pdf_config()
        self.dpi = self.config.get('dpi', 300)
        self.format = self.config.get('format', 'PNG')
        self.grayscale = self.config.get('grayscale', False)
        self.min_page_size = self.config.get('min_page_size', 1000)
        self.max_pages = self.config.get('max_pages', 50)
        
        # Preprocessing settings
        self.preprocessing_config = self.config.get('preprocessing', {})
        self.deskew = False  # Disabled to avoid errors
        self.denoise = False  # Disabled to avoid errors
        self.enhance_contrast = False  # Disabled to avoid errors
        self.fix_orientation = True  # Keep orientation fixing enabled
    
    def convert_pdf_to_images(self, pdf_path: str) -> List[np.ndarray]:
        """
        Convert PDF pages to images.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of images as numpy arrays
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        logger.info(f"Converting PDF to images: {pdf_path}")
        
        try:
            # Use pdf2image for conversion
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt=self.format.lower(),
                grayscale=self.grayscale
            )
            
            # Convert PIL images to numpy arrays
            numpy_images = []
            for i, img in enumerate(images):
                if i >= self.max_pages:
                    logger.warning(f"Reached maximum page limit ({self.max_pages})")
                    break
                
                # Convert to numpy array
                img_array = np.array(img)
                
                # Skip small pages
                if min(img_array.shape[:2]) < self.min_page_size:
                    logger.warning(f"Skipping page {i+1} - too small: {img_array.shape}")
                    continue
                
                # Apply preprocessing
                processed_img = self._preprocess_image(img_array)
                numpy_images.append(processed_img)
                
                logger.debug(f"Processed page {i+1}: {img_array.shape} -> {processed_img.shape}")
            
            logger.info(f"Successfully converted {len(numpy_images)} pages")
            return numpy_images
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            raise
    
    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply preprocessing to an image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        processed = image.copy()
        
        # Fix orientation if enabled
        if self.fix_orientation:
            processed = self._fix_orientation(processed)
        
        # Deskew if enabled
        if self.deskew:
            processed = self._deskew_image(processed)
        
        # Denoise if enabled
        if self.denoise:
            processed = self._denoise_image(processed)
        
        # Enhance contrast if enabled
        if self.enhance_contrast:
            processed = self._enhance_contrast(processed)
        
        return processed
    
    def _deskew_image(self, image: np.ndarray) -> np.ndarray:
        """
        Deskew an image by detecting and correcting rotation.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Deskewed image
        """
        # Convert to binary image
        _, binary = cv2.threshold(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find the largest contour (assumed to be the main content)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Fit a rectangle to the contour
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        
        # Normalize angle
        if angle < -45:
            angle = 90 + angle
        
        # Rotate if angle is significant
        if abs(angle) > 0.5:
            height, width = image.shape[:2]
            center = (width // 2, height // 2)
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                   flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image
    
    def _fix_orientation(self, image: np.ndarray) -> np.ndarray:
        """
        Fix image orientation by detecting if it's rotated 90 degrees.
        Maintains portrait orientation for documents.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Correctly oriented image in portrait format
        """
        height, width = image.shape[:2]
        
        # If image is wider than it is tall, it's likely rotated from portrait
        # We want to maintain portrait orientation for document processing
        if width > height * 1.2:  # If significantly wider than tall
            # Rotate 90 degrees clockwise to make it portrait
            rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
            logger.debug(f"Fixed orientation: rotated 90° clockwise to portrait (width: {width}, height: {height})")
            return rotated
        
        return image
    
    def _denoise_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply denoising to an image.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Denoised image
        """
        # Apply bilateral filter for edge-preserving denoising
        denoised = cv2.bilateralFilter(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image, 9, 75, 75)
        return denoised
    
    def _enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance image contrast using CLAHE.
        
        Args:
            image: Input grayscale image
            
        Returns:
            Contrast-enhanced image
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if len(image.shape) == 3 else image)
        return enhanced
    
    def extract_page_info(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Extract metadata from PDF pages.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of page metadata dictionaries
        """
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        try:
            doc = fitz.open(pdf_path)
            page_info = []
            
            for page_num in range(min(len(doc), self.max_pages)):
                page = doc[page_num]
                
                # Get page dimensions
                rect = page.rect
                width, height = rect.width, rect.height
                
                # Get page rotation
                rotation = page.rotation
                
                # Check if page has text
                text = page.get_text()
                has_text = len(text.strip()) > 0
                
                page_info.append({
                    'page_number': page_num + 1,
                    'width': width,
                    'height': height,
                    'rotation': rotation,
                    'has_text': has_text,
                    'text_length': len(text)
                })
            
            doc.close()
            return page_info
            
        except Exception as e:
            logger.error(f"Failed to extract page info: {e}")
            raise
    
    def get_signature_candidates(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify potential signature regions in an image.
        
        Args:
            image: Input image
            
        Returns:
            List of candidate regions with bounding boxes
        """
        candidates = []
        
