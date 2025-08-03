"""
OCR text extraction utilities for signature detection.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

try:
    from paddleocr import PaddleOCR
    PADDLEOCR_AVAILABLE = True
except ImportError:
    PADDLEOCR_AVAILABLE = False
    logger.warning("PaddleOCR not available. Install with: pip install paddleocr")

try:
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    logger.warning("Tesseract not available. Install with: pip install pytesseract")

try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False
    logger.warning("EasyOCR not available. Install with: pip install easyocr")

from ..utils.config import get_config


class TextExtractor:
    """Handles OCR text extraction and role detection for signature analysis."""
    
    def __init__(self, engine: Optional[str] = None):
        """
        Initialize the text extractor.
        
        Args:
            engine: OCR engine to use ('paddleocr', 'tesseract', 'easyocr')
        """
        self.config = get_config().get_ocr_config()
        self.engine = engine or self.config.get('engine', 'paddleocr')
        
        # Initialize OCR engines
        self._init_ocr_engines()
        
        # Text extraction settings
        self.text_config = self.config.get('text_extraction', {})
        self.min_confidence = self.text_config.get('min_confidence', 0.6)
        self.max_distance = self.text_config.get('max_distance', 100)
        self.role_keywords = self.text_config.get('role_keywords', [])
        
        logger.info(f"Initialized TextExtractor with engine: {self.engine}")
    
    def _init_ocr_engines(self):
        """Initialize OCR engines based on configuration."""
        self.paddleocr = None
        self.tesseract_config = None
        self.easyocr_reader = None
        
        if self.engine == 'paddleocr' and PADDLEOCR_AVAILABLE:
            paddle_config = self.config.get('paddleocr', {})
            logger.info("Initialized PaddleOCR")
        
        elif self.engine == 'tesseract' and TESSERACT_AVAILABLE:
            tesseract_config = self.config.get('tesseract', {})
            self.tesseract_config = tesseract_config.get('config', '--psm 6 --oem 3')
            logger.info("Initialized Tesseract")
        
        elif self.engine == 'easyocr' and EASYOCR_AVAILABLE:
            self.easyocr_reader = easyocr.Reader(['en'])
            logger.info("Initialized EasyOCR")
        
        else:
            available_engines = []
            if PADDLEOCR_AVAILABLE:
                available_engines.append('paddleocr')
            if TESSERACT_AVAILABLE:
                available_engines.append('tesseract')
            if EASYOCR_AVAILABLE:
                available_engines.append('easyocr')
            
            if available_engines:
                logger.warning(f"Requested engine '{self.engine}' not available. "
                             f"Available engines: {available_engines}")
                self.engine = available_engines[0]
                self._init_ocr_engines()
            else:
                raise RuntimeError("No OCR engines available. Please install at least one OCR library.")
    
    def extract_text(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extract text from an image using the configured OCR engine.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of text detection results with bounding boxes and confidence scores
        """
        if self.engine == 'paddleocr' and self.paddleocr:
            return self._extract_text_paddleocr(image)
        elif self.engine == 'tesseract' and TESSERACT_AVAILABLE:
            return self._extract_text_tesseract(image)
        elif self.engine == 'easyocr' and self.easyocr_reader:
            return self._extract_text_easyocr(image)
        else:
            raise RuntimeError(f"OCR engine '{self.engine}' not properly initialized")
    
    def _extract_text_paddleocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using PaddleOCR."""
        try:
            results = self.paddleocr.ocr(image, cls=True)
            text_results = []
            
            if results and results[0]:
                for line in results[0]:
                    if line and len(line) >= 2:
                        bbox, (text, confidence) = line
                        
                        if confidence >= self.min_confidence:
                            # Convert bbox to [x1, y1, x2, y2] format
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            bbox_normalized = [
                                min(x_coords), min(y_coords),
                                max(x_coords), max(y_coords)
                            ]
                            
                            text_results.append({
                                'text': text.strip(),
                                'bbox': bbox_normalized,
                                'confidence': confidence,
                                'bbox_original': bbox
                            })
            
            return text_results
            
        except Exception as e:
            logger.error(f"PaddleOCR extraction failed: {e}")
            return []
    
    def _extract_text_tesseract(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using Tesseract."""
        try:
            # Get detailed OCR data
            data = pytesseract.image_to_data(
                image, 
                config=self.tesseract_config,
                output_type=pytesseract.Output.DICT
            )
            
            text_results = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                confidence = float(data['conf'][i])
                
                if text and confidence >= self.min_confidence * 100:  # Tesseract uses 0-100 scale
                    x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                    bbox = [x, y, x + w, y + h]
                    
                    text_results.append({
                        'text': text,
                        'bbox': bbox,
                        'confidence': confidence / 100.0,  # Normalize to 0-1
                        'bbox_original': [[x, y], [x + w, y], [x + w, y + h], [x, y + h]]
                    })
            
            return text_results
            
        except Exception as e:
            logger.error(f"Tesseract extraction failed: {e}")
            return []
    
    def _extract_text_easyocr(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """Extract text using EasyOCR."""
        try:
            results = self.easyocr_reader.readtext(image)
            text_results = []
            
            for bbox, text, confidence in results:
                if confidence >= self.min_confidence:
                    # Convert bbox to [x1, y1, x2, y2] format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    bbox_normalized = [
                        min(x_coords), min(y_coords),
                        max(x_coords), max(y_coords)
                    ]
                    
                    text_results.append({
                        'text': text.strip(),
                        'bbox': bbox_normalized,
                        'confidence': confidence,
                        'bbox_original': bbox
                    })
            
            return text_results
            
        except Exception as e:
            logger.error(f"EasyOCR extraction failed: {e}")
            return []
    
    def find_role_text(self, text_results: List[Dict[str, Any]], 
                      signature_bbox: List[float]) -> List[Dict[str, Any]]:
        """
        Find text that likely represents roles/titles near a signature.
        
        Args:
            text_results: List of extracted text results
            signature_bbox: Signature bounding box [x1, y1, x2, y2]
            
        Returns:
            List of potential role text results
        """
        role_candidates = []
        
        for text_result in text_results:
            text = text_result['text'].lower()
            text_bbox = text_result['bbox']
            
            # Check if text contains role keywords
            is_role = any(keyword.lower() in text for keyword in self.role_keywords)
            
            if is_role:
                # Calculate distance between signature and text
                distance = self._calculate_distance(signature_bbox, text_bbox)
                
                if distance <= self.max_distance:
                    role_candidates.append({
                        **text_result,
                        'distance': distance,
                        'role_keywords': [kw for kw in self.role_keywords if kw.lower() in text]
                    })
        
        # Sort by distance (closest first)
        role_candidates.sort(key=lambda x: x['distance'])
        
        return role_candidates
    
    def _calculate_distance(self, bbox1: List[float], bbox2: List[float]) -> float:
        """
        Calculate the minimum distance between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            Minimum distance between the boxes
        """
        # Calculate center points
        center1_x = (bbox1[0] + bbox1[2]) / 2
        center1_y = (bbox1[1] + bbox1[3]) / 2
        center2_x = (bbox2[0] + bbox2[2]) / 2
        center2_y = (bbox2[1] + bbox2[3]) / 2
        
        # Calculate Euclidean distance
        distance = np.sqrt((center1_x - center2_x)**2 + (center1_y - center2_y)**2)
        
        return distance
    
    def extract_signature_context(self, image: np.ndarray, 
                                signature_bbox: List[float]) -> Dict[str, Any]:
        """
        Extract contextual information around a signature.
        
        Args:
            image: Input image
            signature_bbox: Signature bounding box [x1, y1, x2, y2]
            
        Returns:
            Dictionary containing signature context information
        """
        # Extract text from the entire image
        text_results = self.extract_text(image)
        
        # Find role text near the signature
        role_candidates = self.find_role_text(text_results, signature_bbox)
        
        # Extract text in the signature region
        signature_region = self._extract_region(image, signature_bbox)
        signature_text = self.extract_text(signature_region)
        
        # Find nearby text (not necessarily roles)
        nearby_text = []
        for text_result in text_results:
            distance = self._calculate_distance(signature_bbox, text_result['bbox'])
            if distance <= self.max_distance * 2:  # Larger search area
                nearby_text.append({
                    **text_result,
                    'distance': distance
                })
        
        # Sort by distance
        nearby_text.sort(key=lambda x: x['distance'])
        
        return {
            'role_candidates': role_candidates,
            'signature_text': signature_text,
            'nearby_text': nearby_text[:10],  # Top 10 closest
            'total_text_detections': len(text_results)
        }
    
    def _extract_region(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Extract a region from an image based on bounding box.
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Extracted region as numpy array
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        return image[y1:y2, x1:x2]
    
    def classify_signature_type(self, signature_region: np.ndarray, 
                              context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify the type of signature based on visual and textual features.
        
        Args:
            signature_region: Image region containing the signature
            context: Context information from extract_signature_context
            
        Returns:
            Classification result with confidence scores
        """
        # This is a placeholder implementation
        # In practice, you would use a trained classifier here
        
        # Simple heuristics based on text content
        signature_text = context.get('signature_text', [])
        has_text = len(signature_text) > 0
        
        # Check for digital signature indicators
        digital_indicators = ['digitally signed', 'electronic signature', 'esigned']
        is_digital = any(
            any(indicator in text_result['text'].lower() 
                for text_result in signature_text)
            for indicator in digital_indicators
        )
        
        # Check if region is mostly blank
        gray_region = cv2.cvtColor(signature_region, cv2.COLOR_RGB2GRAY) if len(signature_region.shape) == 3 else signature_region
        blank_ratio = np.sum(gray_region > 240) / gray_region.size  # Ratio of white pixels
        
        # Classification logic
        if is_digital:
            signature_type = 'digital'
            confidence = 0.8
        elif blank_ratio > 0.8:
            signature_type = 'blank'
            confidence = 0.9
        elif has_text:
            signature_type = 'handwritten'
            confidence = 0.7
        else:
            signature_type = 'handwritten'
            confidence = 0.6
        
        return {
            'signature_type': signature_type,
            'confidence': confidence,
            'features': {
                'has_text': has_text,
                'is_digital': is_digital,
                'blank_ratio': blank_ratio,
                'text_count': len(signature_text)
            }
        } 