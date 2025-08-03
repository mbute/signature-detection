"""
Main signature detection module integrating YOLO detection with OCR and classification.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from loguru import logger

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logger.warning("Ultralytics not available. Install with: pip install ultralytics")

from ..utils.config import get_config
from ..preprocessing.pdf_processor import PDFProcessor
from ..ocr.text_extractor import TextExtractor


class SignatureDetector:
    """Main signature detection class that integrates all components."""
    
    def __init__(self, model_path: Optional[str] = None):
        """
        Initialize the signature detector.
        
        Args:
            model_path: Path to the trained YOLO model weights
        """
        self.config = get_config().get_model_config()
        self.yolo_config = self.config.get('yolo', {})
        
        # Initialize components
        self.pdf_processor = PDFProcessor()
        self.text_extractor = TextExtractor()
        self.model = self._load_model(model_path)
        
        # Detection parameters
        self.confidence_threshold = self.yolo_config.get('confidence_threshold', 0.5)
        self.iou_threshold = self.yolo_config.get('iou_threshold', 0.45)
        self.max_detections = self.yolo_config.get('max_detections', 100)
        self.signature_types = self.config.get('signature_types', [])
        
        logger.info("SignatureDetector initialized")
    
    def _load_model(self, model_path: Optional[str] = None) -> Optional[Any]:
        """
        Load the YOLO detection model.
        
        Args:
            model_path: Path to model weights
            
        Returns:
            Loaded YOLO model or None if not available
        """
        if not YOLO_AVAILABLE:
            logger.warning("YOLO not available - detection will use rule-based methods")
            return None
        
        try:
            if model_path and Path(model_path).exists():
                model = YOLO(model_path)
                logger.info(f"Loaded custom model from {model_path}")
            else:
                # Use default YOLO model for now
                model_size = self.yolo_config.get('model_size', 'yolov8n')
                model = YOLO(f"{model_size}.pt")
                logger.info(f"Loaded default model: {model_size}")
            
            return model
            
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")
            return None
    
    def detect_signatures(self, pdf_path: str) -> Dict[str, Any]:
        """
        Detect signatures in a PDF document.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            Dictionary containing detection results
        """
        logger.info(f"Processing PDF: {pdf_path}")
        
        try:
            # Convert PDF to images
            images = self.pdf_processor.convert_pdf_to_images(pdf_path)
            
            # Extract page information
            page_info = self.pdf_processor.extract_page_info(pdf_path)
            
            # Process each page
            all_detections = []
            for page_num, image in enumerate(images):
                page_detections = self._detect_signatures_in_image(
                    image, page_num, page_info[page_num] if page_num < len(page_info) else {}
                )
                all_detections.extend(page_detections)
            
            # Compile results
            results = {
                'pdf_path': pdf_path,
                'total_pages': len(images),
                'total_signatures': len(all_detections),
                'detections': all_detections,
                'page_info': page_info,
                'processing_time': None  # TODO: Add timing
            }
            
            logger.info(f"Detection complete: {len(all_detections)} signatures found")
            return results
            
        except Exception as e:
            logger.error(f"Signature detection failed: {e}")
            raise
    
    def _detect_signatures_in_image(self, image: np.ndarray, page_num: int, 
                                  page_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Detect signatures in a single image.
        
        Args:
            image: Input image
            page_num: Page number
            page_info: Page metadata
            
        Returns:
            List of signature detections
        """
        detections = []
        
        if self.model is not None:
            # Use YOLO model for detection
            yolo_detections = self._detect_with_yolo(image)
            detections.extend(yolo_detections)
        else:
            # Fall back to rule-based detection
            rule_detections = self._detect_with_rules(image)
            detections.extend(rule_detections)
        
        # Process each detection
        processed_detections = []
        for detection in detections:
            processed = self._process_detection(detection, image, page_num, page_info)
            if processed:
                processed_detections.append(processed)
        
        return processed_detections
    
    def _detect_with_yolo(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect signatures using YOLO model.
        
        Args:
            image: Input image
            
        Returns:
            List of YOLO detections
        """
        try:
            results = self.model(
                image,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                max_det=self.max_detections
            )
            
            detections = []
            for result in results:
                if result.boxes is not None:
                    for box in result.boxes:
                        # Get bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        
                        # Get confidence and class
                        confidence = float(box.conf[0])
                        class_id = int(box.cls[0])
                        class_name = self.signature_types[class_id] if class_id < len(self.signature_types) else 'unknown'
                        
                        detections.append({
                            'bbox': [float(x1), float(y1), float(x2), float(y2)],
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': class_name,
                            'detection_method': 'yolo'
                        })
            
            return detections
            
        except Exception as e:
            logger.error(f"YOLO detection failed: {e}")
            return []
    
    def _detect_with_rules(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect signatures using rule-based methods.
        
        Args:
            image: Input image
            
        Returns:
            List of rule-based detections
        """
        # Use the PDF processor's signature candidate detection
        candidates = self.pdf_processor.get_signature_candidates(image)
        
        detections = []
        for candidate in candidates:
            detections.append({
                'bbox': candidate['bbox'],
                'confidence': candidate['confidence'],
                'class_id': 0,  # Default to first class
                'class_name': 'handwritten',
                'detection_method': 'rule_based',
                'area': candidate['area'],
                'aspect_ratio': candidate['aspect_ratio']
            })
        
        return detections
    
    def _process_detection(self, detection: Dict[str, Any], image: np.ndarray, 
                          page_num: int, page_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process a single detection with OCR and classification.
        
        Args:
            detection: Raw detection result
            image: Input image
            page_num: Page number
            page_info: Page metadata
            
        Returns:
            Processed detection with additional information
        """
        try:
            bbox = detection['bbox']
            
            # Extract signature region
            signature_region = self._extract_signature_region(image, bbox)
            
            # Extract context using OCR
            context = self.text_extractor.extract_signature_context(image, bbox)
            
            # Classify signature type
            classification = self.text_extractor.classify_signature_type(signature_region, context)
            
            # Update detection with additional information
            processed_detection = {
                **detection,
                'page_number': page_num + 1,
                'page_info': page_info,
                'signature_type': classification['signature_type'],
                'signature_confidence': classification['confidence'],
                'classification_features': classification['features'],
                'context': context,
                'role_candidates': context['role_candidates'],
                'nearby_text': context['nearby_text']
            }
            
            return processed_detection
            
        except Exception as e:
            logger.error(f"Failed to process detection: {e}")
            return None
    
    def _extract_signature_region(self, image: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Extract the signature region from an image.
        
        Args:
            image: Input image
            bbox: Bounding box [x1, y1, x2, y2]
            
        Returns:
            Extracted signature region
        """
        x1, y1, x2, y2 = map(int, bbox)
        
        # Ensure coordinates are within image bounds
        height, width = image.shape[:2]
        x1 = max(0, min(x1, width))
        y1 = max(0, min(y1, height))
        x2 = max(0, min(x2, width))
        y2 = max(0, min(y2, height))
        
        return image[y1:y2, x1:x2]
    
    def detect_signatures_from_images(self, image_paths: List[str]) -> Dict[str, Any]:
        """
        Detect signatures from a list of image files.
        
        Args:
            image_paths: List of image file paths
            
        Returns:
            Dictionary containing detection results
        """
        logger.info(f"Processing {len(image_paths)} images")
        
        all_detections = []
        for i, image_path in enumerate(image_paths):
            try:
                # Load image
                image = cv2.imread(image_path)
                if image is None:
                    logger.warning(f"Failed to load image: {image_path}")
                    continue
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Detect signatures
                page_detections = self._detect_signatures_in_image(
                    image, i, {'image_path': image_path}
                )
                all_detections.extend(page_detections)
                
            except Exception as e:
                logger.error(f"Failed to process image {image_path}: {e}")
        
        results = {
            'total_images': len(image_paths),
            'total_signatures': len(all_detections),
            'detections': all_detections
        }
        
        logger.info(f"Image detection complete: {len(all_detections)} signatures found")
        return results
    
    def get_detection_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate a summary of detection results.
        
        Args:
            results: Detection results from detect_signatures
            
        Returns:
            Summary statistics
        """
        detections = results.get('detections', [])
        
        # Count by signature type
        type_counts = {}
        for detection in detections:
            sig_type = detection.get('signature_type', 'unknown')
            type_counts[sig_type] = type_counts.get(sig_type, 0) + 1
        
        # Count by detection method
        method_counts = {}
        for detection in detections:
            method = detection.get('detection_method', 'unknown')
            method_counts[method] = method_counts.get(method, 0) + 1
        
        # Average confidence scores
        confidences = [d.get('confidence', 0) for d in detections]
        avg_confidence = np.mean(confidences) if confidences else 0
        
        # Role analysis
        roles_found = set()
        for detection in detections:
            role_candidates = detection.get('role_candidates', [])
            for candidate in role_candidates:
                roles_found.update(candidate.get('role_keywords', []))
        
        summary = {
            'total_signatures': len(detections),
            'total_pages': results.get('total_pages', 0),
            'signature_types': type_counts,
            'detection_methods': method_counts,
            'average_confidence': avg_confidence,
            'roles_found': list(roles_found),
            'pages_with_signatures': len(set(d.get('page_number', 0) for d in detections))
        }
        
        return summary 