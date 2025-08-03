"""
Basic unit tests for the signature detection system.
"""

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

# Add src to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.utils.config import ConfigManager
from src.preprocessing.pdf_processor import PDFProcessor
from src.ocr.text_extractor import TextExtractor
from src.compliance.compliance_checker import ComplianceChecker


class TestConfigManager:
    """Test configuration management."""
    
    def test_config_loading(self):
        """Test that configuration can be loaded."""
        config = ConfigManager()
        assert config.config is not None
        assert 'model' in config.config
        assert 'ocr' in config.config
    
    def test_config_get(self):
        """Test configuration value retrieval."""
        config = ConfigManager()
        model_size = config.get('model.yolo.model_size')
        assert model_size is not None
    
    def test_config_update(self):
        """Test configuration value updates."""
        config = ConfigManager()
        original_value = config.get('model.yolo.model_size')
        
        # Update value
        config.update('model.yolo.model_size', 'yolov8s')
        new_value = config.get('model.yolo.model_size')
        
        assert new_value == 'yolov8s'
        assert new_value != original_value


class TestPDFProcessor:
    """Test PDF processing functionality."""
    
    def test_processor_initialization(self):
        """Test PDF processor initialization."""
        processor = PDFProcessor()
        assert processor.dpi == 300
        assert processor.format == 'PNG'
    
    def test_signature_candidates(self):
        """Test signature candidate detection."""
        processor = PDFProcessor()
        
        # Create a simple test image
        test_image = np.ones((100, 100), dtype=np.uint8) * 255
        
        # Add some edges to simulate a signature
        test_image[40:60, 30:70] = 0
        
        candidates = processor.get_signature_candidates(test_image)
        assert isinstance(candidates, list)
    
    def test_image_preprocessing(self):
        """Test image preprocessing functions."""
        processor = PDFProcessor()
        
        # Create test image
        test_image = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        
        # Test preprocessing
        processed = processor._preprocess_image(test_image)
        assert processed.shape == test_image.shape
        assert processed.dtype == test_image.dtype


class TestTextExtractor:
    """Test OCR text extraction."""
    
    def test_extractor_initialization(self):
        """Test text extractor initialization."""
        extractor = TextExtractor()
        assert extractor.engine in ['paddleocr', 'tesseract', 'easyocr']
    
    def test_distance_calculation(self):
        """Test distance calculation between bounding boxes."""
        extractor = TextExtractor()
        
        bbox1 = [0, 0, 10, 10]
        bbox2 = [20, 20, 30, 30]
        
        distance = extractor._calculate_distance(bbox1, bbox2)
        assert distance > 0
        assert isinstance(distance, float)
    
    def test_role_mapping(self):
        """Test role text mapping."""
        extractor = TextExtractor()
        
        # Test role detection
        text_results = [
            {
                'text': 'Contracting Officer',
                'bbox': [10, 10, 100, 30],
                'confidence': 0.8
            }
        ]
        
        signature_bbox = [50, 50, 80, 80]
        role_candidates = extractor.find_role_text(text_results, signature_bbox)
        
        assert isinstance(role_candidates, list)


class TestComplianceChecker:
    """Test compliance checking functionality."""
    
    def test_checker_initialization(self):
        """Test compliance checker initialization."""
        checker = ComplianceChecker()
        assert checker.document_types is not None
        assert 'pre_solicitation' in checker.document_types
    
    def test_role_mapping(self):
        """Test role mapping functionality."""
        checker = ComplianceChecker()
        
        # Test standard role mapping
        mapped_role = checker._map_role_to_standard('Contracting Officer')
        assert mapped_role == 'contracting_officer'
        
        # Test variation mapping
        mapped_role = checker._map_role_to_standard('CO')
        assert mapped_role == 'contracting_officer'
    
    def test_compliance_analysis(self):
        """Test signature analysis."""
        checker = ComplianceChecker()
        
        # Mock detection results
        mock_detections = [
            {
                'signature_type': 'handwritten',
                'role_candidates': [
                    {
                        'text': 'Contracting Officer',
                        'confidence': 0.8,
                        'distance': 50,
                        'role_keywords': ['contracting officer']
                    }
                ],
                'page_number': 1
            }
        ]
        
        analysis = checker._analyze_signatures(mock_detections)
        assert 'signature_roles' in analysis
        assert 'signature_types' in analysis
        assert len(analysis['signature_roles']) > 0


class TestIntegration:
    """Integration tests."""
    
    def test_config_to_processor_flow(self):
        """Test configuration flows to processor correctly."""
        config = ConfigManager()
        processor = PDFProcessor()
        
        # Check that processor uses config values
        config_dpi = config.get('pdf.dpi')
        assert processor.dpi == config_dpi
    
    def test_processor_to_extractor_flow(self):
        """Test processor and extractor work together."""
        processor = PDFProcessor()
        extractor = TextExtractor()
        
        # Create test image
        test_image = np.random.randint(0, 255, (200, 200), dtype=np.uint8)
        
        # Get candidates
        candidates = processor.get_signature_candidates(test_image)
        
        if candidates:
            # Test with first candidate
            candidate = candidates[0]
            bbox = candidate['bbox']
            
            # Extract context
            context = extractor.extract_signature_context(test_image, bbox)
            assert isinstance(context, dict)
            assert 'role_candidates' in context


if __name__ == "__main__":
    pytest.main([__file__]) 