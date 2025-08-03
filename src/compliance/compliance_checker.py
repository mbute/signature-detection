"""
Compliance checking module for signature validation.
"""

import re
from typing import List, Dict, Any, Optional, Set
from loguru import logger

from ..utils.config import get_config


class ComplianceChecker:
    """Validates signature compliance against document requirements."""
    
    def __init__(self):
        """Initialize the compliance checker with configuration."""
        self.config = get_config().get_compliance_config()
        self.document_types = self.config.get('document_types', {})
        
        # Role mapping for flexible matching
        self.role_mapping = {
            'contracting_officer': ['contracting officer', 'co', 'contracting official'],
            'contract_specialist': ['contract specialist', 'cs', 'procurement specialist'],
            'program_manager': ['program manager', 'pm', 'project manager'],
            'technical_representative': ['technical representative', 'tech rep', 'technical rep'],
            'legal_counsel': ['legal counsel', 'legal', 'attorney', 'lawyer'],
            'source_selection_authority': ['source selection authority', 'ssa', 'selection authority']
        }
        
        logger.info("ComplianceChecker initialized")
    
    def check_compliance(self, detection_results: Dict[str, Any], 
                        document_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Check compliance of detected signatures against document requirements.
        
        Args:
            detection_results: Results from signature detection
            document_type: Type of document being analyzed
            
        Returns:
            Compliance report with validation results
        """
        logger.info(f"Checking compliance for document type: {document_type}")
        
        # Extract detections
        detections = detection_results.get('detections', [])
        
        # Determine document type if not provided
        if document_type is None:
            document_type = self._infer_document_type(detections)
        
        # Get requirements for this document type
        requirements = self.document_types.get(document_type, {})
        required_signatures = requirements.get('required_signatures', [])
        optional_signatures = requirements.get('optional_signatures', [])
        
        # Analyze signatures
        signature_analysis = self._analyze_signatures(detections)
        
        # Check compliance
        compliance_results = self._check_signature_compliance(
            signature_analysis, required_signatures, optional_signatures
        )
        
        # Generate report
        report = {
            'document_type': document_type,
            'total_signatures': len(detections),
            'required_signatures': required_signatures,
            'optional_signatures': optional_signatures,
            'signature_analysis': signature_analysis,
            'compliance_results': compliance_results,
            'is_compliant': compliance_results['is_compliant'],
            'missing_signatures': compliance_results['missing_signatures'],
            'unexpected_signatures': compliance_results['unexpected_signatures'],
            'warnings': compliance_results['warnings']
        }
        
        logger.info(f"Compliance check complete. Compliant: {report['is_compliant']}")
        return report
    
    def _infer_document_type(self, detections: List[Dict[str, Any]]) -> str:
        """
        Infer document type based on detected signatures and context.
        
        Args:
            detections: List of signature detections
            
        Returns:
            Inferred document type
        """
        # This is a simple heuristic - in practice, you might use more sophisticated methods
        # like document classification or keyword analysis
        
        # Count different role types
        role_counts = {}
        for detection in detections:
            role_candidates = detection.get('role_candidates', [])
            for candidate in role_candidates:
                for keyword in candidate.get('role_keywords', []):
                    role_counts[keyword] = role_counts.get(keyword, 0) + 1
        
        # Simple rules for document type inference
        if 'contracting officer' in role_counts and 'program manager' in role_counts:
            return 'pre_solicitation'
        elif 'contracting officer' in role_counts and 'contract specialist' in role_counts:
            return 'solicitation'
        elif 'contracting officer' in role_counts and 'source selection authority' in role_counts:
            return 'award_decision'
        elif 'contracting officer' in role_counts:
            return 'contract_modification'
        else:
            return 'unknown'
    
    def _analyze_signatures(self, detections: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze detected signatures to extract roles and types.
        
        Args:
            detections: List of signature detections
            
        Returns:
            Analysis of signatures with roles and types
        """
        signature_roles = []
        signature_types = {}
        role_confidence = {}
        
        for detection in detections:
            # Get signature type
            sig_type = detection.get('signature_type', 'unknown')
            signature_types[sig_type] = signature_types.get(sig_type, 0) + 1
            
            # Analyze role candidates
            role_candidates = detection.get('role_candidates', [])
            if role_candidates:
                # Take the best candidate (closest and highest confidence)
                best_candidate = role_candidates[0]
                role_text = best_candidate.get('text', '')
                confidence = best_candidate.get('confidence', 0)
                
                # Map to standard role
                mapped_role = self._map_role_to_standard(role_text)
                
                if mapped_role:
                    signature_roles.append({
                        'detected_role': role_text,
                        'mapped_role': mapped_role,
                        'confidence': confidence,
                        'distance': best_candidate.get('distance', 0),
                        'page_number': detection.get('page_number', 0),
                        'signature_type': sig_type
                    })
                    
                    # Track confidence for each role
                    if mapped_role not in role_confidence:
                        role_confidence[mapped_role] = []
                    role_confidence[mapped_role].append(confidence)
        
        return {
            'signature_roles': signature_roles,
            'signature_types': signature_types,
            'role_confidence': role_confidence,
            'unique_roles': list(set(role['mapped_role'] for role in signature_roles))
        }
    
    def _map_role_to_standard(self, role_text: str) -> Optional[str]:
        """
        Map detected role text to standard role names.
        
        Args:
            role_text: Detected role text
            
        Returns:
            Standard role name or None if no match
        """
        role_text_lower = role_text.lower()
        
        for standard_role, variations in self.role_mapping.items():
            for variation in variations:
                if variation.lower() in role_text_lower:
                    return standard_role
        
        return None
    
    def _check_signature_compliance(self, signature_analysis: Dict[str, Any],
                                  required_signatures: List[str],
                                  optional_signatures: List[str]) -> Dict[str, Any]:
        """
        Check if signatures meet compliance requirements.
        
        Args:
            signature_analysis: Analysis of detected signatures
            required_signatures: List of required signature roles
            optional_signatures: List of optional signature roles
            
        Returns:
            Compliance check results
        """
        detected_roles = set(signature_analysis['unique_roles'])
        required_set = set(required_signatures)
        optional_set = set(optional_signatures)
        
        # Find missing required signatures
        missing_signatures = required_set - detected_roles
        
        # Find unexpected signatures (not in required or optional)
        all_expected = required_set | optional_set
        unexpected_signatures = detected_roles - all_expected
        
        # Check if all required signatures are present
        is_compliant = len(missing_signatures) == 0
        
        # Generate warnings
        warnings = []
        
        if not is_compliant:
            warnings.append(f"Missing required signatures: {list(missing_signatures)}")
        
        if unexpected_signatures:
            warnings.append(f"Unexpected signatures found: {list(unexpected_signatures)}")
        
        # Check for duplicate roles
        role_counts = {}
        for role_info in signature_analysis['signature_roles']:
            role = role_info['mapped_role']
            role_counts[role] = role_counts.get(role, 0) + 1
        
        duplicate_roles = [role for role, count in role_counts.items() if count > 1]
        if duplicate_roles:
            warnings.append(f"Duplicate signatures found for roles: {duplicate_roles}")
        
        # Check signature type distribution
        signature_types = signature_analysis['signature_types']
        if 'blank' in signature_types and signature_types['blank'] > 0:
            warnings.append(f"Found {signature_types['blank']} blank signature blocks")
        
        return {
            'is_compliant': is_compliant,
            'missing_signatures': list(missing_signatures),
            'unexpected_signatures': list(unexpected_signatures),
            'detected_roles': list(detected_roles),
            'duplicate_roles': duplicate_roles,
            'warnings': warnings,
            'role_counts': role_counts
        }
    
    def validate_signature_quality(self, detection_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate the quality and reliability of signature detections.
        
        Args:
            detection_results: Results from signature detection
            
        Returns:
            Quality validation results
        """
        detections = detection_results.get('detections', [])
        
        quality_issues = []
        confidence_scores = []
        
        for detection in detections:
            # Check detection confidence
            confidence = detection.get('confidence', 0)
            confidence_scores.append(confidence)
            
            if confidence < 0.5:
                quality_issues.append({
                    'type': 'low_confidence',
                    'page': detection.get('page_number', 0),
                    'confidence': confidence,
                    'message': f"Low detection confidence: {confidence:.2f}"
                })
            
            # Check signature classification confidence
            sig_confidence = detection.get('signature_confidence', 0)
            if sig_confidence < 0.6:
                quality_issues.append({
                    'type': 'low_classification_confidence',
                    'page': detection.get('page_number', 0),
                    'confidence': sig_confidence,
                    'message': f"Low signature classification confidence: {sig_confidence:.2f}"
                })
            
            # Check for role detection issues
            role_candidates = detection.get('role_candidates', [])
            if not role_candidates:
                quality_issues.append({
                    'type': 'no_role_detected',
                    'page': detection.get('page_number', 0),
                    'message': "No role/title detected near signature"
                })
        
        avg_confidence = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0
        
        return {
            'total_detections': len(detections),
            'quality_issues': quality_issues,
            'average_confidence': avg_confidence,
            'issues_count': len(quality_issues),
            'quality_score': max(0, 1 - len(quality_issues) / max(1, len(detections)))
        }
    
    def generate_compliance_report(self, detection_results: Dict[str, Any],
                                 document_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate a comprehensive compliance report.
        
        Args:
            detection_results: Results from signature detection
            document_type: Type of document being analyzed
            
        Returns:
            Comprehensive compliance report
        """
        # Check compliance
        compliance_report = self.check_compliance(detection_results, document_type)
        
        # Validate quality
        quality_report = self.validate_signature_quality(detection_results)
        
        # Generate summary
        summary = {
            'document_type': compliance_report['document_type'],
            'total_signatures': compliance_report['total_signatures'],
            'is_compliant': compliance_report['is_compliant'],
            'quality_score': quality_report['quality_score'],
            'missing_signatures_count': len(compliance_report['missing_signatures']),
            'quality_issues_count': quality_report['issues_count']
        }
        
        # Compile full report
        full_report = {
            'summary': summary,
            'compliance': compliance_report,
            'quality': quality_report,
            'recommendations': self._generate_recommendations(compliance_report, quality_report)
        }
        
        return full_report
    
    def _generate_recommendations(self, compliance_report: Dict[str, Any],
                                quality_report: Dict[str, Any]) -> List[str]:
        """
        Generate recommendations based on compliance and quality analysis.
        
        Args:
            compliance_report: Compliance check results
            quality_report: Quality validation results
            
        Returns:
            List of recommendations
        """
        recommendations = []
        
        # Compliance recommendations
        if not compliance_report['is_compliant']:
            missing = compliance_report['missing_signatures']
            recommendations.append(f"Obtain signatures from: {', '.join(missing)}")
        
        if compliance_report['unexpected_signatures']:
            unexpected = compliance_report['unexpected_signatures']
            recommendations.append(f"Verify necessity of signatures from: {', '.join(unexpected)}")
        
        # Quality recommendations
        if quality_report['quality_score'] < 0.8:
            recommendations.append("Consider manual review due to low detection quality")
        
        if quality_report['issues_count'] > 0:
            recommendations.append("Review low-confidence detections manually")
        
        if not recommendations:
            recommendations.append("Document appears to be compliant and of good quality")
        
        return recommendations 