"""
Main entry point for the Signature Detection & Compliance Checker.
"""

import click
import json
from pathlib import Path
from typing import Optional
from loguru import logger

from .utils.config import get_config, init_logging
from .detection.signature_detector import SignatureDetector
from .compliance.compliance_checker import ComplianceChecker


@click.group()
@click.option('--config', '-c', help='Path to configuration file')
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def cli(config: Optional[str], verbose: bool):
    """Signature Detection & Compliance Checker CLI."""
    # Initialize logging
    init_logging()
    
    if verbose:
        logger.info("Verbose logging enabled")
    
    # Load configuration
    if config:
        get_config(config)
    else:
        get_config()


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output directory for results')
@click.option('--model', '-m', help='Path to custom YOLO model')
@click.option('--save-images', is_flag=True, help='Save processed images')
def detect(pdf_path: str, output: Optional[str], model: Optional[str], save_images: bool):
    """Detect signatures in a PDF document."""
    try:
        # Initialize detector
        detector = SignatureDetector(model_path=model)
        
        # Detect signatures
        results = detector.detect_signatures(pdf_path)
        
        # Generate summary
        summary = detector.get_detection_summary(results)
        
        # Print results
        click.echo(f"\n📄 Document: {pdf_path}")
        click.echo(f"📊 Total signatures detected: {summary['total_signatures']}")
        click.echo(f"📋 Signature types: {summary['signature_types']}")
        click.echo(f"🎯 Average confidence: {summary['average_confidence']:.2f}")
        click.echo(f"👥 Roles found: {summary['roles_found']}")
        
        # Save results if output specified
        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save JSON results
            results_file = output_path / "detection_results.json"
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            # Save summary
            summary_file = output_path / "summary.json"
            with open(summary_file, 'w') as f:
                json.dump(summary, f, indent=2)
            
            click.echo(f"\n💾 Results saved to: {output_path}")
            
            # Save processed images if requested
            if save_images:
                images_dir = output_path / "processed_images"
                detector.pdf_processor.save_processed_images(
                    detector.pdf_processor.convert_pdf_to_images(pdf_path),
                    str(images_dir),
                    Path(pdf_path).stem
                )
                click.echo(f"📸 Processed images saved to: {images_dir}")
        
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.argument('pdf_path', type=click.Path(exists=True))
@click.option('--document-type', '-t', help='Document type for compliance checking')
@click.option('--output', '-o', help='Output directory for results')
@click.option('--model', '-m', help='Path to custom YOLO model')
def check(pdf_path: str, document_type: Optional[str], output: Optional[str], model: Optional[str]):
    """Check signature compliance for a PDF document."""
    try:
        # Initialize components
        detector = SignatureDetector(model_path=model)
        checker = ComplianceChecker()
        
        # Detect signatures
        click.echo("🔍 Detecting signatures...")
        detection_results = detector.detect_signatures(pdf_path)
        
        # Check compliance
        click.echo("✅ Checking compliance...")
        compliance_report = checker.generate_compliance_report(detection_results, document_type)
        
        # Print results
        summary = compliance_report['summary']
        click.echo(f"\n📄 Document: {pdf_path}")
        click.echo(f"📋 Document type: {summary['document_type']}")
        click.echo(f"✅ Compliant: {'Yes' if summary['is_compliant'] else 'No'}")
        click.echo(f"📊 Total signatures: {summary['total_signatures']}")
        click.echo(f"🎯 Quality score: {summary['quality_score']:.2f}")
        
        if not summary['is_compliant']:
            click.echo(f"❌ Missing signatures: {compliance_report['compliance']['missing_signatures']}")
        
        if summary['quality_issues_count'] > 0:
            click.echo(f"⚠️  Quality issues: {summary['quality_issues_count']}")
        
        # Print recommendations
        click.echo("\n💡 Recommendations:")
        for rec in compliance_report['recommendations']:
            click.echo(f"  • {rec}")
        
        # Save results if output specified
        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Save full report
            report_file = output_path / "compliance_report.json"
            with open(report_file, 'w') as f:
                json.dump(compliance_report, f, indent=2, default=str)
            
            click.echo(f"\n💾 Report saved to: {report_file}")
        
    except Exception as e:
        logger.error(f"Compliance check failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
@click.argument('input_dir', type=click.Path(exists=True))
@click.option('--output', '-o', help='Output directory for results')
@click.option('--model', '-m', help='Path to custom YOLO model')
def batch(input_dir: str, output: Optional[str], model: Optional[str]):
    """Process multiple PDF files in batch."""
    try:
        input_path = Path(input_dir)
        pdf_files = list(input_path.glob("*.pdf"))
        
        if not pdf_files:
            click.echo(f"No PDF files found in {input_dir}")
            return
        
        click.echo(f"📁 Processing {len(pdf_files)} PDF files...")
        
        # Initialize components
        detector = SignatureDetector(model_path=model)
        checker = ComplianceChecker()
        
        # Process each file
        all_results = []
        for i, pdf_file in enumerate(pdf_files, 1):
            click.echo(f"\n[{i}/{len(pdf_files)}] Processing {pdf_file.name}...")
            
            try:
                # Detect signatures
                detection_results = detector.detect_signatures(str(pdf_file))
                
                # Check compliance
                compliance_report = checker.generate_compliance_report(detection_results)
                
                # Store results
                file_result = {
                    'file': pdf_file.name,
                    'detection': detection_results,
                    'compliance': compliance_report
                }
                all_results.append(file_result)
                
                # Print summary
                summary = compliance_report['summary']
                status = "✅" if summary['is_compliant'] else "❌"
                click.echo(f"  {status} {summary['total_signatures']} signatures, "
                          f"compliant: {summary['is_compliant']}")
                
            except Exception as e:
                logger.error(f"Failed to process {pdf_file.name}: {e}")
                click.echo(f"  ❌ Error: {e}")
        
        # Generate batch summary
        total_files = len(all_results)
        compliant_files = sum(1 for r in all_results if r['compliance']['summary']['is_compliant'])
        total_signatures = sum(r['compliance']['summary']['total_signatures'] for r in all_results)
        
        click.echo(f"\n📊 Batch Summary:")
        click.echo(f"  Total files: {total_files}")
        click.echo(f"  Compliant files: {compliant_files}")
        click.echo(f"  Total signatures: {total_signatures}")
        click.echo(f"  Compliance rate: {compliant_files/total_files*100:.1f}%")
        
        # Save batch results
        if output:
            output_path = Path(output)
            output_path.mkdir(parents=True, exist_ok=True)
            
            batch_file = output_path / "batch_results.json"
            with open(batch_file, 'w') as f:
                json.dump(all_results, f, indent=2, default=str)
            
            click.echo(f"\n💾 Batch results saved to: {batch_file}")
        
    except Exception as e:
        logger.error(f"Batch processing failed: {e}")
        click.echo(f"❌ Error: {e}", err=True)


@cli.command()
def info():
    """Show system information and configuration."""
    try:
        config = get_config()
        
        click.echo("🔧 System Information:")
        click.echo(f"  Configuration file: {config.config_path}")
        click.echo(f"  YOLO model size: {config.get('model.yolo.model_size')}")
        click.echo(f"  OCR engine: {config.get('ocr.engine')}")
        click.echo(f"  PDF DPI: {config.get('pdf.dpi')}")
        
        # Check available components
        click.echo("\n📦 Available Components:")
        
        try:
            from ultralytics import YOLO
            click.echo("  ✅ YOLO (Ultralytics)")
        except ImportError:
            click.echo("  ❌ YOLO (Ultralytics) - not installed")
        
        try:
            from paddleocr import PaddleOCR
            click.echo("  ✅ PaddleOCR")
        except ImportError:
            click.echo("  ❌ PaddleOCR - not installed")
        
        try:
            import pytesseract
            click.echo("  ✅ Tesseract")
        except ImportError:
            click.echo("  ❌ Tesseract - not installed")
        
        try:
            import fitz
            click.echo("  ✅ PyMuPDF")
        except ImportError:
            click.echo("  ❌ PyMuPDF - not installed")
        
    except Exception as e:
        logger.error(f"Failed to get system info: {e}")
        click.echo(f"❌ Error: {e}", err=True)


if __name__ == '__main__':
    cli() 