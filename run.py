#!/usr/bin/env python3
"""
Entry point for Signature Detection & Compliance Checker.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

if __name__ == "__main__":
    from src.main import cli
    cli() 