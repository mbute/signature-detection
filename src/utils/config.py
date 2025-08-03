"""
Configuration management utilities for the signature detection system.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional
from loguru import logger


class ConfigManager:
    """Manages configuration loading and validation for the signature detection system."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the configuration file. If None, uses default location.
        """
        if config_path is None:
            config_path = self._get_default_config_path()
        
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _get_default_config_path(self) -> str:
        """Get the default configuration file path."""
        # Look for config.yaml in the config directory
        config_dir = Path(__file__).parent.parent.parent / "config"
        config_file = config_dir / "config.yaml"
        
        if not config_file.exists():
            # Fall back to example config
            example_config = config_dir / "config.example.yaml"
            if example_config.exists():
                logger.warning(f"Config file not found at {config_file}. Using example config.")
                return str(example_config)
            else:
                raise FileNotFoundError(f"No configuration file found in {config_dir}")
        
        return str(config_file)
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"Configuration loaded from {self.config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load configuration from {self.config_path}: {e}")
            raise
    
    def _validate_config(self):
        """Validate the loaded configuration."""
        required_sections = ['model', 'ocr', 'pdf', 'compliance', 'data', 'logging']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate model configuration
        if 'yolo' not in self.config['model']:
            raise ValueError("Missing YOLO configuration in model section")
        
        # Validate OCR configuration
        if 'engine' not in self.config['ocr']:
            raise ValueError("Missing OCR engine configuration")
        
        logger.info("Configuration validation passed")
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value using dot notation.
        
        Args:
            key: Configuration key in dot notation (e.g., 'model.yolo.model_size')
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        keys = key.split('.')
        value = self.config
        
        try:
            for k in keys:
                value = value[k]
            return value
        except (KeyError, TypeError):
            return default
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration section."""
        return self.config.get('model', {})
    
    def get_ocr_config(self) -> Dict[str, Any]:
        """Get OCR configuration section."""
        return self.config.get('ocr', {})
    
    def get_pdf_config(self) -> Dict[str, Any]:
        """Get PDF processing configuration section."""
        return self.config.get('pdf', {})
    
    def get_compliance_config(self) -> Dict[str, Any]:
        """Get compliance configuration section."""
        return self.config.get('compliance', {})
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data processing configuration section."""
        return self.config.get('data', {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration section."""
        return self.config.get('logging', {})
    
    def get_performance_config(self) -> Dict[str, Any]:
        """Get performance configuration section."""
        return self.config.get('performance', {})
    
    def get_output_config(self) -> Dict[str, Any]:
        """Get output configuration section."""
        return self.config.get('output', {})
    
    def update(self, key: str, value: Any):
        """
        Update a configuration value.
        
        Args:
            key: Configuration key in dot notation
            value: New value
        """
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
        logger.info(f"Updated configuration: {key} = {value}")
    
    def save(self, path: Optional[str] = None):
        """
        Save the current configuration to a file.
        
        Args:
            path: Path to save the configuration. If None, saves to the original path.
        """
        save_path = Path(path) if path else self.config_path
        
        try:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                yaml.dump(self.config, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save configuration to {save_path}: {e}")
            raise


# Global configuration instance
_config_manager: Optional[ConfigManager] = None


def get_config() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def init_logging():
    """Initialize logging based on configuration."""
    config = get_config()
    logging_config = config.get_logging_config()
    
    # Remove default logger
    logger.remove()
    
    # Add console logger
    logger.add(
        lambda msg: print(msg, end=""),
        level=logging_config.get('level', 'INFO'),
        format=logging_config.get('format', "{time} | {level} | {message}")
    )
    
    # Add file logger if path is specified
    log_file = logging_config.get('file_path')
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file,
            level=logging_config.get('level', 'INFO'),
            format=logging_config.get('format', "{time} | {level} | {name}:{function}:{line} | {message}"),
            rotation=logging_config.get('rotation', '1 day'),
            retention=logging_config.get('retention', '30 days'),
            compression=logging_config.get('compression', 'zip')
        )
    
    logger.info("Logging initialized") 