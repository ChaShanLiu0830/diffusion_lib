import yaml
from typing import Any, Dict, Optional


class ConfigManager:
    """Configuration manager for loading and managing YAML configurations."""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Loads configuration from a YAML file.

        Args:
            config_path: Path to the configuration YAML file.
        """
        self.config = {}
        if config_path:
            self.load_config(config_path)

    def load_config(self, config_path: str) -> None:
        """
        Load configuration from YAML file.
        
        Args:
            config_path: Path to the configuration YAML file.
        """
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Error parsing YAML file: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        """
        Retrieves configuration by key with support for nested keys.

        Args:
            key: Key to fetch configuration data (supports dot notation for nested keys).
            default: Default value if key is not found.

        Returns:
            Config data associated with the key.
        """
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value

    def set(self, key: str, value: Any) -> None:
        """
        Set configuration value by key.
        
        Args:
            key: Key to set (supports dot notation for nested keys).
            value: Value to set.
        """
        keys = key.split('.')
        config_ref = self.config
        
        # Navigate to the parent of the target key
        for k in keys[:-1]:
            if k not in config_ref:
                config_ref[k] = {}
            config_ref = config_ref[k]
        
        # Set the value
        config_ref[keys[-1]] = value

    def update(self, updates: Dict[str, Any]) -> None:
        """
        Update configuration with a dictionary of updates.
        
        Args:
            updates: Dictionary of key-value pairs to update.
        """
        for key, value in updates.items():
            self.set(key, value)

    def save_config(self, output_path: str) -> None:
        """
        Save current configuration to a YAML file.
        
        Args:
            output_path: Path where to save the configuration.
        """
        with open(output_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, indent=2)

    def get_model_config(self) -> Dict[str, Any]:
        """
        Get model configuration.
        
        Returns:
            Model configuration dictionary.
        """
        return self.get('model', {})

    def get_training_config(self) -> Dict[str, Any]:
        """
        Get training configuration.
        
        Returns:
            Training configuration dictionary.
        """
        return self.get('training', {})

    def get_diffusion_config(self) -> Dict[str, Any]:
        """
        Get diffusion method configuration.
        
        Returns:
            Diffusion configuration dictionary.
        """
        return self.get('diffusion', {})

    def get_data_config(self) -> Dict[str, Any]:
        """
        Get data configuration.
        
        Returns:
            Data configuration dictionary.
        """
        return self.get('data', {})

    def validate_required_keys(self, required_keys: list) -> None:
        """
        Validate that required configuration keys exist.
        
        Args:
            required_keys: List of required configuration keys.
            
        Raises:
            ValueError: If any required key is missing.
        """
        missing_keys = []
        for key in required_keys:
            if self.get(key) is None:
                missing_keys.append(key)
        
        if missing_keys:
            raise ValueError(f"Missing required configuration keys: {missing_keys}")

    def __str__(self) -> str:
        """String representation of the configuration."""
        return yaml.dump(self.config, default_flow_style=False, indent=2)

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-style setting."""
        self.set(key, value) 