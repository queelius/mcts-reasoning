"""
Configuration management for MCTS-Reasoning.

Handles loading and saving configuration from ~/.mcts-reasoning/config.json
"""

import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class Config:
    """Manages MCTS-Reasoning configuration."""

    DEFAULT_CONFIG = {
        "default_provider": "mock",
        "default_model": "default",
        "providers": {
            "openai": {"model": "gpt-4", "temperature": 0.7},
            "anthropic": {"model": "claude-3-5-sonnet-20241022", "temperature": 0.7},
            "ollama": {
                "model": "llama2",
                "base_url": None,  # Will use localhost if None
                "temperature": 0.7,
            },
        },
        "tui": {"use_rich": True, "save_history": True, "max_history": 100},
        "mcts": {
            "exploration_constant": 1.414,
            "max_rollout_depth": 5,
            "use_compositional": True,
        },
        "recent_models": [],
    }

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize config manager.

        Args:
            config_dir: Directory for config files. Defaults to ~/.mcts-reasoning
        """
        self.config_dir = config_dir or Path.home() / ".mcts-reasoning"
        self.config_file = self.config_dir / "config.json"
        self._config = None

    def load(self) -> Dict[str, Any]:
        """
        Load configuration from file.

        Returns:
            Configuration dictionary
        """
        if self._config is not None:
            return self._config

        # Create config directory if it doesn't exist
        self.config_dir.mkdir(parents=True, exist_ok=True)

        # Load from file if it exists
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    loaded_config = json.load(f)
                    # Merge with defaults (in case new keys were added)
                    self._config = self._merge_configs(
                        self.DEFAULT_CONFIG.copy(), loaded_config
                    )
                    logger.info(f"Loaded config from {self.config_file}")
            except Exception as e:
                logger.warning(f"Failed to load config from {self.config_file}: {e}")
                logger.info("Using default configuration")
                self._config = self.DEFAULT_CONFIG.copy()
        else:
            logger.info("No config file found, using defaults")
            self._config = self.DEFAULT_CONFIG.copy()
            # Save defaults
            self.save()

        return self._config

    def save(self) -> bool:
        """
        Save configuration to file.

        Returns:
            True if successful, False otherwise
        """
        if self._config is None:
            self._config = self.DEFAULT_CONFIG.copy()

        try:
            self.config_dir.mkdir(parents=True, exist_ok=True)
            with open(self.config_file, "w") as f:
                json.dump(self._config, f, indent=2)
            logger.info(f"Saved config to {self.config_file}")
            return True
        except Exception as e:
            logger.error(f"Failed to save config to {self.config_file}: {e}")
            return False

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a configuration value.

        Args:
            key: Configuration key (supports dot notation like "providers.ollama.base_url")
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        config = self.load()

        # Support dot notation
        keys = key.split(".")
        value = config
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default

        return value

    def set(self, key: str, value: Any, save: bool = True) -> bool:
        """
        Set a configuration value.

        Args:
            key: Configuration key (supports dot notation)
            value: Value to set
            save: Whether to save to file immediately

        Returns:
            True if successful, False otherwise
        """
        config = self.load()

        # Support dot notation
        keys = key.split(".")
        current = config
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]

        current[keys[-1]] = value
        self._config = config

        if save:
            return self.save()
        return True

    def get_provider_config(self, provider: str) -> Dict[str, Any]:
        """
        Get configuration for a specific provider.

        Args:
            provider: Provider name (openai, anthropic, ollama)

        Returns:
            Provider configuration dictionary
        """
        config = self.load()
        return config.get("providers", {}).get(provider, {})

    def set_provider_config(
        self, provider: str, config_dict: Dict[str, Any], save: bool = True
    ) -> bool:
        """
        Set configuration for a specific provider.

        Args:
            provider: Provider name
            config_dict: Configuration dictionary
            save: Whether to save immediately

        Returns:
            True if successful
        """
        config = self.load()
        if "providers" not in config:
            config["providers"] = {}

        config["providers"][provider] = config_dict
        self._config = config

        if save:
            return self.save()
        return True

    def add_recent_model(self, provider: str, model: str, max_recent: int = 10) -> bool:
        """
        Add a model to the recent models list.

        Args:
            provider: Provider name
            model: Model name
            max_recent: Maximum number of recent models to keep

        Returns:
            True if successful
        """
        config = self.load()
        recent = config.get("recent_models", [])

        # Create entry
        entry = {"provider": provider, "model": model}

        # Remove if already exists
        recent = [
            r
            for r in recent
            if not (r.get("provider") == provider and r.get("model") == model)
        ]

        # Add to front
        recent.insert(0, entry)

        # Trim to max
        recent = recent[:max_recent]

        config["recent_models"] = recent
        self._config = config

        return self.save()

    def get_recent_models(self, limit: int = 5) -> list:
        """
        Get recently used models.

        Args:
            limit: Maximum number to return

        Returns:
            List of recent model entries
        """
        config = self.load()
        recent = config.get("recent_models", [])
        return recent[:limit]

    @staticmethod
    def _merge_configs(base: Dict, override: Dict) -> Dict:
        """
        Recursively merge two configuration dictionaries.

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        result = base.copy()

        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                result[key] = Config._merge_configs(result[key], value)
            else:
                result[key] = value

        return result


# Global config instance
_global_config = None


def get_config() -> Config:
    """
    Get the global configuration instance.

    Returns:
        Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config()
    return _global_config
