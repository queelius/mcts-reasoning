"""
Session state management for TUI.

Maintains:
- Current MCTS tree
- LLM provider configuration
- MCP client connections
- Settings (temperature, exploration, etc.)
- Conversation history
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
import json
from pathlib import Path

from ..core import MCTS
from ..reasoning import ReasoningMCTS
from ..compositional.providers import LLMProvider, get_llm, MockLLMProvider
from ..config import get_config


@dataclass
class SessionState:
    """Maintains session state across commands."""

    # Current MCTS instance
    mcts: Optional[ReasoningMCTS] = None

    # LLM configuration
    llm_provider: Optional[LLMProvider] = None
    provider_name: str = "mock"
    model_name: str = "default"

    # MCP configuration
    mcp_client: Optional[Any] = None  # MCPClient type
    mcp_enabled: bool = False

    # Settings
    temperature: float = 0.7
    exploration_constant: float = 1.414
    max_rollout_depth: int = 5
    use_compositional: bool = True

    # Current question/task
    current_question: Optional[str] = None
    initial_state: Optional[str] = None

    # History
    command_history: List[str] = field(default_factory=list)
    session_start: datetime = field(default_factory=datetime.now)

    # Metadata
    session_name: Optional[str] = None
    save_dir: Path = field(default_factory=lambda: Path.home() / ".mcts-reasoning" / "sessions")

    def __post_init__(self):
        """Initialize with default LLM from config."""
        if self.llm_provider is None:
            # Load from config
            config = get_config()
            self.provider_name = config.get("default_provider", "mock")

            # Try to initialize the configured provider
            try:
                if self.provider_name != "mock":
                    provider_config = config.get_provider_config(self.provider_name)
                    self.model_name = provider_config.get("model", "default")
                    self.llm_provider = get_llm(self.provider_name, model=self.model_name)
                else:
                    self.llm_provider = MockLLMProvider()
                    self.model_name = "default"
            except Exception:
                # Fall back to mock if configured provider fails
                self.llm_provider = MockLLMProvider()
                self.provider_name = "mock"
                self.model_name = "default"

            # Load other settings from config
            self.temperature = config.get("providers." + self.provider_name + ".temperature", 0.7)
            self.exploration_constant = config.get("mcts.exploration_constant", 1.414)
            self.max_rollout_depth = config.get("mcts.max_rollout_depth", 5)
            self.use_compositional = config.get("mcts.use_compositional", True)

    def initialize_mcts(self) -> ReasoningMCTS:
        """Create or reset MCTS instance with current settings."""
        mcts = (
            ReasoningMCTS()
            .with_llm(self.llm_provider)
            .with_exploration(self.exploration_constant)
            .with_max_rollout_depth(self.max_rollout_depth)
            .with_compositional_actions(enabled=self.use_compositional)
        )

        if self.current_question:
            mcts.with_question(self.current_question)

        self.mcts = mcts
        return mcts

    def switch_model(self, provider: str, model: Optional[str] = None, **kwargs) -> bool:
        """
        Switch to a different LLM provider/model.

        Args:
            provider: Provider name (openai, anthropic, ollama, mock)
            model: Model name (optional)
            **kwargs: Additional provider-specific arguments (e.g., base_url for Ollama)

        Returns:
            True if successful
        """
        try:
            config = get_config()

            if provider == "mock":
                self.llm_provider = MockLLMProvider()
                self.provider_name = "mock"
                self.model_name = "default"
            else:
                # Get config for this provider
                provider_config = config.get_provider_config(provider)

                # Use provided model or config default
                if model:
                    provider_config["model"] = model
                    self.model_name = model
                else:
                    self.model_name = provider_config.get("model", "default")

                # Merge kwargs (like base_url) into provider_config
                provider_config.update(kwargs)

                # Filter provider_config to only include parameters the provider accepts in __init__
                # Note: OllamaProvider doesn't accept temperature/max_tokens in __init__,
                # but they're still saved in config and used in generate() calls
                if provider.lower() == "ollama":
                    # Only pass init params for Ollama
                    init_params = {k: v for k, v in provider_config.items()
                                  if k in ['model', 'host', 'port', 'base_url']}
                    self.llm_provider = get_llm(provider, **init_params)
                else:
                    # Other providers (OpenAI, Anthropic) accept temperature/max_tokens in __init__
                    self.llm_provider = get_llm(provider, **provider_config)

                self.provider_name = provider

                # Save updated config
                config.set_provider_config(provider, provider_config, save=True)

                # Add to recent models
                config.add_recent_model(provider, self.model_name)

            # Update default provider in config
            config.set("default_provider", provider, save=True)

            # Recreate MCTS if it exists
            if self.mcts is not None:
                self.initialize_mcts()

            return True

        except Exception as e:
            # Log the actual error for debugging
            import logging
            logging.error(f"Failed to switch model to {provider}/{model}: {e}", exc_info=True)
            # Store error message for user
            self._last_error = str(e)
            return False

    def set_temperature(self, temp: float) -> bool:
        """Set temperature."""
        if 0.0 <= temp <= 2.0:
            self.temperature = temp
            return True
        return False

    def set_exploration(self, constant: float) -> bool:
        """Set exploration constant."""
        if constant > 0:
            self.exploration_constant = constant
            if self.mcts:
                self.mcts.exploration_constant = constant
            return True
        return False

    def enable_mcp(self) -> bool:
        """Enable MCP integration."""
        try:
            from ..compositional.mcp import create_mcp_client, create_mcp_provider

            if self.mcp_client is None:
                self.mcp_client = create_mcp_client({
                    "python": {"type": "python"}
                })

            # Wrap current LLM
            self.llm_provider = create_mcp_provider(
                self.llm_provider,
                mcp_client=self.mcp_client
            )

            self.mcp_enabled = True

            # Recreate MCTS
            if self.mcts is not None:
                self.initialize_mcts()

            return True

        except ImportError:
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get current session status."""
        status = {
            "provider": self.provider_name,
            "model": self.model_name,
            "temperature": self.temperature,
            "exploration": self.exploration_constant,
            "max_depth": self.max_rollout_depth,
            "compositional": self.use_compositional,
            "mcp_enabled": self.mcp_enabled,
            "question": self.current_question,
            "has_tree": self.mcts is not None and self.mcts.root is not None,
        }

        if self.mcts and self.mcts.root:
            stats = self.mcts.stats
            status.update({
                "tree_nodes": stats.get("total_nodes", 0),
                "tree_depth": stats.get("max_depth", 0),
                "best_value": stats.get("best_value", 0.0),
            })

        return status

    def save_session(self, filename: Optional[str] = None) -> Path:
        """Save session to file."""
        self.save_dir.mkdir(parents=True, exist_ok=True)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"session_{timestamp}.json"

        filepath = self.save_dir / filename

        session_data = {
            "session_name": self.session_name,
            "session_start": self.session_start.isoformat(),
            "provider": self.provider_name,
            "model": self.model_name,
            "temperature": self.temperature,
            "exploration": self.exploration_constant,
            "max_depth": self.max_rollout_depth,
            "compositional": self.use_compositional,
            "mcp_enabled": self.mcp_enabled,
            "question": self.current_question,
            "initial_state": self.initial_state,
            "command_history": self.command_history[-100:],  # Last 100 commands
        }

        # Save MCTS tree if it exists
        if self.mcts and self.mcts.root:
            tree_file = filepath.with_suffix('.tree.json')
            self.mcts.save(tree_file)
            session_data["tree_file"] = str(tree_file)

        with open(filepath, 'w') as f:
            json.dump(session_data, f, indent=2)

        return filepath

    def load_session(self, filepath: Path) -> bool:
        """Load session from file."""
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)

            self.session_name = data.get("session_name")
            self.provider_name = data.get("provider", "mock")
            self.model_name = data.get("model", "default")
            self.temperature = data.get("temperature", 0.7)
            self.exploration_constant = data.get("exploration", 1.414)
            self.max_rollout_depth = data.get("max_depth", 5)
            self.use_compositional = data.get("compositional", True)
            self.mcp_enabled = data.get("mcp_enabled", False)
            self.current_question = data.get("question")
            self.initial_state = data.get("initial_state")
            self.command_history = data.get("command_history", [])

            # Restore LLM
            self.switch_model(self.provider_name, self.model_name)

            # Restore MCP if enabled
            if self.mcp_enabled:
                self.enable_mcp()

            # Restore tree if it exists
            tree_file = data.get("tree_file")
            if tree_file and Path(tree_file).exists():
                self.mcts = ReasoningMCTS.load(tree_file)
                self.mcts.llm = self.llm_provider

            return True

        except Exception:
            return False


__all__ = ['SessionState']
