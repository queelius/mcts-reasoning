"""
Command parser and handlers for TUI.

Implements slash commands similar to Claude Code:
- /model <provider> [model] - Switch LLM provider/model
- /temperature <value> - Set temperature
- /ask <question> - Start reasoning session
- /search <n> - Run N more simulations
- /tree - Visualize search tree
- /sample <n> - Sample N diverse paths
- /consistency - Check solution consistency
- /solution - Show best solution
- /save [filename] - Save session
- /load <filename> - Load session
- /status - Show current status
- /help - Show help
- /exit - Exit TUI
"""

from typing import Optional, List, Tuple, Any
import shlex
from dataclasses import dataclass

from .session import SessionState


@dataclass
class Command:
    """Represents a parsed command."""
    name: str
    args: List[str]
    raw: str


class CommandParser:
    """Parse slash commands."""

    @staticmethod
    def parse(line: str) -> Optional[Command]:
        """
        Parse a command line.

        Args:
            line: Input line

        Returns:
            Command object if line starts with /, None otherwise
        """
        line = line.strip()

        if not line.startswith('/'):
            return None

        # Remove leading /
        line = line[1:]

        if not line:
            return None

        # Split into tokens (handling quotes)
        try:
            tokens = shlex.split(line)
        except ValueError:
            # Fallback to simple split if shlex fails
            tokens = line.split()

        if not tokens:
            return None

        return Command(
            name=tokens[0].lower(),
            args=tokens[1:],
            raw=line
        )


class CommandHandler:
    """Handle command execution."""

    def __init__(self, session: SessionState):
        """Initialize with session state."""
        self.session = session
        self.handlers = {
            'model': self.handle_model,
            'models': self.handle_models,
            'model-info': self.handle_model_info,
            'temperature': self.handle_temperature,
            'temp': self.handle_temperature,  # Alias
            'exploration': self.handle_exploration,
            'ask': self.handle_ask,
            'search': self.handle_search,
            'continue': self.handle_search,  # Alias
            'tree': self.handle_tree,
            'sample': self.handle_sample,
            'consistency': self.handle_consistency,
            'solution': self.handle_solution,
            'save': self.handle_save,
            'load': self.handle_load,
            'status': self.handle_status,
            'mcp-enable': self.handle_mcp_enable,
            'mcp-connect': self.handle_mcp_connect,
            'mcp-list': self.handle_mcp_list,
            'mcp-tools': self.handle_mcp_tools,
            'help': self.handle_help,
            'exit': self.handle_exit,
            'quit': self.handle_exit,  # Alias
        }

    def execute(self, command: Command) -> Tuple[bool, str]:
        """
        Execute a command.

        Args:
            command: Parsed command

        Returns:
            (success: bool, message: str)
        """
        handler = self.handlers.get(command.name)

        if handler is None:
            return False, f"Unknown command: /{command.name}. Type /help for available commands."

        try:
            return handler(command.args)
        except Exception as e:
            return False, f"Error executing /{command.name}: {e}"

    # ========== Command Handlers ==========

    def handle_model(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /model command."""
        if not args:
            # Show current model
            provider_name = self.session.llm_provider.get_provider_name()
            return True, f"Current model: {provider_name}"

        provider = args[0]
        model = args[1] if len(args) > 1 else None

        # Parse optional kwargs (like base_url for Ollama)
        kwargs = {}
        for i in range(2, len(args)):
            if '=' in args[i]:
                key, value = args[i].split('=', 1)
                kwargs[key] = value

        if self.session.switch_model(provider, model, **kwargs):
            provider_name = self.session.llm_provider.get_provider_name()
            return True, f"Switched to {provider_name}"
        else:
            return False, f"Failed to switch to {provider}" + (f"/{model}" if model else "")

    def handle_temperature(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /temperature command."""
        if not args:
            return True, f"Current temperature: {self.session.temperature}"

        try:
            temp = float(args[0])
            if self.session.set_temperature(temp):
                return True, f"Temperature set to {temp}"
            else:
                return False, "Temperature must be between 0.0 and 2.0"
        except ValueError:
            return False, "Invalid temperature value"

    def handle_exploration(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /exploration command."""
        if not args:
            return True, f"Current exploration constant: {self.session.exploration_constant}"

        try:
            constant = float(args[0])
            if self.session.set_exploration(constant):
                return True, f"Exploration constant set to {constant}"
            else:
                return False, "Exploration constant must be positive"
        except ValueError:
            return False, "Invalid exploration constant"

    def handle_models(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /models command - list available models."""
        from ..compositional.providers import OllamaProvider
        from ..config import get_config

        # Check if current provider supports listing models
        if isinstance(self.session.llm_provider, OllamaProvider):
            models = self.session.llm_provider.list_models()
            if not models:
                return False, "No models found on Ollama server"

            # Format output
            output = f"Available models on {self.session.llm_provider.base_url}:\n"
            for i, model in enumerate(models, 1):
                name = model.get("name", "unknown")
                size = model.get("size", 0) / (1024**3)  # Convert to GB
                output += f"  {i}. {name} ({size:.1f}GB)\n"

            # Add recent models
            config = get_config()
            recent = config.get_recent_models()
            if recent:
                output += "\nRecently used:\n"
                for entry in recent:
                    output += f"  - {entry['provider']}/{entry['model']}\n"

            return True, output
        else:
            # Show recent models for non-Ollama providers
            config = get_config()
            recent = config.get_recent_models()
            if recent:
                output = "Recently used models:\n"
                for entry in recent:
                    output += f"  - {entry['provider']}/{entry['model']}\n"
                return True, output
            else:
                return True, "No recently used models"

    def handle_model_info(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /model-info command - show model information."""
        from ..compositional.providers import OllamaProvider

        # Check if current provider supports model info
        if isinstance(self.session.llm_provider, OllamaProvider):
            model_name = args[0] if args else None
            info = self.session.llm_provider.get_model_info(model_name)

            if not info:
                model = model_name or self.session.llm_provider.model
                return False, f"Failed to get info for model: {model}"

            # Format output
            output = f"Model: {info.get('modelfile', 'unknown')}\n"
            output += f"Template: {info.get('template', 'N/A')[:100]}...\n"

            params = info.get('parameters', {})
            if params:
                output += "Parameters:\n"
                for key, value in list(params.items())[:5]:  # Show first 5
                    output += f"  {key}: {value}\n"

            return True, output
        else:
            # For other providers, just show basic info
            output = f"Provider: {self.session.provider_name}\n"
            output += f"Model: {self.session.model_name}\n"
            output += f"Temperature: {self.session.temperature}\n"
            return True, output

    def handle_ask(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /ask command."""
        if not args:
            return False, "Usage: /ask <question>"

        question = " ".join(args)
        self.session.current_question = question
        self.session.initial_state = f"Question: {question}\n\nLet me think about this systematically."

        # Initialize MCTS
        self.session.initialize_mcts()

        return True, f"Starting reasoning session for:\n{question}\n\nUse /search <n> to run simulations."

    def handle_search(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /search command."""
        if self.session.mcts is None:
            return False, "No active reasoning session. Use /ask <question> first."

        # Default to 20 simulations
        n_sims = 20
        if args:
            try:
                n_sims = int(args[0])
            except ValueError:
                return False, "Invalid number of simulations"

        # Run search
        if self.session.mcts.root is None:
            # First search
            self.session.mcts.search(self.session.initial_state, simulations=n_sims)
            msg = f"Completed {n_sims} simulations."
        else:
            # Continue search
            self.session.mcts.search(self.session.mcts.root.state, simulations=n_sims)
            msg = f"Completed {n_sims} more simulations."

        stats = self.session.mcts.stats
        msg += f"\n\nTree statistics:"
        msg += f"\n  Total nodes: {stats['total_nodes']}"
        msg += f"\n  Max depth: {stats['max_depth']}"
        msg += f"\n  Best value: {stats['best_value']:.3f}"

        return True, msg

    def handle_tree(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /tree command."""
        if self.session.mcts is None or self.session.mcts.root is None:
            return False, "No search tree available"

        # Return tree visualization data
        # The TUI will render this specially
        return True, "TREE_VISUALIZATION"

    def handle_sample(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /sample command."""
        if self.session.mcts is None or self.session.mcts.root is None:
            return False, "No search tree available"

        n = 5  # Default
        if args:
            try:
                n = int(args[0])
            except ValueError:
                return False, "Invalid number"

        paths = self.session.mcts.sample(n=n, strategy="diverse", temperature=1.5)

        msg = f"Sampled {len(paths)} diverse reasoning paths:\n"
        for i, path in enumerate(paths, 1):
            msg += f"\n--- Path {i} (length={path.length}, value={path.total_value:.2f}) ---\n"
            msg += path.final_state[-300:] if len(path.final_state) > 300 else path.final_state
            msg += "\n"

        return True, msg

    def handle_consistency(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /consistency command."""
        if self.session.mcts is None or self.session.mcts.root is None:
            return False, "No search tree available"

        n_samples = 10
        if args:
            try:
                n_samples = int(args[0])
            except ValueError:
                return False, "Invalid number"

        result = self.session.mcts.check_consistency(n_samples=n_samples)

        msg = f"Consistency check ({n_samples} samples):\n"
        msg += f"  Confidence: {result['confidence']:.1%}\n"
        msg += f"  Support: {result['support']}/{result['total_samples']} samples\n"
        msg += f"  Clusters: {len(result['clusters'])}\n"
        msg += f"\nMost consistent solution:\n"
        msg += result['solution'][-500:] if len(result['solution']) > 500 else result['solution']

        return True, msg

    def handle_solution(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /solution command."""
        if self.session.mcts is None or self.session.mcts.root is None:
            return False, "No search tree available"

        solution = self.session.mcts.solution
        confidence = self.session.mcts.best_value

        msg = f"Best solution (confidence={confidence:.2%}):\n\n"
        msg += solution

        return True, msg

    def handle_save(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /save command."""
        filename = args[0] if args else None

        try:
            filepath = self.session.save_session(filename)
            return True, f"Session saved to {filepath}"
        except Exception as e:
            return False, f"Failed to save session: {e}"

    def handle_load(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /load command."""
        if not args:
            return False, "Usage: /load <filename>"

        from pathlib import Path
        filepath = Path(args[0])

        if not filepath.exists():
            # Try in save directory
            filepath = self.session.save_dir / args[0]

        if not filepath.exists():
            return False, f"File not found: {args[0]}"

        if self.session.load_session(filepath):
            return True, f"Session loaded from {filepath}"
        else:
            return False, "Failed to load session"

    def handle_status(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /status command."""
        status = self.session.get_status()

        msg = "Current status:\n"
        msg += f"  Provider: {status['provider']}\n"
        msg += f"  Model: {status['model']}\n"
        msg += f"  Temperature: {status['temperature']}\n"
        msg += f"  Exploration: {status['exploration']}\n"
        msg += f"  Max depth: {status['max_depth']}\n"
        msg += f"  Compositional: {status['compositional']}\n"
        msg += f"  MCP enabled: {status['mcp_enabled']}\n"

        if status['question']:
            msg += f"\nCurrent question:\n  {status['question']}\n"

        if status['has_tree']:
            msg += f"\nSearch tree:\n"
            msg += f"  Nodes: {status['tree_nodes']}\n"
            msg += f"  Depth: {status['tree_depth']}\n"
            msg += f"  Best value: {status['best_value']:.3f}\n"

        return True, msg

    def handle_mcp_enable(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /mcp-enable command."""
        if self.session.enable_mcp():
            tools = list(self.session.mcp_client.tools.keys()) if self.session.mcp_client else []
            msg = "MCP enabled"
            if tools:
                msg += f"\nAvailable tools: {', '.join(tools)}"
            return True, msg
        else:
            return False, "Failed to enable MCP"

    def handle_mcp_connect(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /mcp-connect command."""
        if not self.session.mcp_enabled:
            return False, "MCP not enabled. Use /mcp-enable first."

        if not args:
            return False, "Usage: /mcp-connect <server_name> <type>\nTypes: python, web, filesystem"

        server_name = args[0]
        server_type = args[1] if len(args) > 1 else "python"

        try:
            self.session.mcp_client.connect_server(server_name, {"type": server_type})
            return True, f"Connected to MCP server: {server_name} (type: {server_type})"
        except Exception as e:
            return False, f"Failed to connect: {e}"

    def handle_mcp_list(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /mcp-list command."""
        if not self.session.mcp_enabled or not self.session.mcp_client:
            return False, "MCP not enabled"

        servers = self.session.mcp_client.servers
        if not servers:
            return True, "No MCP servers connected"

        msg = "Connected MCP servers:\n"
        for name, config in servers.items():
            msg += f"  - {name}: {config.get('type', 'unknown')}\n"

        return True, msg

    def handle_mcp_tools(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /mcp-tools command."""
        if not self.session.mcp_enabled or not self.session.mcp_client:
            return False, "MCP not enabled"

        tools = self.session.mcp_client.tools
        if not tools:
            return True, "No tools available"

        msg = "Available MCP tools:\n"
        for name, tool in tools.items():
            msg += f"\n  {name}:\n"
            msg += f"    {tool.description}\n"
            msg += f"    Server: {tool.server_name or 'N/A'}\n"

        return True, msg

    def handle_help(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /help command."""
        help_text = """
Available commands:

Session Management:
  /ask <question>        - Start a new reasoning session
  /search <n>            - Run N simulations (default: 20)
  /continue <n>          - Continue search (alias for /search)
  /solution              - Show best solution
  /save [filename]       - Save session
  /load <filename>       - Load session
  /status                - Show current status

Configuration:
  /model [provider] [model] [key=value...]  - Switch LLM or show current
  /models                                   - List available models
  /model-info [model]                       - Show model information
  /temperature <value>                      - Set temperature (0.0-2.0)
  /exploration <value>                      - Set exploration constant

Analysis:
  /tree                  - Visualize search tree
  /sample <n>            - Sample N diverse paths
  /consistency [n]       - Check solution consistency

MCP Tools:
  /mcp-enable            - Enable MCP integration
  /mcp-connect <name> <type>  - Connect to MCP server
  /mcp-list              - List connected servers
  /mcp-tools             - Show available tools

Other:
  /help                  - Show this help
  /exit                  - Exit TUI

Examples:
  /model                                 # Show current model
  /model openai gpt-4                    # Switch to OpenAI GPT-4
  /model ollama llama2 base_url=http://192.168.0.225:11434
  /models                                # List available models
  /model-info                            # Show current model info
  /ask What is the sum of primes less than 20?
  /search 50
  /sample 5
  /consistency 20
"""
        return True, help_text

    def handle_exit(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /exit command."""
        return True, "EXIT"


__all__ = ['Command', 'CommandParser', 'CommandHandler']
