"""
Command parser and handlers for TUI.

Implements CLI commands (slash optional for backward compatibility):
- model <provider> [model] - Switch LLM provider/model
- models - List available models
- model-info [model] - Show model information
- probe <url> [provider] - Probe endpoint for available models
- temperature <value> - Set temperature
- ask <question> - Start reasoning session
- search <n> - Run N more simulations
- tree - Visualize search tree
- sample <n> - Sample N diverse paths
- consistency - Check solution consistency
- solution - Show best solution
- nodes - List all nodes in tree
- inspect <index> - Show detailed node info (preview)
- inspect-full <index> - Show FULL state (no truncation)
- show-prompt <index> - Show exact prompt LLM saw
- path <index> - Show reasoning path to node
- export-tree <file> - Export tree to JSON
- stats - Show context management and solution detection stats
- solutions - List all finalized solutions
- compare <i1> <i2> ... - Compare multiple nodes
- config [feature] [on|off] - Configure features or show current config
- save [filename] - Save session
- load <filename> - Load session
- status - Show current status
- help - Show help
- exit - Exit TUI
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
    """Parse commands (with or without leading slash)."""

    @staticmethod
    def parse(line: str) -> Optional[Command]:
        """
        Parse a command line.

        Args:
            line: Input line (e.g., "ask How does this work?" or "/ask How does this work?")

        Returns:
            Command object, or None if line is empty
        """
        line = line.strip()

        if not line:
            return None

        # Remove leading slash if present (optional for backward compatibility)
        if line.startswith('/'):
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
            'probe': self.handle_probe,
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
            # Diagnostic commands
            'nodes': self.handle_nodes,
            'inspect': self.handle_inspect,
            'inspect-full': self.handle_inspect_full,
            'show-prompt': self.handle_show_prompt,
            'path': self.handle_path,
            'export': self.handle_export,
            'export-tree': self.handle_export_tree,
            'verify': self.handle_verify,
            # Stats and solution commands
            'stats': self.handle_stats,
            'solutions': self.handle_solutions,
            'compare': self.handle_compare,
            'config': self.handle_config,
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
            error_msg = f"Failed to switch to {provider}" + (f"/{model}" if model else "")
            # Include actual error if available
            if hasattr(self.session, '_last_error'):
                error_msg += f"\nReason: {self.session._last_error}"
            return False, error_msg

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

    def handle_probe(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /probe command - probe an endpoint for available models."""
        from ..compositional.providers import OllamaProvider, OpenAIProvider, AnthropicProvider, ProviderFactory

        if not args:
            return False, "Usage: /probe <endpoint_url> [provider_type]\nExample: /probe http://192.168.0.225:11434 ollama"

        endpoint_url = args[0]
        provider_type = args[1].lower() if len(args) > 1 else "ollama"  # Default to Ollama

        # Map provider type to class
        provider_map = {
            'ollama': OllamaProvider,
            'openai': OpenAIProvider,
            'anthropic': AnthropicProvider,
        }

        provider_class = provider_map.get(provider_type)
        if provider_class is None:
            return False, f"Unknown provider type: {provider_type}\nSupported: {', '.join(provider_map.keys())}"

        # Probe the endpoint
        try:
            result = provider_class.probe_endpoint(endpoint_url)

            if not result.get('available', False):
                error = result.get('error', 'Unknown error')
                return False, f"Endpoint not available: {error}"

            # Format successful response
            output = f"Endpoint probe successful!\n"
            output += f"URL: {result.get('base_url', endpoint_url)}\n"
            output += f"Provider: {provider_type}\n"

            models = result.get('models', [])
            if models:
                output += f"\nAvailable models ({len(models)}):\n"
                for i, model in enumerate(models[:20], 1):  # Show up to 20
                    if isinstance(model, str):
                        output += f"  {i}. {model}\n"
                    else:
                        # Handle dict format (with size, etc.)
                        name = model.get('name', 'unknown')
                        output += f"  {i}. {name}\n"

                if len(models) > 20:
                    output += f"  ... and {len(models) - 20} more\n"

                output += f"\nTo use a model from this endpoint:\n"
                output += f"  /model {provider_type} <model_name> base_url={endpoint_url}\n"
            else:
                output += "\nNo models found on this endpoint.\n"

            return True, output

        except Exception as e:
            return False, f"Failed to probe endpoint: {e}"

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

    def handle_nodes(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /nodes command - list all nodes in the tree."""
        if self.session.mcts is None or self.session.mcts.root is None:
            return False, "No search tree available"

        nodes = self.session.mcts.get_all_nodes()

        msg = f"Tree nodes ({len(nodes)} total):\n\n"
        msg += f"{'Index':<6} {'Depth':<6} {'Visits':<8} {'Value':<12} {'Avg':<10} {'Action':<40}\n"
        msg += "=" * 90 + "\n"

        for i, node in enumerate(nodes):
            avg_value = node.value / node.visits if node.visits > 0 else 0.0
            action_str = str(node.action_taken) if node.action_taken else "ROOT"
            # Truncate action if too long
            if len(action_str) > 37:
                action_str = action_str[:34] + "..."

            msg += f"{i:<6} {node.depth:<6} {node.visits:<8} {node.value:<12.2f} {avg_value:<10.3f} {action_str}\n"

        msg += "\nUse /inspect <index> to see details about a specific node"

        return True, msg

    def handle_inspect(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /inspect command - show detailed info about a node."""
        if self.session.mcts is None or self.session.mcts.root is None:
            return False, "No search tree available"

        if not args:
            return False, "Usage: /inspect <node_index>"

        try:
            index = int(args[0])
        except ValueError:
            return False, "Invalid node index"

        node = self.session.mcts.get_node_by_index(index)
        if node is None:
            return False, f"Node {index} not found"

        details = self.session.mcts.get_node_details(node)

        msg = f"Node {index} Details:\n"
        msg += "=" * 70 + "\n"
        msg += f"Depth:        {details['depth']}\n"
        msg += f"Visits:       {details['visits']}\n"
        msg += f"Total Value:  {details['value']:.3f}\n"
        msg += f"Avg Value:    {details['avg_value']:.3f}\n"
        msg += f"UCB1:         {details['ucb1']:.3f}\n"
        msg += f"Children:     {details['num_children']}\n"
        msg += f"Is Leaf:      {details['is_leaf']}\n"
        msg += f"\nAction Taken:\n{details['action']}\n"
        msg += f"\nState ({details['state_length']} chars):\n"
        msg += "=" * 70 + "\n"
        msg += details['state_preview']
        msg += "\n" + "=" * 70

        return True, msg

    def handle_inspect_full(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /inspect-full command - show FULL state without truncation."""
        if self.session.mcts is None or self.session.mcts.root is None:
            return False, "No search tree available"

        if not args:
            return False, "Usage: /inspect-full <node_index>"

        try:
            index = int(args[0])
        except ValueError:
            return False, "Invalid node index"

        node = self.session.mcts.get_node_by_index(index)
        if node is None:
            return False, f"Node {index} not found"

        msg = f"Node {index} - FULL State:\n"
        msg += "=" * 70 + "\n"
        msg += f"Length: {len(node.state)} characters\n"
        msg += f"Depth: {node.depth}\n"
        msg += f"Action: {str(node.action_taken) if node.action_taken else 'ROOT'}\n"
        msg += "=" * 70 + "\n"
        msg += node.state  # Full state, no truncation!
        msg += "\n" + "=" * 70

        return True, msg

    def handle_show_prompt(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /show-prompt command - show what LLM saw when creating this node."""
        if self.session.mcts is None or self.session.mcts.root is None:
            return False, "No search tree available"

        if not args:
            return False, "Usage: /show-prompt <node_index>"

        try:
            index = int(args[0])
        except ValueError:
            return False, "Invalid node index"

        node = self.session.mcts.get_node_by_index(index)
        if node is None:
            return False, f"Node {index} not found"

        if node.is_root:
            return False, "Root node has no prompt (it's the initial state)"

        # Get parent node's state (what was passed to LLM)
        parent_state = node.parent.state if node.parent else ""

        # Try to reconstruct what the LLM saw
        from mcts_reasoning.compositional.actions import CompositionalAction

        msg = f"Prompt shown to LLM for Node {index}:\n"
        msg += "=" * 70 + "\n"
        msg += f"Action taken: {node.action_taken}\n"
        msg += "=" * 70 + "\n\n"

        if isinstance(node.action_taken, CompositionalAction):
            # Reconstruct the prompt using the action's method
            try:
                prompt = node.action_taken.to_prompt(
                    parent_state,
                    self.session.current_question or "Unknown question",
                    None
                )
                msg += "RECONSTRUCTED PROMPT:\n"
                msg += "-" * 70 + "\n"
                msg += prompt
                msg += "\n" + "-" * 70 + "\n"
                msg += f"\nPrompt length: {len(prompt)} characters"
            except Exception as e:
                msg += f"Error reconstructing prompt: {e}\n"
                msg += f"\nParent state ({len(parent_state)} chars):\n"
                msg += parent_state
        else:
            msg += f"Simple action (not compositional)\n"
            msg += f"Parent state used as context ({len(parent_state)} chars):\n"
            msg += "-" * 70 + "\n"
            msg += parent_state
            msg += "\n" + "-" * 70

        return True, msg

    def handle_path(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /path command - show path from root to a node."""
        if self.session.mcts is None or self.session.mcts.root is None:
            return False, "No search tree available"

        if not args:
            return False, "Usage: /path <node_index>"

        try:
            index = int(args[0])
        except ValueError:
            return False, "Invalid node index"

        node = self.session.mcts.get_node_by_index(index)
        if node is None:
            return False, f"Node {index} not found"

        path = node.path_to_root
        msg = f"Path to Node {index} ({len(path)} nodes):\n\n"

        for i, path_node in enumerate(path):
            action_str = str(path_node.action_taken) if path_node.action_taken else "ROOT"
            avg_val = path_node.value / path_node.visits if path_node.visits > 0 else 0.0

            msg += f"{'  ' * i}[{i}] {action_str}\n"
            msg += f"{'  ' * i}    visits={path_node.visits}, value={path_node.value:.2f}, avg={avg_val:.3f}\n"

            # Show state diff (last 200 chars of new content)
            if i > 0:
                prev_len = len(path[i-1].state)
                new_content = path_node.state[prev_len:]
                if len(new_content) > 200:
                    new_content = "..." + new_content[-200:]
                if new_content.strip():
                    msg += f"{'  ' * i}    Added: {new_content[:150]}...\n"

            msg += "\n"

        return True, msg

    def handle_export(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /export command - export tree in various formats."""
        if self.session.mcts is None or self.session.mcts.root is None:
            return False, "No search tree available"

        if not args:
            return False, "Usage: /export <format> <filename>\nFormats: json, markdown, dot, csv"

        from pathlib import Path
        import json

        fmt = args[0].lower()
        filename = args[1] if len(args) > 1 else None

        # Set default filename based on format
        if not filename:
            extensions = {'json': '.json', 'markdown': '.md', 'dot': '.dot', 'csv': '.csv'}
            filename = f"export{extensions.get(fmt, '.txt')}"

        filepath = Path(filename)

        try:
            nodes = self.session.mcts.get_all_nodes()

            if fmt == 'json':
                content = self._export_json(nodes)
            elif fmt in ['markdown', 'md']:
                content = self._export_markdown(nodes)
            elif fmt == 'dot':
                content = self._export_dot(nodes)
            elif fmt == 'csv':
                content = self._export_csv(nodes)
            else:
                return False, f"Unknown format: {fmt}. Use: json, markdown, dot, csv"

            # Save to file
            filepath.parent.mkdir(parents=True, exist_ok=True)
            filepath.write_text(content)

            return True, f"Exported to {filepath} ({len(nodes)} nodes)"

        except Exception as e:
            return False, f"Failed to export: {e}"

    def _export_json(self, nodes: List) -> str:
        """Export as JSON."""
        import json

        tree_data = self.session.mcts.to_json()

        # Add node list for easy inspection
        tree_data['node_list'] = [
            {
                'index': i,
                'depth': node.depth,
                'visits': node.visits,
                'value': node.value,
                'avg_value': node.value / node.visits if node.visits > 0 else 0.0,
                'action': str(node.action_taken) if node.action_taken else 'ROOT',
                'state_length': len(node.state),
                'num_children': len(node.children)
            }
            for i, node in enumerate(nodes)
        ]

        return json.dumps(tree_data, indent=2)

    def _export_markdown(self, nodes: List) -> str:
        """Export as Markdown report."""
        lines = ["# MCTS Reasoning Tree\n"]

        # Overview
        lines.append("## Overview\n")
        best_node = max(nodes, key=lambda n: n.value/n.visits if n.visits > 0 else 0)
        lines.append(f"- **Total Nodes**: {len(nodes)}")
        lines.append(f"- **Max Depth**: {max(n.depth for n in nodes)}")
        lines.append(f"- **Best Value**: {best_node.value/best_node.visits:.3f}")
        lines.append(f"- **Question**: {self.session.mcts.original_question}\n")

        # Best solution
        lines.append("## Best Solution\n")
        lines.append("```")
        lines.append(self.session.mcts.solution)
        lines.append("```\n")

        # Top paths
        lines.append("## Top 5 Paths\n")
        sorted_nodes = sorted(nodes, key=lambda n: n.value/n.visits if n.visits > 0 else 0, reverse=True)
        for i, node in enumerate(sorted_nodes[:5], 1):
            avg_val = node.value / node.visits if node.visits > 0 else 0
            lines.append(f"### Path {i} (Value: {avg_val:.3f})")
            lines.append(f"- Depth: {node.depth}")
            lines.append(f"- Visits: {node.visits}")
            lines.append(f"- Action: {node.action_taken if node.action_taken else 'ROOT'}\n")

        # Statistics
        if self.session.mcts.context_manager:
            stats = self.session.mcts.context_manager.get_stats()
            lines.append("## Statistics\n")
            lines.append(f"- Context Summarizations: {stats.get('summarization_count', 0)}")

        return "\n".join(lines)

    def _export_dot(self, nodes: List) -> str:
        """Export as Graphviz DOT format."""
        lines = ["digraph MCTSTree {"]
        lines.append('  rankdir=TB;')
        lines.append('  node [shape=box, style=rounded];')

        # Create node labels
        for i, node in enumerate(nodes):
            avg_val = node.value / node.visits if node.visits > 0 else 0
            label = f"Node {i}\\nVisits: {node.visits}\\nValue: {avg_val:.2f}"
            color = self._get_node_color(avg_val)
            lines.append(f'  node{id(node)} [label="{label}", fillcolor="{color}", style="rounded,filled"];')

        # Create edges
        for node in nodes:
            for child in node.children:
                lines.append(f'  node{id(node)} -> node{id(child)};')

        lines.append("}")
        return "\n".join(lines)

    def _get_node_color(self, value: float) -> str:
        """Get color for node based on value."""
        if value >= 0.8:
            return "#90EE90"  # Light green
        elif value >= 0.6:
            return "#FFFFE0"  # Light yellow
        elif value >= 0.4:
            return "#FFE4B5"  # Moccasin
        else:
            return "#FFB6C1"  # Light pink

    def _export_csv(self, nodes: List) -> str:
        """Export as CSV."""
        lines = ["Index,Depth,Visits,Value,AvgValue,Action,StateLength,Children"]

        for i, node in enumerate(nodes):
            avg_val = node.value / node.visits if node.visits > 0 else 0
            action = str(node.action_taken).replace(',', ';') if node.action_taken else 'ROOT'
            lines.append(
                f"{i},{node.depth},{node.visits},{node.value},{avg_val:.4f},"
                f'"{action}",{len(node.state)},{len(node.children)}'
            )

        return "\n".join(lines)

    def handle_export_tree(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /export-tree command - alias for /export json."""
        return self.handle_export(['json'] + args)

    def handle_verify(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /verify command - verify solution correctness using LLM."""
        if self.session.mcts is None or self.session.mcts.root is None:
            return False, "No search tree available"

        if not self.session.llm_provider:
            return False, "No LLM provider configured"

        # Determine what to verify
        if args and args[0].isdigit():
            # Verify specific node
            node_idx = int(args[0])
            nodes = self.session.mcts.get_all_nodes()
            if node_idx >= len(nodes):
                return False, f"Node index {node_idx} out of range (0-{len(nodes)-1})"
            solution = nodes[node_idx].state
            solution_label = f"Node {node_idx}"
        else:
            # Verify current solution
            if not self.session.mcts.solution:
                return False, "No solution available to verify"
            solution = self.session.mcts.solution
            solution_label = "Current solution"

        # Build verification prompt
        prompt = f"""Verify if this solution is correct and complete.

Solution:
{solution}

Is this solution:
1. Logically correct?
2. Complete (fully answers the question)?
3. Well-reasoned?

Respond with:
VERDICT: CORRECT/INCORRECT/PARTIAL
CONFIDENCE: 0.0-1.0
REASONING: Brief explanation

Your response:
"""

        try:
            # Get LLM verification
            response = self.session.llm_provider.generate(prompt, max_tokens=300)

            # Parse response
            verdict = "UNKNOWN"
            confidence = 0.5
            reasoning = response

            if "VERDICT:" in response:
                verdict_line = [l for l in response.split('\n') if 'VERDICT:' in l][0]
                verdict = verdict_line.split('VERDICT:')[1].strip().split()[0]

            if "CONFIDENCE:" in response:
                conf_line = [l for l in response.split('\n') if 'CONFIDENCE:' in l][0]
                conf_str = conf_line.split('CONFIDENCE:')[1].strip().split()[0]
                try:
                    confidence = float(conf_str)
                except:
                    pass

            if "REASONING:" in response:
                reasoning_parts = response.split('REASONING:')
                if len(reasoning_parts) > 1:
                    reasoning = reasoning_parts[1].strip()

            # Format output
            msg = f"Verification Results for {solution_label}:\n\n"
            msg += f"Verdict: {verdict}\n"
            msg += f"Confidence: {confidence:.2f}\n\n"
            msg += f"Reasoning:\n{reasoning}\n"

            return True, msg

        except Exception as e:
            return False, f"Verification failed: {e}"

    def handle_stats(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /stats command - show detection and summarization stats."""
        if self.session.mcts is None:
            return False, "No MCTS session available"

        msg = "Session Statistics:\n\n"

        # Context management stats
        if self.session.mcts.context_manager:
            ctx_stats = self.session.mcts.context_manager.get_stats()
            msg += "Context Management:\n"
            msg += f"  Summarizations: {ctx_stats['summarization_count']}\n"
            if ctx_stats['last_summarization_tokens']:
                msg += f"  Last summary size: {ctx_stats['last_summarization_tokens']} tokens\n"
            msg += f"  Max context: {ctx_stats['max_context_tokens']} tokens\n"
            msg += f"  Threshold: {int(ctx_stats['summarize_threshold'] * 100)}%\n"
            msg += f"  Target: {int(ctx_stats['summarize_target'] * 100)}%\n"
            msg += f"  Token counting: {ctx_stats['use_token_counting']}\n"
        else:
            msg += "Context Management: Not enabled\n"

        # Solution detection stats
        if self.session.mcts.solution_detector:
            msg += "\nSolution Detection:\n"
            msg += f"  Judgments made: {self.session.mcts.solution_detector._judgment_count}\n"
            msg += f"  Threshold: {self.session.mcts.solution_detector.threshold}\n"

        if self.session.mcts.solution_finalizer:
            fin_stats = self.session.mcts.solution_finalizer.get_stats()
            msg += f"  Solutions finalized: {fin_stats['finalization_count']}\n"

        # Count finalized nodes
        if self.session.mcts.root:
            from mcts_reasoning.solution_detection import is_finalized_solution
            nodes = self.session.mcts.get_all_nodes()
            finalized_count = sum(1 for n in nodes if is_finalized_solution(n.state))
            msg += f"  Finalized nodes: {finalized_count}/{len(nodes)}\n"

        # Tree stats
        if self.session.mcts.root:
            msg += "\nTree Statistics:\n"
            msg += f"  Total nodes: {len(self.session.mcts.get_all_nodes())}\n"
            msg += f"  Max depth: {self.session.mcts.reasoning_depth}\n"
            msg += f"  Best value: {self.session.mcts.best_value:.3f}\n"
            msg += f"  Exploration breadth: {self.session.mcts.exploration_breadth:.2%}\n"

        return True, msg

    def handle_solutions(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /solutions command - list all finalized solutions."""
        if self.session.mcts is None or self.session.mcts.root is None:
            return False, "No search tree available"

        from mcts_reasoning.solution_detection import is_finalized_solution

        nodes = self.session.mcts.get_all_nodes()
        solutions = [
            (i, node) for i, node in enumerate(nodes)
            if is_finalized_solution(node.state)
        ]

        if not solutions:
            return True, "No finalized solutions found in tree"

        msg = f"Finalized Solutions ({len(solutions)} found):\n\n"

        for idx, node in solutions:
            msg += f"Node {idx}:\n"
            msg += f"  Depth: {node.depth}\n"
            msg += f"  Visits: {node.visits}\n"
            msg += f"  Value: {node.value:.3f}\n"
            msg += f"  Avg value: {node.value / node.visits if node.visits > 0 else 0:.3f}\n"

            # Extract first few lines of finalized answer
            lines = node.state.split('\n')
            preview_lines = []
            in_answer = False
            for line in lines:
                if '## Final Answer' in line:
                    in_answer = True
                if in_answer:
                    preview_lines.append(line)
                    if len(preview_lines) >= 5:
                        break

            if preview_lines:
                msg += "  Preview:\n"
                for line in preview_lines:
                    msg += f"    {line}\n"
            msg += "\n"

        msg += f"\nUse 'inspect-full <index>' to see complete solutions"

        return True, msg

    def handle_compare(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /compare command - compare multiple solutions."""
        if self.session.mcts is None or self.session.mcts.root is None:
            return False, "No search tree available"

        if len(args) < 2:
            return False, "Usage: /compare <index1> <index2> [index3...]"

        from mcts_reasoning.solution_detection import is_finalized_solution

        nodes = self.session.mcts.get_all_nodes()

        # Parse indices
        try:
            indices = [int(arg) for arg in args]
        except ValueError:
            return False, "All arguments must be node indices (integers)"

        # Validate indices
        for idx in indices:
            if idx < 0 or idx >= len(nodes):
                return False, f"Invalid node index: {idx} (tree has {len(nodes)} nodes)"

        msg = f"Comparing {len(indices)} nodes:\n\n"

        for idx in indices:
            node = nodes[idx]
            is_solution = is_finalized_solution(node.state)

            msg += f"Node {idx}:\n"
            msg += f"  Depth: {node.depth}\n"
            msg += f"  Visits: {node.visits}\n"
            msg += f"  Value: {node.value:.3f}\n"
            msg += f"  Avg value: {node.value / node.visits if node.visits > 0 else 0:.3f}\n"
            msg += f"  Is solution: {'Yes [FINALIZED]' if is_solution else 'No'}\n"
            msg += f"  Action: {node.action_taken if node.action_taken else 'ROOT'}\n"

            # Show preview
            preview = node.state[-300:] if len(node.state) > 300 else node.state
            msg += f"  State preview:\n"
            for line in preview.split('\n')[:5]:
                msg += f"    {line}\n"
            msg += "\n"

        return True, msg

    def handle_config(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /config command - configure features dynamically."""
        if not args:
            # Show current configuration
            msg = "Current Configuration:\n\n"
            if self.session.mcts:
                msg += f"  Compositional actions: {self.session.mcts.use_compositional}\n"
                msg += f"  Solution detection: {'Enabled' if self.session.mcts.solution_detector else 'Disabled'}\n"
                msg += f"  Auto-finalize: {self.session.mcts.auto_finalize_solutions if self.session.mcts.solution_detector else 'N/A'}\n"
                msg += f"  Context management: {'Enabled' if self.session.mcts.context_manager else 'Disabled'}\n"
                msg += f"  Terminal detection: {'LLM' if self.session.mcts.terminal_check_with_llm else 'Pattern'}\n"
            else:
                msg += "  No MCTS session active\n"
            return True, msg

        if len(args) < 2:
            return False, "Usage: /config <feature> <on|off>\n  Features: solution-detection, auto-summarize, terminal-llm"

        feature = args[0].lower()
        value_str = args[1].lower()

        if value_str not in ['on', 'off', 'true', 'false', 'enabled', 'disabled']:
            return False, "Value must be one of: on, off, true, false, enabled, disabled"

        value = value_str in ['on', 'true', 'enabled']

        if self.session.mcts is None:
            return False, "No MCTS session available. Start a session with /ask first"

        # Apply configuration
        if feature in ['solution-detection', 'solution', 'solutions']:
            if value and self.session.provider:
                self.session.mcts.with_solution_detection(enabled=True)
                return True, "✓ Solution detection enabled"
            else:
                self.session.mcts.solution_detector = None
                self.session.mcts.solution_finalizer = None
                return True, "✗ Solution detection disabled"

        elif feature in ['auto-summarize', 'context', 'summarize']:
            if value:
                self.session.mcts.with_context_config(auto_configure=True)
                return True, "✓ Context management enabled"
            else:
                self.session.mcts.context_manager = None
                return True, "✗ Context management disabled"

        elif feature in ['terminal-llm', 'terminal']:
            self.session.mcts.terminal_check_with_llm = value
            return True, f"✓ Terminal detection: {'LLM-based' if value else 'Pattern-based'}"

        else:
            return False, f"Unknown feature: {feature}\n  Available: solution-detection, auto-summarize, terminal-llm"

    def handle_help(self, args: List[str]) -> Tuple[bool, str]:
        """Handle help command."""
        help_text = """
Available commands:

Session Management:
  ask <question>         - Start a new reasoning session
  search <n>             - Run N simulations (default: 20)
  continue <n>           - Continue search (alias for search)
  solution               - Show best solution
  save [filename]        - Save session
  load <filename>        - Load session
  status                 - Show current status

Configuration:
  model [provider] [model] [key=value...]  - Switch LLM or show current
  models                                   - List available models
  model-info [model]                       - Show model information
  probe <url> [provider]                   - Probe endpoint for models
  temperature <value>                      - Set temperature (0.0-2.0)
  exploration <value>                      - Set exploration constant

Analysis:
  tree                   - Visualize search tree
  sample <n>             - Sample N diverse paths
  consistency [n]        - Check solution consistency

Tree Diagnostics:
  nodes                  - List all nodes in tree with indices
  inspect <index>        - Show detailed info about a node (last 500 chars)
  inspect-full <index>   - Show FULL state (no truncation)
  show-prompt <index>    - Show exact prompt LLM saw for this node
  path <index>           - Show reasoning path to a node
  export <format> <file> - Export tree (formats: json, markdown, dot, csv)
  export-tree <file>     - Export tree to JSON (alias for export json)
  verify [index]         - Verify solution correctness using LLM

Statistics & Solutions:
  stats                  - Show context management and solution detection stats
  solutions              - List all finalized solutions
  compare <i1> <i2> ...  - Compare multiple nodes side-by-side
  config [feature] [on|off]  - Configure features or show current config

MCP Tools:
  mcp-enable             - Enable MCP integration
  mcp-connect <name> <type>  - Connect to MCP server
  mcp-list               - List connected servers
  mcp-tools              - Show available tools

Other:
  help                   - Show this help
  exit                   - Exit TUI

Examples:
  model                                           # Show current model
  model openai gpt-4                              # Switch to OpenAI GPT-4
  model ollama llama2 base_url=http://192.168.0.225:11434
  probe http://192.168.0.225:11434                # Check available models
  probe http://192.168.0.225:11434 ollama         # Probe with provider type
  models                                          # List available models
  model-info                                      # Show current model info
  ask What is the sum of primes less than 20?
  search 50
  sample 5
  consistency 20
  nodes                                           # List all nodes
  inspect 5                                       # Inspect node 5 (preview)
  inspect-full 5                                  # Show full state of node 5
  show-prompt 5                                   # See what LLM saw
  path 5                                          # Show path to node 5
  export json my_tree.json                        # Export tree as JSON
  export markdown report.md                       # Export as markdown report
  export dot graph.dot                            # Export as Graphviz DOT
  export csv data.csv                             # Export as CSV
  verify                                          # Verify current solution
  verify 5                                        # Verify specific node
  stats                                           # Show session statistics
  solutions                                       # List finalized solutions
  compare 5 12 18                                 # Compare three nodes
  config                                          # Show current configuration
  config solution-detection on                    # Enable solution detection
  config auto-summarize off                       # Disable context summarization

Note: Slash prefix is optional (both 'ask' and '/ask' work).
"""
        return True, help_text

    def handle_exit(self, args: List[str]) -> Tuple[bool, str]:
        """Handle /exit command."""
        return True, "EXIT"


__all__ = ['Command', 'CommandParser', 'CommandHandler']
