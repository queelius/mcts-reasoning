"""
Base command system for composable shell commands.

Each command is a filter that transforms streams.
"""

import sys
from typing import Optional, Dict, Any, Callable, List
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .streams import Stream, TextStream


@dataclass
class CommandContext:
    """
    Context passed to commands during execution.

    Contains:
    - Shell state (config, LLM provider, etc.)
    - Input stream (from pipe or file)
    - Environment variables
    """
    # Shell state
    config: Dict[str, Any] = field(default_factory=dict)
    llm_provider: Optional[Any] = None
    rag_store: Optional[Any] = None

    # Current input stream
    input_stream: Optional[Stream] = None

    # Environment (for variable expansion)
    env: Dict[str, Any] = field(default_factory=dict)

    # Output capture (for variable assignment)
    capture_output: bool = False


class CommandError(Exception):
    """Base exception for command errors."""
    pass


class CommandBase(ABC):
    """
    Base class for shell commands.

    Commands are composable filters:
    - Consume an input stream (or None)
    - Produce an output stream
    - Can have side effects (e.g., training models, saving files)
    """

    def __init__(self):
        self.name = self.__class__.__name__.lower().replace('command', '')
        self.description = self.__doc__.strip() if self.__doc__ else ""

    @abstractmethod
    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        """
        Execute the command.

        Args:
            args: Positional arguments
            kwargs: Keyword arguments (from --flags)
            context: Execution context with input stream and shell state

        Returns:
            Output stream

        Raises:
            CommandError: If command fails
        """
        pass

    def get_help(self) -> str:
        """
        Get help text for this command.

        Override to provide detailed help.
        """
        return f"{self.name}: {self.description}"

    def validate_args(self, args: List[str], kwargs: Dict[str, Any]):
        """
        Validate command arguments.

        Raise CommandError if arguments are invalid.
        Override in subclasses for specific validation.
        """
        pass

    def requires_input(self) -> bool:
        """
        Whether this command requires an input stream.

        Override to return True for filter-only commands.
        """
        return False

    def produces_output(self) -> bool:
        """
        Whether this command produces output.

        Override to return False for side-effect only commands.
        """
        return True


class CommandRegistry:
    """
    Registry of available commands.

    Commands register themselves and can be looked up by name.
    """

    def __init__(self):
        self._commands: Dict[str, CommandBase] = {}
        self._aliases: Dict[str, str] = {}

    def register(self, command: CommandBase, aliases: Optional[List[str]] = None):
        """Register a command."""
        self._commands[command.name] = command

        if aliases:
            for alias in aliases:
                self._aliases[alias] = command.name

    def get(self, name: str) -> Optional[CommandBase]:
        """Get command by name or alias."""
        # Check if it's an alias
        if name in self._aliases:
            name = self._aliases[name]

        return self._commands.get(name)

    def all_commands(self) -> List[CommandBase]:
        """Get all registered commands."""
        return list(self._commands.values())

    def command_names(self) -> List[str]:
        """Get all command names (including aliases)."""
        names = list(self._commands.keys())
        names.extend(self._aliases.keys())
        return sorted(names)

    def get_help_text(self) -> str:
        """Get help text for all commands."""
        lines = ["Available commands:"]

        # Group commands by category
        categories = {}
        for cmd in self._commands.values():
            # Extract category from docstring if present
            category = "General"
            if cmd.description:
                if any(word in cmd.description.lower() for word in ['search', 'reasoning', 'mcts']):
                    category = "Reasoning"
                elif any(word in cmd.description.lower() for word in ['sample', 'path', 'solution']):
                    category = "Sampling"
                elif any(word in cmd.description.lower() for word in ['load', 'save', 'export', 'import']):
                    category = "I/O"
                elif any(word in cmd.description.lower() for word in ['filter', 'sort', 'head', 'tail']):
                    category = "Filtering"
                elif any(word in cmd.description.lower() for word in ['stats', 'tree', 'analyze']):
                    category = "Analysis"

            if category not in categories:
                categories[category] = []
            categories[category].append(cmd)

        # Format by category
        for category in sorted(categories.keys()):
            lines.append(f"\n{category}:")
            for cmd in sorted(categories[category], key=lambda c: c.name):
                lines.append(f"  {cmd.name:15s} {cmd.description}")

        return '\n'.join(lines)


# Decorator for easy command registration
def command(name: Optional[str] = None, aliases: Optional[List[str]] = None):
    """
    Decorator to register a command.

    Usage:
        @command(aliases=['ls'])
        class ListCommand(CommandBase):
            ...
    """
    def decorator(cls):
        # Instantiate and register
        instance = cls()
        if name:
            instance.name = name

        # Auto-register if registry is available
        # (will be set up in shell initialization)
        if hasattr(cls, '_registry'):
            cls._registry.register(instance, aliases)

        return cls

    return decorator


class HelpCommand(CommandBase):
    """Show help for commands."""

    def __init__(self, registry: CommandRegistry):
        super().__init__()
        self.registry = registry

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if args:
            # Help for specific command
            cmd_name = args[0]
            cmd = self.registry.get(cmd_name)

            if cmd:
                help_text = cmd.get_help()
            else:
                help_text = f"Unknown command: {cmd_name}"
        else:
            # General help
            help_text = self.registry.get_help_text()

        return TextStream([help_text])


class EchoCommand(CommandBase):
    """Print text to output."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        text = ' '.join(str(a) for a in args)

        # Expand variables
        for var_name, var_value in context.env.items():
            text = text.replace(f'${var_name}', str(var_value))

        return TextStream([text])


class ExitCommand(CommandBase):
    """Exit the shell."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        # Signal exit by raising a special exception
        raise SystemExit(0)

    def produces_output(self) -> bool:
        return False
