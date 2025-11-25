"""
Main shell REPL with Unix-style command execution.
"""

import sys
import os
from pathlib import Path
from typing import Optional, Dict, Any, List

# Prompt toolkit for enhanced REPL
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.styles import Style
    _has_prompt_toolkit = True
except ImportError:
    _has_prompt_toolkit = False
    PromptSession = None

from .parser import ShellParser, split_multiple_commands
from .command_base import CommandBase, CommandContext, CommandRegistry, CommandError, HelpCommand, EchoCommand, ExitCommand
from .executor import PipelineExecutor, display_stream
from .streams import Stream, TextStream, StatsStream

# Import all command modules
from .reasoning_commands import (
    AskCommand, SearchCommand, ExploreCommand,
    SampleCommand, BestCommand, WorstCommand, RandomCommand
)
from .filter_commands import (
    FilterCommand, SortCommand, HeadCommand, TailCommand,
    GrepCommand, UniqueCommand, CountCommand
)
from .io_commands import (
    LoadCommand, SaveCommand, ExportCommand,
    FormatCommand, CatCommand
)
from .analysis_commands import (
    StatsCommand, TreeCommand, VerifyCommand,
    ConsistencyCommand, DiffCommand, ExplainCommand
)

# Config
from ..config import Config


class Shell:
    """
    Unix-style shell for MCTS reasoning.

    Features:
    - Piping: cmd1 | cmd2 | cmd3
    - I/O redirection: > file, >> file, < file
    - Command history and completion
    - Variable expansion
    - Composable commands
    """

    def __init__(self, config: Optional[Config] = None, use_rich: bool = True):
        """
        Initialize shell.

        Args:
            config: Configuration object (or None to create default)
            use_rich: Use rich formatting for output
        """
        self.config = config or Config()
        self.use_rich = use_rich
        self.parser = ShellParser()
        self.registry = CommandRegistry()
        self.running = False

        # Shell state
        self.llm_provider = None
        self.rag_store = None
        self.variables: Dict[str, Any] = {}

        # Set up commands
        self._register_commands()

        # Command context
        self.context = CommandContext(
            config=self.config.load() if self.config else {},
            llm_provider=self.llm_provider,
            rag_store=self.rag_store,
            env=self.variables
        )

        # Executor
        self.executor = PipelineExecutor(self.registry, self.context)

        # Set up prompt toolkit if available
        self.prompt_session = None
        if _has_prompt_toolkit:
            self._setup_prompt()

    def _register_commands(self):
        """Register all available commands."""
        # Core commands
        self.registry.register(HelpCommand(self.registry), aliases=['?'])
        self.registry.register(EchoCommand())
        self.registry.register(ExitCommand(), aliases=['quit', 'q'])

        # Reasoning commands
        self.registry.register(AskCommand())
        self.registry.register(SearchCommand())
        self.registry.register(ExploreCommand())
        self.registry.register(SampleCommand())
        self.registry.register(BestCommand())
        self.registry.register(WorstCommand())
        self.registry.register(RandomCommand())

        # Filter commands
        self.registry.register(FilterCommand())
        self.registry.register(SortCommand())
        self.registry.register(HeadCommand())
        self.registry.register(TailCommand())
        self.registry.register(GrepCommand())
        self.registry.register(UniqueCommand())
        self.registry.register(CountCommand(), aliases=['wc'])

        # I/O commands
        self.registry.register(LoadCommand())
        self.registry.register(SaveCommand())
        self.registry.register(ExportCommand())
        self.registry.register(FormatCommand())
        self.registry.register(CatCommand())

        # Analysis commands
        self.registry.register(StatsCommand())
        self.registry.register(TreeCommand())
        self.registry.register(VerifyCommand())
        self.registry.register(ConsistencyCommand())
        self.registry.register(DiffCommand())
        self.registry.register(ExplainCommand())

        # Config commands
        self.registry.register(SetCommand(self))
        self.registry.register(GetCommand(self))
        self.registry.register(UseCommand(self))

    def _setup_prompt(self):
        """Set up prompt_toolkit with history and completion."""
        # History file
        history_dir = Path.home() / '.mcts-reasoning'
        history_dir.mkdir(exist_ok=True)
        history_file = history_dir / 'shell_history'

        # Command completer
        command_names = self.registry.command_names()
        completer = WordCompleter(
            command_names,
            ignore_case=True,
            sentence=True  # Complete within sentences
        )

        # Style
        style = Style.from_dict({
            'prompt': '#00aa00 bold',
        })

        # Create session
        self.prompt_session = PromptSession(
            history=FileHistory(str(history_file)),
            completer=completer,
            auto_suggest=AutoSuggestFromHistory(),
            style=style,
            enable_history_search=True
        )

    def start(self):
        """Start the shell REPL."""
        self.running = True

        # Show welcome message
        self._show_welcome()

        # Initialize LLM provider if configured
        self._init_llm()

        # Main REPL loop
        while self.running:
            try:
                # Get prompt string
                prompt_str = self._get_prompt()

                # Read input
                if self.prompt_session:
                    line = self.prompt_session.prompt(prompt_str)
                else:
                    line = input(prompt_str)

                # Skip empty lines
                if not line or line.strip().startswith('#'):
                    continue

                # Handle multiple commands (separated by ;)
                commands = split_multiple_commands(line)

                for cmd_line in commands:
                    self._execute_line(cmd_line)

            except KeyboardInterrupt:
                print("\n(Use 'exit' or Ctrl+D to quit)")
                continue

            except EOFError:
                print("\nGoodbye!")
                break

            except SystemExit:
                print("Goodbye!")
                break

            except Exception as e:
                print(f"Error: {e}", file=sys.stderr)

    def _execute_line(self, line: str):
        """Execute a single command line."""
        # Parse
        pipeline = self.parser.parse(line)

        if not pipeline:
            return

        # Update context
        self.context.llm_provider = self.llm_provider
        self.context.rag_store = self.rag_store
        self.context.env = self.variables
        self.context.config = self.config.load() if self.config else {}

        # Execute
        try:
            output_stream = self.executor.execute(pipeline)

            # Display output (unless redirected)
            if output_stream:
                display_stream(output_stream, self.use_rich)

        except CommandError as e:
            print(f"Error: {e}", file=sys.stderr)

        except SystemExit:
            raise

        except Exception as e:
            print(f"Unexpected error: {e}", file=sys.stderr)
            if os.getenv('DEBUG'):
                import traceback
                traceback.print_exc()

    def _init_llm(self):
        """Initialize LLM provider from config."""
        try:
            from ..compositional.providers import get_llm

            provider_name = self.config.get('shell.provider', default='mock')
            model = self.config.get('shell.model', default=None)

            if provider_name and provider_name != 'mock':
                self.llm_provider = get_llm(provider_name, model=model)

        except Exception as e:
            print(f"Warning: Could not initialize LLM provider: {e}", file=sys.stderr)

    def _get_prompt(self) -> str:
        """Get prompt string."""
        # Show current model if configured
        provider = self.config.get('shell.provider', default='mock')

        if provider == 'mock':
            return "mcts> "
        else:
            model = self.config.get('shell.model', default='')
            model_short = model.split('/')[-1] if model else provider
            return f"mcts({model_short})> "

    def _show_welcome(self):
        """Show welcome message."""
        print("MCTS-Reasoning Shell")
        print("A Unix-style composable shell for LLM reasoning")
        print("")
        print("Type 'help' for available commands, 'exit' to quit.")
        print("")

        # Show current config
        provider = self.config.get('shell.provider', default='mock')
        if provider != 'mock':
            model = self.config.get('shell.model', default='')
            print(f"Using {provider}" + (f" ({model})" if model else ""))
        else:
            print("Using mock provider (no LLM calls)")
            print("Set a provider with: set provider openai")

        print("")


class SetCommand(CommandBase):
    """Set configuration variable."""

    def __init__(self, shell: Shell):
        super().__init__()
        self.shell = shell

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if len(args) < 2:
            raise CommandError("Usage: set <key> <value>")

        key = args[0]
        value = ' '.join(args[1:])

        # Special handling for some keys
        if key == 'provider':
            self.shell.config.set('shell.provider', value)
            # Reinitialize LLM
            self.shell._init_llm()
            context.llm_provider = self.shell.llm_provider
            return TextStream([f"Provider set to: {value}"])

        elif key == 'model':
            self.shell.config.set('shell.model', value)
            self.shell._init_llm()
            context.llm_provider = self.shell.llm_provider
            return TextStream([f"Model set to: {value}"])

        elif key == 'exploration':
            try:
                float_val = float(value)
                self.shell.config.set('shell.exploration', float_val)
                context.config['exploration'] = float_val
                return TextStream([f"Exploration constant set to: {float_val}"])
            except ValueError:
                raise CommandError(f"Invalid number: {value}")

        elif key == 'temperature':
            try:
                float_val = float(value)
                self.shell.config.set('shell.temperature', float_val)
                context.config['temperature'] = float_val
                return TextStream([f"Temperature set to: {float_val}"])
            except ValueError:
                raise CommandError(f"Invalid number: {value}")

        else:
            # Generic config setting
            self.shell.config.set(f'shell.{key}', value)
            context.config[key] = value
            return TextStream([f"{key} = {value}"])

    def get_help(self) -> str:
        return """set - Set configuration variable

Usage:
  set <key> <value>           Set config value

Keys:
  provider                    LLM provider (openai, anthropic, ollama, mock)
  model                       Model name
  exploration                 MCTS exploration constant (default: 1.414)
  temperature                 Sampling temperature (default: 1.0)

Examples:
  set provider openai
  set model gpt-4
  set exploration 2.0
  set temperature 0.7
"""


class GetCommand(CommandBase):
    """Get configuration variable."""

    def __init__(self, shell: Shell):
        super().__init__()
        self.shell = shell

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not args:
            # Show all config
            config_dict = self.shell.config.get('shell', default={})
            if not config_dict:
                config_dict = self.shell.config.load().get('shell', {})
            return StatsStream(config_dict if config_dict else {})

        key = args[0]
        value = self.shell.config.get(f'shell.{key}', default='<not set>')

        return TextStream([f"{key} = {value}"])

    def get_help(self) -> str:
        return """get - Get configuration variable

Usage:
  get [key]                   Get config value (or all if no key)

Examples:
  get provider
  get
"""


class UseCommand(CommandBase):
    """Load RAG store or other resources."""

    def __init__(self, shell: Shell):
        super().__init__()
        self.shell = shell

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not args:
            raise CommandError("Usage: use <resource>")

        resource_type = args[0]

        if resource_type == 'rag':
            # Load RAG store
            if len(args) < 2:
                raise CommandError("Usage: use rag <store_name>")

            store_name = args[1]

            try:
                from ..compositional.rag import load_rag_store
                self.shell.rag_store = load_rag_store(store_name)
                context.rag_store = self.shell.rag_store
                return TextStream([f"Loaded RAG store: {store_name}"])

            except Exception as e:
                raise CommandError(f"Error loading RAG store: {e}")

        else:
            raise CommandError(f"Unknown resource type: {resource_type}")

    def get_help(self) -> str:
        return """use - Load resources

Usage:
  use rag <name>              Load RAG store

Examples:
  use rag math
  use rag code

Loads RAG stores or other resources for reasoning guidance.
"""
