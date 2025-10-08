"""
Enhanced prompt system for TUI using prompt_toolkit.

Provides:
- Persistent command history
- Tab completion for slash commands
- Syntax highlighting
- Vi/Emacs key bindings
"""

from pathlib import Path
from typing import List, Optional, Iterable

from prompt_toolkit import PromptSession
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.history import FileHistory
from prompt_toolkit.lexers import Lexer
from prompt_toolkit.styles import Style
from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.document import Document


class CommandCompleter(Completer):
    """Tab completion for slash commands."""

    # All available commands
    COMMANDS = [
        # Session management
        '/ask',
        '/search',
        '/continue',
        '/solution',
        '/save',
        '/load',
        '/status',

        # Configuration
        '/model',
        '/models',
        '/model-info',
        '/temperature',
        '/temp',
        '/exploration',

        # Analysis
        '/tree',
        '/sample',
        '/consistency',

        # MCP
        '/mcp-enable',
        '/mcp-connect',
        '/mcp-list',
        '/mcp-tools',

        # Other
        '/help',
        '/exit',
        '/quit',
    ]

    # Command descriptions for meta display
    DESCRIPTIONS = {
        '/ask': 'Start a new reasoning session',
        '/search': 'Run N simulations',
        '/continue': 'Continue search (alias for /search)',
        '/solution': 'Show best solution',
        '/save': 'Save session',
        '/load': 'Load session',
        '/status': 'Show current status',
        '/model': 'Switch LLM or show current',
        '/models': 'List available models',
        '/model-info': 'Show model information',
        '/temperature': 'Set temperature',
        '/temp': 'Set temperature (alias)',
        '/exploration': 'Set exploration constant',
        '/tree': 'Visualize search tree',
        '/sample': 'Sample N diverse paths',
        '/consistency': 'Check solution consistency',
        '/mcp-enable': 'Enable MCP integration',
        '/mcp-connect': 'Connect to MCP server',
        '/mcp-list': 'List connected servers',
        '/mcp-tools': 'Show available tools',
        '/help': 'Show help',
        '/exit': 'Exit TUI',
        '/quit': 'Exit TUI (alias)',
    }

    # Provider names for /model completion
    PROVIDERS = ['openai', 'anthropic', 'ollama', 'mock']

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        """Generate completions for the current input."""
        text = document.text_before_cursor

        # Only complete if we're at the start or after whitespace
        if not text or text[-1].isspace():
            return

        # Split into words
        words = text.split()
        if not words:
            return

        # First word - complete command names
        if len(words) == 1 and text.startswith('/'):
            prefix = words[0].lower()
            for cmd in self.COMMANDS:
                if cmd.startswith(prefix):
                    yield Completion(
                        cmd,
                        start_position=-len(prefix),
                        display=cmd,
                        display_meta=self.DESCRIPTIONS.get(cmd, '')
                    )

        # Second word for /model - complete provider names
        elif len(words) == 2 and words[0] in ['/model']:
            prefix = words[1].lower()
            for provider in self.PROVIDERS:
                if provider.startswith(prefix):
                    yield Completion(
                        provider,
                        start_position=-len(prefix),
                        display=provider,
                        display_meta=f'{provider.title()} LLM provider'
                    )


class CommandLexer(Lexer):
    """Syntax highlighting for commands."""

    def lex_document(self, document: Document):
        """Apply syntax highlighting to the input."""
        def get_line_tokens(line_number):
            line = document.lines[line_number]

            # Tokenize the line
            tokens = []

            if line.startswith('/'):
                # Command - highlight in cyan
                parts = line.split(None, 1)
                command = parts[0]

                # Command name in cyan
                tokens.append(('class:command', command))

                # Arguments in default color
                if len(parts) > 1:
                    tokens.append(('', ' '))
                    tokens.append(('class:argument', parts[1]))
            else:
                # Regular text
                tokens.append(('', line))

            return tokens

        return get_line_tokens


def create_prompt_style():
    """Create style for the prompt."""
    return Style.from_dict({
        # Prompt symbol
        'prompt': '#00d7ff bold',  # Cyan

        # Command highlighting
        'command': '#00d7ff',  # Cyan
        'argument': '#ffffff',  # White

        # Completion menu
        'completion-menu': 'bg:#262626 #ffffff',
        'completion-menu.completion': 'bg:#262626 #ffffff',
        'completion-menu.completion.current': 'bg:#005f87 #ffffff bold',
        'completion-menu.meta.completion': 'bg:#262626 #888888',
        'completion-menu.meta.completion.current': 'bg:#005f87 #ffffff',
    })


def create_prompt_session(history_file: Optional[Path] = None) -> PromptSession:
    """
    Create a configured prompt session.

    Args:
        history_file: Path to history file. Defaults to ~/.mcts-reasoning/history

    Returns:
        Configured PromptSession
    """
    # Default history file location
    if history_file is None:
        history_dir = Path.home() / ".mcts-reasoning"
        history_dir.mkdir(parents=True, exist_ok=True)
        history_file = history_dir / "history"

    # Create session
    session = PromptSession(
        history=FileHistory(str(history_file)),
        completer=CommandCompleter(),
        lexer=CommandLexer(),
        style=create_prompt_style(),
        enable_history_search=True,  # Ctrl+R for history search
        complete_while_typing=True,  # Show completions as you type
        mouse_support=False,  # Disable mouse to avoid conflicts
        vi_mode=False,  # Use emacs mode by default
    )

    return session


def get_prompt_message(provider: str, model: str, use_rich: bool = True) -> FormattedText:
    """
    Create the prompt message with current model info.

    Args:
        provider: Current provider name
        model: Current model name
        use_rich: Whether to use rich formatting

    Returns:
        Formatted prompt message
    """
    if use_rich:
        return FormattedText([
            ('class:prompt', '\n> '),
        ])
    else:
        return FormattedText([
            ('', '\n> '),
        ])


class EnhancedPrompt:
    """Enhanced prompt manager with history and completion."""

    def __init__(self, history_file: Optional[Path] = None):
        """
        Initialize enhanced prompt.

        Args:
            history_file: Path to history file
        """
        self.session = create_prompt_session(history_file)
        self.provider = "mock"
        self.model = "default"

    def update_context(self, provider: str, model: str):
        """Update the current provider/model context."""
        self.provider = provider
        self.model = model

    def prompt(self, use_rich: bool = True) -> str:
        """
        Show prompt and get user input.

        Args:
            use_rich: Whether to use rich formatting

        Returns:
            User input string
        """
        message = get_prompt_message(self.provider, self.model, use_rich)

        try:
            text = self.session.prompt(message)
            return text.strip()
        except (EOFError, KeyboardInterrupt):
            # User pressed Ctrl+D or Ctrl+C
            return '/exit'

    def clear_history(self):
        """Clear the history file."""
        if hasattr(self.session.history, 'clear'):
            self.session.history.clear()


# Convenience function for creating prompt
def create_enhanced_prompt(history_file: Optional[Path] = None) -> EnhancedPrompt:
    """
    Create an enhanced prompt instance.

    Args:
        history_file: Optional path to history file

    Returns:
        EnhancedPrompt instance
    """
    return EnhancedPrompt(history_file)
