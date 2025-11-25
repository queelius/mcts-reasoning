"""
Pipeline executor - runs parsed commands with proper stream handling.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any

from .parser import Pipeline, Command, Redirection, RedirectType
from .command_base import CommandContext, CommandError, CommandRegistry
from .streams import Stream, TextStream, stream_from_file


class PipelineExecutor:
    """
    Execute command pipelines with Unix-style semantics.

    Handles:
    - Pipe connections between commands
    - I/O redirection
    - Error propagation
    - Background jobs (basic support)
    """

    def __init__(self, registry: CommandRegistry, context: CommandContext):
        self.registry = registry
        self.context = context

    def execute(self, pipeline: Pipeline) -> Optional[Stream]:
        """
        Execute a pipeline of commands.

        Args:
            pipeline: Parsed pipeline

        Returns:
            Output stream from last command (or None if redirected)

        Raises:
            CommandError: If any command fails
        """
        # Handle input redirection
        input_stream = None
        if pipeline.input_redirect:
            input_stream = self._load_redirect_input(pipeline.input_redirect)

        # Execute pipeline
        current_stream = input_stream

        for i, cmd in enumerate(pipeline.commands):
            # Get command handler
            handler = self.registry.get(cmd.name)

            if not handler:
                raise CommandError(f"Unknown command: {cmd.name}")

            # Validate arguments
            try:
                handler.validate_args(cmd.args, cmd.kwargs)
            except CommandError as e:
                raise CommandError(f"In command '{cmd.name}': {e}")

            # Check if command requires input
            if handler.requires_input() and current_stream is None:
                raise CommandError(
                    f"Command '{cmd.name}' requires input from pipe or file. "
                    f"Use: <previous_command> | {cmd.name}"
                )

            # Set up context with current stream
            exec_context = CommandContext(
                config=self.context.config.copy(),
                llm_provider=self.context.llm_provider,
                rag_store=self.context.rag_store,
                input_stream=current_stream,
                env=self.context.env.copy(),
                capture_output=self.context.capture_output
            )

            # Execute command
            try:
                output_stream = handler.execute(cmd.args, cmd.kwargs, exec_context)
                current_stream = output_stream

            except SystemExit:
                # Exit command - propagate
                raise

            except CommandError as e:
                raise CommandError(f"In command '{cmd.name}': {e}")

            except Exception as e:
                raise CommandError(f"Error in command '{cmd.name}': {e}")

        # Handle output redirection
        if pipeline.output_redirect and current_stream:
            self._save_redirect_output(pipeline.output_redirect, current_stream)
            return None  # Don't display when redirected

        return current_stream

    def _load_redirect_input(self, redirect: Redirection) -> Stream:
        """Load input from file for < redirection."""
        filepath = redirect.target

        try:
            return stream_from_file(filepath)
        except FileNotFoundError:
            raise CommandError(f"Input file not found: {filepath}")
        except Exception as e:
            raise CommandError(f"Error loading input file: {e}")

    def _save_redirect_output(self, redirect: Redirection, stream: Stream):
        """Save output to file for > or >> redirection."""
        filepath = redirect.target
        path = Path(filepath)

        # Create parent directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Determine write mode
        if redirect.type == RedirectType.APPEND:
            mode = 'a'
        else:
            mode = 'w'

        # Get output content
        # Try text first, fall back to JSON
        try:
            content = stream.to_text()
        except:
            content = stream.to_json()

        # Write to file
        try:
            with open(path, mode) as f:
                f.write(content)
                if mode == 'a':
                    f.write('\n')  # Add separator for appends

        except Exception as e:
            raise CommandError(f"Error writing to file: {e}")


def format_output(stream: Optional[Stream], use_rich: bool = False) -> str:
    """
    Format stream for terminal display.

    Args:
        stream: Stream to format
        use_rich: Use rich formatting if available

    Returns:
        Formatted string for display
    """
    if stream is None:
        return ""

    # Default: use text representation
    return stream.to_text()


def display_stream(stream: Optional[Stream], use_rich: bool = False):
    """
    Display stream to terminal.

    Args:
        stream: Stream to display
        use_rich: Use rich formatting if available
    """
    if stream is None:
        return

    output = format_output(stream, use_rich)

    if output:
        print(output)
