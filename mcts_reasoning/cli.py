#!/usr/bin/env python3
"""
MCTS-Reasoning CLI Entry Point

Non-interactive command-line interface for MCTS reasoning.

Examples:
    mcts ask "What is 2+2?" --search 50
    mcts verify
    mcts export json output.json
    mcts solution
"""

import sys
import argparse
from pathlib import Path
from typing import List, Optional

from mcts_reasoning.tui.session import SessionState
from mcts_reasoning.tui.commands import CommandParser, CommandHandler


class MCTSCLIError(Exception):
    """CLI-specific error."""
    pass


class MCTSCLI:
    """
    Non-interactive CLI for MCTS-Reasoning.

    Executes a single command and exits.
    """

    def __init__(self, session_file: Optional[str] = None):
        """
        Initialize CLI.

        Args:
            session_file: Optional session file to load
        """
        self.session = SessionState()
        self.parser = CommandParser()
        self.handler = CommandHandler(self.session)

        # Load session if specified
        if session_file:
            self._load_session(session_file)

    def _load_session(self, filename: str):
        """Load a saved session."""
        success, message = self.handler.execute(
            self.parser.parse(f"load {filename}")
        )
        if not success:
            raise MCTSCLIError(f"Failed to load session: {message}")

    def execute_command(self, command_str: str) -> str:
        """
        Execute a single command.

        Args:
            command_str: Command string to execute

        Returns:
            Command output message

        Raises:
            MCTSCLIError: If command fails
        """
        # Parse command
        command = self.parser.parse(command_str)

        if command is None:
            raise MCTSCLIError(f"Invalid command: {command_str}")

        # Execute
        success, message = self.handler.execute(command)

        if not success:
            raise MCTSCLIError(message)

        return message


def build_command_from_args(args: argparse.Namespace) -> str:
    """
    Build command string from parsed arguments.

    Args:
        args: Parsed command-line arguments

    Returns:
        Command string for CommandParser
    """
    cmd_parts = [args.command]

    # Handle subcommand arguments
    if args.command == 'ask':
        if not args.question:
            raise MCTSCLIError("ask command requires a question")
        cmd_parts.append(args.question)
    elif args.command == 'search':
        cmd_parts.append(str(args.simulations or 20))
    elif args.command == 'sample':
        cmd_parts.append(str(args.count or 5))
    elif args.command == 'consistency':
        if args.trials:
            cmd_parts.append(str(args.trials))
    elif args.command == 'export':
        if not args.format:
            raise MCTSCLIError("export command requires format (json, markdown, dot, csv)")
        cmd_parts.append(args.format)
        if args.output:
            cmd_parts.append(args.output)
    elif args.command == 'verify':
        if args.node_index is not None:
            cmd_parts.append(str(args.node_index))
    elif args.command == 'inspect':
        if args.node_index is None:
            raise MCTSCLIError("inspect command requires node index")
        cmd_parts.append(str(args.node_index))
    elif args.command == 'inspect-full':
        if args.node_index is None:
            raise MCTSCLIError("inspect-full command requires node index")
        cmd_parts.append(str(args.node_index))
    elif args.command == 'path':
        if args.node_index is None:
            raise MCTSCLIError("path command requires node index")
        cmd_parts.append(str(args.node_index))
    elif args.command == 'compare':
        if not args.indices:
            raise MCTSCLIError("compare command requires node indices")
        cmd_parts.extend(map(str, args.indices))
    elif args.command == 'save':
        if args.filename:
            cmd_parts.append(args.filename)
    elif args.command == 'load':
        if not args.filename:
            raise MCTSCLIError("load command requires filename")
        cmd_parts.append(args.filename)
    elif args.command == 'model':
        if args.provider:
            cmd_parts.append(args.provider)
        if args.model:
            cmd_parts.append(args.model)
    elif args.command == 'temperature':
        if args.value is None:
            raise MCTSCLIError("temperature command requires value")
        cmd_parts.append(str(args.value))
    elif args.command == 'exploration':
        if args.value is None:
            raise MCTSCLIError("exploration command requires value")
        cmd_parts.append(str(args.value))

    # Add extra arguments if provided
    if hasattr(args, 'extra_args') and args.extra_args:
        cmd_parts.extend(args.extra_args)

    return ' '.join(cmd_parts)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for CLI."""
    parser = argparse.ArgumentParser(
        description='MCTS-Reasoning CLI - Non-interactive command execution',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s ask "What is 2+2?" --search 50
  %(prog)s search 100 --session my_session.json
  %(prog)s solution --session my_session.json
  %(prog)s verify --session my_session.json
  %(prog)s export json output.json --session my_session.json
  %(prog)s sample 10
  %(prog)s consistency 20
        """
    )

    # Global options
    parser.add_argument('--session', '-s', type=str,
                       help='Session file to load')
    parser.add_argument('--provider', type=str,
                       help='LLM provider (openai, anthropic, ollama, mock)')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')

    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # ask command
    ask_parser = subparsers.add_parser('ask', help='Start new reasoning session')
    ask_parser.add_argument('question', type=str, help='Question to reason about')
    ask_parser.add_argument('--search', type=int, dest='simulations',
                           help='Number of simulations to run')

    # search command
    search_parser = subparsers.add_parser('search', help='Run MCTS simulations')
    search_parser.add_argument('simulations', type=int, nargs='?', default=20,
                              help='Number of simulations (default: 20)')

    # solution command
    subparsers.add_parser('solution', help='Show best solution')

    # tree command
    subparsers.add_parser('tree', help='Visualize search tree')

    # sample command
    sample_parser = subparsers.add_parser('sample', help='Sample diverse paths')
    sample_parser.add_argument('count', type=int, nargs='?', default=5,
                               help='Number of paths to sample (default: 5)')

    # consistency command
    consistency_parser = subparsers.add_parser('consistency',
                                              help='Check solution consistency')
    consistency_parser.add_argument('trials', type=int, nargs='?',
                                   help='Number of trials (optional)')

    # export command
    export_parser = subparsers.add_parser('export', help='Export tree')
    export_parser.add_argument('format', type=str,
                              choices=['json', 'markdown', 'md', 'dot', 'csv'],
                              help='Export format')
    export_parser.add_argument('output', type=str, nargs='?',
                              help='Output filename (optional)')

    # verify command
    verify_parser = subparsers.add_parser('verify',
                                         help='Verify solution correctness')
    verify_parser.add_argument('node_index', type=int, nargs='?',
                              help='Node index to verify (optional, default: current solution)')

    # inspect commands
    inspect_parser = subparsers.add_parser('inspect', help='Inspect node (preview)')
    inspect_parser.add_argument('node_index', type=int, help='Node index')

    inspect_full_parser = subparsers.add_parser('inspect-full',
                                               help='Inspect node (full state)')
    inspect_full_parser.add_argument('node_index', type=int, help='Node index')

    # path command
    path_parser = subparsers.add_parser('path', help='Show path to node')
    path_parser.add_argument('node_index', type=int, help='Node index')

    # compare command
    compare_parser = subparsers.add_parser('compare', help='Compare nodes')
    compare_parser.add_argument('indices', type=int, nargs='+',
                               help='Node indices to compare')

    # nodes command
    subparsers.add_parser('nodes', help='List all nodes')

    # stats command
    subparsers.add_parser('stats', help='Show session statistics')

    # solutions command
    subparsers.add_parser('solutions', help='List finalized solutions')

    # status command
    subparsers.add_parser('status', help='Show current status')

    # save/load commands
    save_parser = subparsers.add_parser('save', help='Save session')
    save_parser.add_argument('filename', type=str, nargs='?',
                            help='Filename (optional)')

    load_parser = subparsers.add_parser('load', help='Load session')
    load_parser.add_argument('filename', type=str, help='Filename')

    # config commands
    model_parser = subparsers.add_parser('model', help='Configure LLM model')
    model_parser.add_argument('provider', type=str, nargs='?',
                             help='Provider name')
    model_parser.add_argument('model', type=str, nargs='?',
                             help='Model name')

    temp_parser = subparsers.add_parser('temperature', help='Set temperature')
    temp_parser.add_argument('value', type=float, help='Temperature (0.0-2.0)')

    explore_parser = subparsers.add_parser('exploration',
                                          help='Set exploration constant')
    explore_parser.add_argument('value', type=float,
                               help='Exploration constant')

    # help command
    subparsers.add_parser('help', help='Show help')

    return parser


def main():
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Check if command was provided
    if not args.command:
        parser.print_help()
        sys.exit(1)

    try:
        # Create CLI instance
        cli = MCTSCLI(session_file=args.session)

        # Set provider if specified
        if args.provider:
            provider_cmd = f"model {args.provider}"
            cli.execute_command(provider_cmd)

        # Build and execute command
        command_str = build_command_from_args(args)

        if args.verbose:
            print(f"Executing: {command_str}", file=sys.stderr)

        result = cli.execute_command(command_str)

        # Print result
        print(result)

        sys.exit(0)

    except MCTSCLIError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f"Unexpected error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
