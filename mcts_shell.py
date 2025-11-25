#!/usr/bin/env python3
"""
MCTS-Reasoning Shell

A Unix-style composable shell for MCTS reasoning.

Usage:
    python mcts_shell.py [options]

Options:
    --no-rich           Disable rich formatting
    --config FILE       Use custom config file
    --provider NAME     Set LLM provider (openai, anthropic, ollama, mock)
    --model NAME        Set model name
    --debug             Enable debug mode
"""

import sys
import argparse
from pathlib import Path

# Add parent directory to path for development
sys.path.insert(0, str(Path(__file__).parent))

from mcts_reasoning.shell.core import Shell
from mcts_reasoning.config import Config


def main():
    """Main entry point for the shell."""
    parser = argparse.ArgumentParser(
        description='MCTS-Reasoning Shell - Unix-style composable reasoning',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mcts_shell.py
  mcts_shell.py --provider openai --model gpt-4
  mcts_shell.py --config ~/.mcts-reasoning/custom.json

Commands:
  ask <question>              Start new reasoning task
  search <N>                  Run N MCTS simulations
  sample <N>                  Sample N paths
  best                        Get best solution

  filter --min-value 0.8      Filter by criteria
  sort --by value             Sort paths
  head <N>, tail <N>          Take first/last N

  load <file>                 Load tree from file
  save <file>                 Save to file
  export <format>             Export (markdown, json, dot, csv)

  stats                       Show statistics
  tree                        Display tree structure
  verify                      Verify solution correctness
  consistency <N>             Check consistency

  help                        Show all commands
  exit                        Quit shell

Piping:
  ask "primes < 100" | search 100 | sample 5 | format table
  load tree.json | sample 10 --strategy diverse | best | verify
  ask "quadratic" | search 50 | export markdown > report.md
"""
    )

    parser.add_argument(
        '--no-rich',
        action='store_true',
        help='Disable rich formatting'
    )

    parser.add_argument(
        '--config',
        type=str,
        help='Path to config file'
    )

    parser.add_argument(
        '--provider',
        type=str,
        choices=['openai', 'anthropic', 'ollama', 'mock'],
        help='LLM provider'
    )

    parser.add_argument(
        '--model',
        type=str,
        help='Model name'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode'
    )

    args = parser.parse_args()

    # DEPRECATION WARNING
    print("=" * 70, file=sys.stderr)
    print("WARNING: mcts_shell.py is DEPRECATED", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("", file=sys.stderr)
    print("This Unix-style piping shell is deprecated. Please use:", file=sys.stderr)
    print("", file=sys.stderr)
    print("  • mcts-shell    - Interactive TUI with rich formatting", file=sys.stderr)
    print("  • mcts          - Non-interactive CLI for single commands", file=sys.stderr)
    print("", file=sys.stderr)
    print("Examples:", file=sys.stderr)
    print("  mcts-shell                    # Interactive shell", file=sys.stderr)
    print('  mcts ask "What is 2+2?"       # Single command', file=sys.stderr)
    print("", file=sys.stderr)
    print("This script will be removed in a future version.", file=sys.stderr)
    print("=" * 70, file=sys.stderr)
    print("", file=sys.stderr)

    # Set debug mode
    if args.debug:
        import os
        os.environ['DEBUG'] = '1'

    # Load or create config
    if args.config:
        config = Config(config_path=args.config)
    else:
        config = Config()

    # Override provider/model if specified
    if args.provider:
        config.set('shell', 'provider', args.provider)

    if args.model:
        config.set('shell', 'model', args.model)

    # Save config
    config.save()

    # Create and start shell
    use_rich = not args.no_rich

    try:
        shell = Shell(config=config, use_rich=use_rich)
        shell.start()

    except KeyboardInterrupt:
        print("\nInterrupted")
        sys.exit(1)

    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        if args.debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
