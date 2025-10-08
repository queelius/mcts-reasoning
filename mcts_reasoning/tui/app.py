"""
Main TUI application using Rich for formatting.

Provides a Claude Code-style REPL interface with:
- Colored output
- Tree visualization
- Progress indicators
- Command history and completion (via prompt_toolkit)
"""

import sys
from typing import Optional
from pathlib import Path

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.tree import Tree as RichTree
    from rich.syntax import Syntax
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.markdown import Markdown
    from rich import print as rprint
    _has_rich = True
except ImportError:
    _has_rich = False
    Console = None

from .session import SessionState
from .commands import CommandParser, CommandHandler
from .prompt import create_enhanced_prompt


class ReasoningTUI:
    """
    Text User Interface for MCTS-Reasoning.

    A stateful, interactive REPL for reasoning with MCTS.
    """

    def __init__(self, use_rich: bool = True):
        """
        Initialize TUI.

        Args:
            use_rich: Use rich formatting (if available)
        """
        self.session = SessionState()
        self.parser = CommandParser()
        self.handler = CommandHandler(self.session)
        self.running = False

        # Rich console
        self.use_rich = use_rich and _has_rich
        if self.use_rich:
            self.console = Console()
        else:
            self.console = None

        # Enhanced prompt with history and completion
        self.prompt = create_enhanced_prompt()
        self.prompt.update_context(self.session.provider_name, self.session.model_name)

    def start(self):
        """Start the TUI REPL."""
        self.running = True

        # Show welcome message
        self._show_welcome()

        # Main REPL loop
        while self.running:
            try:
                # Get input with history and completion
                line = self.prompt.prompt(use_rich=self.use_rich)

                if not line:
                    continue

                # Add to history
                self.session.command_history.append(line)

                # Parse command
                command = self.parser.parse(line)

                if command is None:
                    self._print_error("Not a command. Use /help for available commands.")
                    continue

                # Execute command
                success, message = self.handler.execute(command)

                # Update prompt context (in case model changed)
                self.prompt.update_context(self.session.provider_name, self.session.model_name)

                # Handle special messages
                if message == "EXIT":
                    self._print_info("Goodbye!")
                    self.running = False
                    break

                elif message == "TREE_VISUALIZATION":
                    self._show_tree()

                elif success:
                    self._print_success(message)

                else:
                    self._print_error(message)

            except KeyboardInterrupt:
                self._print_info("\nUse /exit to quit")
                continue

            except EOFError:
                self.running = False
                break

            except Exception as e:
                self._print_error(f"Unexpected error: {e}")
                if "--debug" in sys.argv:
                    raise

    def _show_welcome(self):
        """Show welcome message."""
        if self.use_rich:
            self.console.print(Panel.fit(
                "[bold cyan]MCTS-Reasoning TUI[/bold cyan]\n\n"
                "Interactive reasoning with Monte Carlo Tree Search\n"
                "Type [bold]/help[/bold] for available commands",
                border_style="cyan"
            ))

            # Show initial status
            status = self.session.get_status()
            table = Table(show_header=False, box=None)
            table.add_row("Provider:", f"[yellow]{status['provider']}[/yellow]")
            table.add_row("Model:", f"[yellow]{status['model']}[/yellow]")
            table.add_row("Temperature:", f"[yellow]{status['temperature']}[/yellow]")

            self.console.print("\n[bold]Current Configuration:[/bold]")
            self.console.print(table)
        else:
            print("=" * 60)
            print("MCTS-Reasoning TUI")
            print("=" * 60)
            print("\nInteractive reasoning with Monte Carlo Tree Search")
            print("Type /help for available commands\n")

            status = self.session.get_status()
            print(f"Provider: {status['provider']}")
            print(f"Model: {status['model']}")
            print(f"Temperature: {status['temperature']}")

    def _show_tree(self):
        """Visualize the MCTS tree."""
        if not self.session.mcts or not self.session.mcts.root:
            self._print_error("No tree to visualize")
            return

        if self.use_rich:
            tree = self._build_rich_tree()
            self.console.print("\n[bold]MCTS Search Tree:[/bold]")
            self.console.print(tree)

            # Show statistics
            stats = self.session.mcts.stats
            table = Table(title="Tree Statistics", show_header=False)
            table.add_row("Total Nodes:", f"[cyan]{stats['total_nodes']}[/cyan]")
            table.add_row("Max Depth:", f"[cyan]{stats['max_depth']}[/cyan]")
            table.add_row("Root Visits:", f"[cyan]{stats['root_visits']}[/cyan]")
            table.add_row("Best Value:", f"[cyan]{stats['best_value']:.3f}[/cyan]")

            self.console.print("\n")
            self.console.print(table)

        else:
            print("\nMCTS Search Tree:")
            self._print_tree_text(self.session.mcts.root, prefix="", is_last=True)

            stats = self.session.mcts.stats
            print(f"\nTree Statistics:")
            print(f"  Total Nodes: {stats['total_nodes']}")
            print(f"  Max Depth: {stats['max_depth']}")
            print(f"  Best Value: {stats['best_value']:.3f}")

    def _build_rich_tree(self, max_depth: int = 3) -> RichTree:
        """Build rich tree visualization."""
        root_node = self.session.mcts.root

        tree = RichTree(
            f"[bold cyan]Root[/bold cyan] "
            f"(visits={root_node.visits}, value={root_node.value:.2f})"
        )

        self._add_rich_children(tree, root_node, depth=0, max_depth=max_depth)

        return tree

    def _add_rich_children(self, tree: RichTree, node, depth: int, max_depth: int):
        """Recursively add children to rich tree."""
        if depth >= max_depth or not node.children:
            return

        # Sort children by visits (show most visited first)
        sorted_children = sorted(node.children, key=lambda n: n.visits, reverse=True)

        # Show top 5 children at each level
        for i, child in enumerate(sorted_children[:5]):
            action_str = str(child.action_taken)
            if len(action_str) > 50:
                action_str = action_str[:47] + "..."

            label = (
                f"[yellow]{action_str}[/yellow] "
                f"(visits={child.visits}, value={child.value:.2f}, "
                f"ucb={child.ucb1():.3f})"
            )

            branch = tree.add(label)
            self._add_rich_children(branch, child, depth + 1, max_depth)

        # Show count of remaining children
        if len(sorted_children) > 5:
            tree.add(f"[dim]... {len(sorted_children) - 5} more children[/dim]")

    def _print_tree_text(self, node, prefix: str = "", is_last: bool = True):
        """Print tree in plain text."""
        # Print current node
        connector = "└── " if is_last else "├── "
        action_str = str(node.action_taken) if node.action_taken else "Root"
        if len(action_str) > 50:
            action_str = action_str[:47] + "..."

        print(f"{prefix}{connector}{action_str} (v={node.visits}, val={node.value:.2f})")

        # Print children (limit to top 3)
        if node.children:
            sorted_children = sorted(node.children, key=lambda n: n.visits, reverse=True)[:3]
            extension = "    " if is_last else "│   "

            for i, child in enumerate(sorted_children):
                is_last_child = (i == len(sorted_children) - 1)
                self._print_tree_text(child, prefix + extension, is_last_child)

    def _print_success(self, message: str):
        """Print success message."""
        if self.use_rich:
            self.console.print(f"\n[green]{message}[/green]")
        else:
            print(f"\n✓ {message}")

    def _print_error(self, message: str):
        """Print error message."""
        if self.use_rich:
            self.console.print(f"\n[red]Error:[/red] {message}")
        else:
            print(f"\n✗ Error: {message}")

    def _print_info(self, message: str):
        """Print info message."""
        if self.use_rich:
            self.console.print(f"\n[cyan]{message}[/cyan]")
        else:
            print(f"\n{message}")


def run_tui(use_rich: bool = True):
    """
    Run the TUI application.

    Args:
        use_rich: Use rich formatting (if available)
    """
    if use_rich and not _has_rich:
        print("Warning: 'rich' library not installed. Install with: pip install rich")
        print("Falling back to plain text mode...\n")
        use_rich = False

    tui = ReasoningTUI(use_rich=use_rich)
    tui.start()


def main():
    """Entry point for command line."""
    import argparse

    parser = argparse.ArgumentParser(description="MCTS-Reasoning TUI")
    parser.add_argument("--no-rich", action="store_true", help="Disable rich formatting")
    parser.add_argument("--load", type=str, help="Load session from file")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    args = parser.parse_args()

    # Create TUI
    tui = ReasoningTUI(use_rich=not args.no_rich)

    # Load session if specified
    if args.load:
        filepath = Path(args.load)
        if tui.session.load_session(filepath):
            tui._print_success(f"Loaded session from {filepath}")
        else:
            tui._print_error(f"Failed to load session from {filepath}")

    # Start
    tui.start()


if __name__ == "__main__":
    main()


__all__ = ['ReasoningTUI', 'run_tui', 'main']
