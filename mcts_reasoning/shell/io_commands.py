"""
I/O commands for loading, saving, and formatting data.

Commands:
- load: Load MCTS tree or data from file
- save: Save stream to file
- export: Export in various formats
- format: Format stream for display
- import: Import from various formats
"""

import json
from pathlib import Path
from typing import List, Dict, Any

from .command_base import CommandBase, CommandContext, CommandError
from .streams import Stream, MCTSStream, PathStream, SolutionStream, TextStream, stream_from_file


class LoadCommand(CommandBase):
    """Load MCTS tree or data from file."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not args:
            raise CommandError("Usage: load <filepath>")

        filepath = args[0]

        try:
            stream = stream_from_file(filepath)
            return stream
        except FileNotFoundError:
            raise CommandError(f"File not found: {filepath}")
        except Exception as e:
            raise CommandError(f"Error loading file: {e}")

    def get_help(self) -> str:
        return """load - Load data from file

Usage:
  load <filepath>             Load MCTS tree or data

Examples:
  load session.json
  load tree.json | search 50

Loads MCTS trees, paths, or other data from JSON files.
Auto-detects format based on file content.
"""


class SaveCommand(CommandBase):
    """Save stream to file."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not args:
            raise CommandError("Usage: save <filepath>")

        if not context.input_stream:
            raise CommandError("save requires input stream")

        filepath = args[0]
        path = Path(filepath)

        # Get format
        fmt = kwargs.get('format', 'json')

        # Create parent directory if needed
        path.parent.mkdir(parents=True, exist_ok=True)

        # Write to file
        try:
            if fmt == 'json':
                content = context.input_stream.to_json()
            elif fmt == 'text' or fmt == 'txt':
                content = context.input_stream.to_text()
            else:
                raise CommandError(f"Unknown format: {fmt}")

            path.write_text(content)

        except Exception as e:
            raise CommandError(f"Error saving file: {e}")

        # Return input stream unchanged (pass-through)
        return context.input_stream

    def requires_input(self) -> bool:
        return True

    def produces_output(self) -> bool:
        return True  # Pass-through

    def get_help(self) -> str:
        return """save - Save stream to file

Usage:
  save <filepath>             Save as JSON (default)
  save <filepath> --format text   Save as text
  save <filepath> --format json   Save as JSON

Examples:
  ask "problem" | search 100 | save result.json
  best | save solution.txt --format text

Saves the input stream to a file.
Acts as pass-through (outputs the same stream).
"""


class ExportCommand(CommandBase):
    """Export stream in various formats."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("export requires input stream")

        # Get format
        fmt = args[0] if args else 'markdown'

        if fmt == 'markdown' or fmt == 'md':
            content = self._export_markdown(context.input_stream)
        elif fmt == 'json':
            content = context.input_stream.to_json()
        elif fmt == 'text' or fmt == 'txt':
            content = context.input_stream.to_text()
        elif fmt == 'dot':
            content = self._export_dot(context.input_stream)
        elif fmt == 'csv':
            content = self._export_csv(context.input_stream)
        else:
            raise CommandError(f"Unknown export format: {fmt}")

        return TextStream([content])

    def _export_markdown(self, stream: Stream) -> str:
        """Export as Markdown."""
        if isinstance(stream, MCTSStream):
            # Use built-in markdown export
            if hasattr(stream.mcts, 'to_markdown'):
                return stream.mcts.to_markdown()
            else:
                # Fallback
                stats = stream.mcts.stats
                lines = [
                    "# MCTS Reasoning Tree",
                    "",
                    f"**Nodes**: {stats['total_nodes']}",
                    f"**Max depth**: {stats['max_depth']}",
                    f"**Best value**: {stats['best_value']:.3f}",
                    "",
                    "## Best Solution",
                    "```",
                    stream.mcts.solution,
                    "```"
                ]
                return '\n'.join(lines)

        elif isinstance(stream, PathStream):
            lines = ["# Sampled Paths", ""]
            for i, path in enumerate(stream.paths, 1):
                avg_value = path.total_value / max(path.total_visits, 1)
                lines.append(f"## Path {i}")
                lines.append(f"- **Value**: {avg_value:.3f}")
                lines.append(f"- **Visits**: {path.total_visits}")
                lines.append(f"- **Length**: {path.length}")
                lines.append("")
                lines.append("### Solution")
                lines.append("```")
                lines.append(path.final_state)
                lines.append("```")
                lines.append("")

            return '\n'.join(lines)

        elif isinstance(stream, SolutionStream):
            lines = ["# Solutions", ""]
            for i, sol in enumerate(stream.solutions, 1):
                lines.append(f"## Solution {i}")
                lines.append("```")
                lines.append(sol)
                lines.append("```")
                lines.append("")

            return '\n'.join(lines)

        else:
            return stream.to_text()

    def _export_dot(self, stream: Stream) -> str:
        """Export as Graphviz DOT format."""
        if not isinstance(stream, MCTSStream):
            raise CommandError("DOT export only works with MCTSStream")

        mcts = stream.mcts
        if not mcts.root:
            return "digraph G {}"

        lines = ["digraph MCTS {", "  rankdir=TB;", ""]

        # Track node IDs
        node_ids = {}
        node_counter = [0]

        def add_node(node):
            if node not in node_ids:
                node_ids[node] = f"n{node_counter[0]}"
                node_counter[0] += 1

            node_id = node_ids[node]
            avg_value = node.value / max(node.visits, 1)

            # Truncate state for label
            label = node.state[:50].replace('"', '\\"')
            if len(node.state) > 50:
                label += "..."

            lines.append(f'  {node_id} [label="{label}\\nv={avg_value:.2f} n={node.visits}"];')

            # Add edges to children
            for child in node.children:
                add_node(child)
                child_id = node_ids[child]
                lines.append(f"  {node_id} -> {child_id};")

        add_node(mcts.root)
        lines.append("}")

        return '\n'.join(lines)

    def _export_csv(self, stream: Stream) -> str:
        """Export as CSV."""
        if isinstance(stream, PathStream):
            lines = ["path_id,length,total_value,total_visits,avg_value,final_state"]
            for i, path in enumerate(stream.paths):
                avg_value = path.total_value / max(path.total_visits, 1)
                final_state = path.final_state.replace('"', '""')[:100]  # Truncate and escape
                lines.append(f'{i},{path.length},{path.total_value},{path.total_visits},{avg_value},"{final_state}"')

            return '\n'.join(lines)

        elif isinstance(stream, SolutionStream):
            lines = ["solution_id,solution"]
            for i, sol in enumerate(stream.solutions):
                sol_escaped = sol.replace('"', '""')[:200]
                lines.append(f'{i},"{sol_escaped}"')

            return '\n'.join(lines)

        else:
            raise CommandError("CSV export only works with PathStream or SolutionStream")

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """export - Export in various formats

Usage:
  export [format]             Export in format (default: markdown)

Formats:
  markdown, md                Markdown report
  json                        JSON data
  text, txt                   Plain text
  dot                         Graphviz DOT (for trees)
  csv                         CSV spreadsheet

Examples:
  ask "problem" | search 100 | export markdown > report.md
  sample 10 | export csv > paths.csv
  load tree.json | export dot | dot -Tpng > tree.png

Exports data in various formats for documentation or analysis.
"""


class FormatCommand(CommandBase):
    """Format stream for display."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("format requires input stream")

        fmt = args[0] if args else 'text'

        if fmt == 'json':
            content = context.input_stream.to_json()
        elif fmt == 'text' or fmt == 'txt':
            content = context.input_stream.to_text()
        elif fmt == 'table':
            content = self._format_table(context.input_stream)
        elif fmt == 'tree':
            content = self._format_tree(context.input_stream)
        elif fmt == 'solution':
            content = self._format_solution(context.input_stream)
        else:
            raise CommandError(f"Unknown format: {fmt}")

        return TextStream([content])

    def _format_table(self, stream: Stream) -> str:
        """Format as ASCII table."""
        if isinstance(stream, PathStream):
            lines = []
            lines.append("ID  | Length | Value  | Visits | Final State")
            lines.append("----|--------|--------|--------|-------------------")

            for i, path in enumerate(stream.paths):
                avg_value = path.total_value / max(path.total_visits, 1)
                final_preview = path.final_state[:30].replace('\n', ' ')
                if len(path.final_state) > 30:
                    final_preview += "..."

                lines.append(f"{i:3d} | {path.length:6d} | {avg_value:6.3f} | {path.total_visits:6d} | {final_preview}")

            return '\n'.join(lines)

        else:
            return stream.to_text()

    def _format_tree(self, stream: Stream) -> str:
        """Format as ASCII tree."""
        if not isinstance(stream, MCTSStream):
            raise CommandError("tree format only works with MCTSStream")

        mcts = stream.mcts
        if not mcts.root:
            return "Empty tree"

        lines = []

        def print_node(node, prefix="", is_last=True):
            # Node info
            avg_value = node.value / max(node.visits, 1)
            connector = "└── " if is_last else "├── "

            state_preview = node.state[:40].replace('\n', ' ')
            if len(node.state) > 40:
                state_preview += "..."

            lines.append(f"{prefix}{connector}{state_preview} (v={avg_value:.2f}, n={node.visits})")

            # Children
            if node.children:
                extension = "    " if is_last else "│   "
                for i, child in enumerate(node.children):
                    print_node(child, prefix + extension, i == len(node.children) - 1)

        print_node(mcts.root)
        return '\n'.join(lines)

    def _format_solution(self, stream: Stream) -> str:
        """Format as just the solution text."""
        if isinstance(stream, SolutionStream):
            if len(stream.solutions) == 1:
                return stream.solutions[0]
            else:
                return '\n\n---\n\n'.join(stream.solutions)

        elif isinstance(stream, MCTSStream):
            return stream.mcts.solution

        elif isinstance(stream, PathStream):
            paths = list(stream.paths)
            if paths:
                best = max(paths, key=lambda p: p.total_value / max(p.total_visits, 1))
                return best.final_state

        return stream.to_text()

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """format - Format stream for display

Usage:
  format [type]               Format for display (default: text)

Types:
  text                        Plain text
  json                        JSON format
  table                       ASCII table (for paths)
  tree                        ASCII tree (for MCTS)
  solution                    Just the solution text

Examples:
  ask "problem" | search 100 | format tree
  sample 10 | format table
  best | format solution

Formats data for human-readable display.
"""


class CatCommand(CommandBase):
    """Display file contents (like Unix cat)."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not args:
            # No file specified - act as pass-through
            if context.input_stream:
                return context.input_stream
            raise CommandError("Usage: cat <file>")

        filepath = args[0]

        try:
            stream = stream_from_file(filepath)
            # Convert to text for display
            return TextStream([stream.to_text()])
        except FileNotFoundError:
            raise CommandError(f"File not found: {filepath}")
        except Exception as e:
            raise CommandError(f"Error reading file: {e}")

    def get_help(self) -> str:
        return """cat - Display file contents

Usage:
  cat <file>                  Display file
  cat                         Display input stream (pass-through)

Examples:
  cat session.json
  best | cat

Displays file contents or passes through input stream.
"""
