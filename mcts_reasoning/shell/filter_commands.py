"""
Filter and transformation commands.

Commands that filter, sort, and transform streams:
- filter: Filter paths by criteria
- sort: Sort paths by value/visits
- head: Take first N items
- tail: Take last N items
- grep: Search in solutions/paths
- unique: Remove duplicates
"""

import re
from typing import List, Dict, Any

from .command_base import CommandBase, CommandContext, CommandError
from .streams import Stream, PathStream, SolutionStream, TextStream


class FilterCommand(CommandBase):
    """Filter paths by criteria."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("filter requires input stream")

        # Only works on PathStream
        if not isinstance(context.input_stream, PathStream):
            raise CommandError("filter only works on PathStream")

        paths = list(context.input_stream.paths)
        filtered = []

        # Filter by value
        min_value = kwargs.get('min-value') or kwargs.get('min_value')
        max_value = kwargs.get('max-value') or kwargs.get('max_value')

        # Filter by visits
        min_visits = kwargs.get('min-visits') or kwargs.get('min_visits')
        max_visits = kwargs.get('max-visits') or kwargs.get('max_visits')

        # Filter by length
        min_length = kwargs.get('min-length') or kwargs.get('min_length')
        max_length = kwargs.get('max-length') or kwargs.get('max_length')

        for path in paths:
            avg_value = path.total_value / max(path.total_visits, 1)

            # Check all conditions
            if min_value is not None and avg_value < float(min_value):
                continue
            if max_value is not None and avg_value > float(max_value):
                continue
            if min_visits is not None and path.total_visits < int(min_visits):
                continue
            if max_visits is not None and path.total_visits > int(max_visits):
                continue
            if min_length is not None and path.length < int(min_length):
                continue
            if max_length is not None and path.length > int(max_length):
                continue

            filtered.append(path)

        return PathStream(filtered, context.input_stream.source_mcts)

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """filter - Filter paths by criteria

Usage:
  filter --min-value 0.8      Keep paths with value >= 0.8
  filter --max-value 0.5      Keep paths with value <= 0.5
  filter --min-visits 10      Keep paths with visits >= 10
  filter --max-visits 100     Keep paths with visits <= 100
  filter --min-length 3       Keep paths with length >= 3
  filter --max-length 10      Keep paths with length <= 10

Examples:
  sample 20 | filter --min-value 0.7
  sample 50 | filter --min-visits 5 --max-length 8

Filters paths based on value, visits, or length criteria.
"""


class SortCommand(CommandBase):
    """Sort paths by criterion."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("sort requires input stream")

        # Get sort criterion
        by = kwargs.get('by', 'value')
        reverse = kwargs.get('reverse', True)  # Default: descending

        if isinstance(context.input_stream, PathStream):
            paths = list(context.input_stream.paths)

            if by == 'value':
                key_func = lambda p: p.total_value / max(p.total_visits, 1)
            elif by == 'visits':
                key_func = lambda p: p.total_visits
            elif by == 'length':
                key_func = lambda p: p.length
            else:
                raise CommandError(f"Unknown sort criterion: {by}")

            sorted_paths = sorted(paths, key=key_func, reverse=reverse)
            return PathStream(sorted_paths, context.input_stream.source_mcts)

        elif isinstance(context.input_stream, TextStream):
            lines = list(context.input_stream.lines)
            sorted_lines = sorted(lines, reverse=reverse)
            return TextStream(sorted_lines)

        else:
            raise CommandError(f"sort cannot process {type(context.input_stream).__name__}")

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """sort - Sort paths or text

Usage:
  sort                        Sort by value (descending)
  sort --by value             Sort by average value
  sort --by visits            Sort by visit count
  sort --by length            Sort by path length
  sort --reverse false        Sort ascending

Examples:
  sample 10 | sort --by visits
  sample 20 | sort --by value --reverse false

Sorts paths or text lines.
"""


class HeadCommand(CommandBase):
    """Take first N items from stream."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("head requires input stream")

        n = int(args[0]) if args else 10

        if isinstance(context.input_stream, PathStream):
            paths = list(context.input_stream.paths)[:n]
            return PathStream(paths, context.input_stream.source_mcts)

        elif isinstance(context.input_stream, SolutionStream):
            solutions = context.input_stream.solutions[:n]
            metadata = context.input_stream.metadata[:n]
            return SolutionStream(solutions, metadata)

        elif isinstance(context.input_stream, TextStream):
            lines = context.input_stream.lines[:n]
            return TextStream(lines)

        else:
            raise CommandError(f"head cannot process {type(context.input_stream).__name__}")

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """head - Take first N items

Usage:
  head [N]                    Take first N items (default: 10)

Examples:
  sample 100 | head 5
  sample 50 | sort | head 10

Takes the first N items from the input stream.
"""


class TailCommand(CommandBase):
    """Take last N items from stream."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("tail requires input stream")

        n = int(args[0]) if args else 10

        if isinstance(context.input_stream, PathStream):
            paths = list(context.input_stream.paths)[-n:]
            return PathStream(paths, context.input_stream.source_mcts)

        elif isinstance(context.input_stream, SolutionStream):
            solutions = context.input_stream.solutions[-n:]
            metadata = context.input_stream.metadata[-n:]
            return SolutionStream(solutions, metadata)

        elif isinstance(context.input_stream, TextStream):
            lines = context.input_stream.lines[-n:]
            return TextStream(lines)

        else:
            raise CommandError(f"tail cannot process {type(context.input_stream).__name__}")

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """tail - Take last N items

Usage:
  tail [N]                    Take last N items (default: 10)

Examples:
  sample 100 | tail 5
  sample 50 | sort | tail 10

Takes the last N items from the input stream.
"""


class GrepCommand(CommandBase):
    """Search for pattern in stream content."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("grep requires input stream")

        if not args:
            raise CommandError("Usage: grep <pattern>")

        pattern = args[0]
        case_sensitive = not kwargs.get('i', False)  # -i for case-insensitive

        # Compile regex
        flags = 0 if case_sensitive else re.IGNORECASE
        try:
            regex = re.compile(pattern, flags)
        except re.error as e:
            raise CommandError(f"Invalid regex pattern: {e}")

        if isinstance(context.input_stream, PathStream):
            # Filter paths whose final state matches pattern
            paths = list(context.input_stream.paths)
            matched = [p for p in paths if regex.search(p.final_state)]
            return PathStream(matched, context.input_stream.source_mcts)

        elif isinstance(context.input_stream, SolutionStream):
            # Filter solutions matching pattern
            matched_solutions = []
            matched_metadata = []
            for sol, meta in zip(context.input_stream.solutions, context.input_stream.metadata):
                if regex.search(sol):
                    matched_solutions.append(sol)
                    matched_metadata.append(meta)
            return SolutionStream(matched_solutions, matched_metadata)

        elif isinstance(context.input_stream, TextStream):
            # Filter text lines
            matched = [line for line in context.input_stream.lines if regex.search(line)]
            return TextStream(matched)

        else:
            raise CommandError(f"grep cannot process {type(context.input_stream).__name__}")

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """grep - Search for pattern

Usage:
  grep <pattern>              Search for regex pattern
  grep <pattern> -i           Case-insensitive search

Examples:
  sample 10 | grep "prime"
  best | grep "\\d+" -i

Filters stream items matching the regex pattern.
"""


class UniqueCommand(CommandBase):
    """Remove duplicate items."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("unique requires input stream")

        if isinstance(context.input_stream, SolutionStream):
            # Remove duplicate solutions
            seen = set()
            unique_solutions = []
            unique_metadata = []

            for sol, meta in zip(context.input_stream.solutions, context.input_stream.metadata):
                if sol not in seen:
                    seen.add(sol)
                    unique_solutions.append(sol)
                    unique_metadata.append(meta)

            return SolutionStream(unique_solutions, unique_metadata)

        elif isinstance(context.input_stream, TextStream):
            # Remove duplicate lines
            seen = set()
            unique_lines = []
            for line in context.input_stream.lines:
                if line not in seen:
                    seen.add(line)
                    unique_lines.append(line)

            return TextStream(unique_lines)

        else:
            raise CommandError(f"unique cannot process {type(context.input_stream).__name__}")

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """unique - Remove duplicates

Usage:
  unique                      Remove duplicate items

Examples:
  sample 20 | best | unique

Removes duplicate solutions or text lines.
"""


class CountCommand(CommandBase):
    """Count items in stream."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("count requires input stream")

        count = 0

        if isinstance(context.input_stream, PathStream):
            count = len(context.input_stream.paths)
        elif isinstance(context.input_stream, SolutionStream):
            count = len(context.input_stream.solutions)
        elif isinstance(context.input_stream, TextStream):
            count = len(context.input_stream.lines)
        else:
            # Count items by iterating
            count = sum(1 for _ in context.input_stream.items())

        return TextStream([str(count)])

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """count - Count items in stream

Usage:
  count                       Count items

Examples:
  sample 100 | filter --min-value 0.8 | count
  load tree.json | sample 50 | count

Returns the number of items in the stream.
"""
