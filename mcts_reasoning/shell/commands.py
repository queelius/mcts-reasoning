"""
Central import point for all commands.

This module re-exports all commands for easy importing.
"""

from .command_base import (
    CommandBase,
    CommandContext,
    CommandError,
    CommandRegistry,
    HelpCommand,
    EchoCommand,
    ExitCommand
)

from .reasoning_commands import (
    AskCommand,
    SearchCommand,
    ExploreCommand,
    SampleCommand,
    BestCommand,
    WorstCommand,
    RandomCommand
)

from .filter_commands import (
    FilterCommand,
    SortCommand,
    HeadCommand,
    TailCommand,
    GrepCommand,
    UniqueCommand,
    CountCommand
)

from .io_commands import (
    LoadCommand,
    SaveCommand,
    ExportCommand,
    FormatCommand,
    CatCommand
)

from .analysis_commands import (
    StatsCommand,
    TreeCommand,
    VerifyCommand,
    ConsistencyCommand,
    DiffCommand,
    ExplainCommand
)

__all__ = [
    # Base
    'CommandBase',
    'CommandContext',
    'CommandError',
    'CommandRegistry',
    'HelpCommand',
    'EchoCommand',
    'ExitCommand',

    # Reasoning
    'AskCommand',
    'SearchCommand',
    'ExploreCommand',
    'SampleCommand',
    'BestCommand',
    'WorstCommand',
    'RandomCommand',

    # Filtering
    'FilterCommand',
    'SortCommand',
    'HeadCommand',
    'TailCommand',
    'GrepCommand',
    'UniqueCommand',
    'CountCommand',

    # I/O
    'LoadCommand',
    'SaveCommand',
    'ExportCommand',
    'FormatCommand',
    'CatCommand',

    # Analysis
    'StatsCommand',
    'TreeCommand',
    'VerifyCommand',
    'ConsistencyCommand',
    'DiffCommand',
    'ExplainCommand',
]
