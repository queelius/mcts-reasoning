"""
MCTS-Reasoning Shell

A Unix-style composable shell for MCTS reasoning with piping, I/O redirection,
and scriptable workflows.

Example usage:
    $ mcts-shell
    mcts> ask "Find primes < 100" | search 50 | sample 3 | format json
    mcts> load session.json | search 100 | best | save result.txt
"""

from .core import Shell
from .streams import Stream, MCTSStream, PathStream, TextStream
from .commands import CommandRegistry

__all__ = ['Shell', 'Stream', 'MCTSStream', 'PathStream', 'TextStream', 'CommandRegistry']
