"""
Core reasoning commands for MCTS search and exploration.

Commands:
- ask: Create a new reasoning task
- search: Run MCTS simulations
- explore: Explore to a specific depth
- rollout: Single rollout from current state
- sample: Sample paths from tree
- best: Get best solution
"""

from typing import List, Dict, Any

from .command_base import CommandBase, CommandContext, CommandError
from .streams import Stream, MCTSStream, PathStream, SolutionStream, TextStream

from ..reasoning import ReasoningMCTS
from ..compositional.providers import get_llm


class AskCommand(CommandBase):
    """Create a new reasoning task."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not args:
            raise CommandError("Usage: ask <question>")

        question = ' '.join(args)

        # Get LLM provider from context or create new one
        llm = context.llm_provider
        if not llm:
            # Try to get from config
            provider_name = context.config.get('provider', 'mock')
            model = context.config.get('model')
            llm = get_llm(provider_name, model=model)
            context.llm_provider = llm

        # Create MCTS instance
        mcts = (
            ReasoningMCTS()
            .with_llm(llm)
            .with_question(question)
            .with_exploration(context.config.get('exploration', 1.414))
            .with_max_rollout_depth(context.config.get('rollout_depth', 5))
        )

        # Check for compositional actions
        if kwargs.get('compositional', context.config.get('compositional', True)):
            mcts.with_compositional_actions(enabled=True)

        # Check for RAG
        rag_name = kwargs.get('rag')
        if rag_name and context.rag_store:
            mcts.with_rag_store(context.rag_store)

        # Initialize with initial state
        initial_state = f"Question: {question}\n\nLet me think through this step by step:"

        return MCTSStream(mcts, metadata={'question': question, 'state': 'initialized'})

    def get_help(self) -> str:
        return """ask - Create a new reasoning task

Usage:
  ask <question>              Create reasoning task
  ask <question> --rag math   Use RAG guidance
  ask <question> --compositional false  Disable compositional actions

Examples:
  ask "What are the prime numbers less than 100?"
  ask "Solve x^2 + 5x + 6 = 0" --rag math

Starts a new MCTS reasoning session for the given question.
Outputs an MCTSStream that can be piped to search, explore, etc.
"""


class SearchCommand(CommandBase):
    """Run MCTS search simulations."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        # Get simulations count
        if not args:
            simulations = 100
        else:
            try:
                simulations = int(args[0])
            except ValueError:
                raise CommandError(f"Invalid simulation count: {args[0]}")

        # Get MCTS from input stream
        if not context.input_stream:
            raise CommandError("search requires an MCTS tree as input. Use: ask <question> | search <N>")

        if not isinstance(context.input_stream, MCTSStream):
            raise CommandError("search requires MCTSStream input")

        mcts = context.input_stream.mcts

        # Run search
        initial_state = kwargs.get('state') or f"Question: {mcts.original_question}\n\nLet me reason through this:"

        mcts.search(initial_state, simulations=simulations)

        # Update metadata
        metadata = context.input_stream.metadata.copy()
        metadata['simulations'] = simulations
        metadata['state'] = 'searched'

        return MCTSStream(mcts, metadata=metadata)

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """search - Run MCTS search simulations

Usage:
  search [N]                  Run N simulations (default: 100)
  search 50                   Run 50 simulations

Examples:
  ask "problem" | search 100
  load tree.json | search 50

Runs MCTS search on the input tree, exploring reasoning paths.
Outputs the updated MCTSStream.
"""


class ExploreCommand(CommandBase):
    """Explore tree to a specific depth."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        # Get depth
        if not args:
            depth = 5
        else:
            try:
                depth = int(args[0])
            except ValueError:
                raise CommandError(f"Invalid depth: {args[0]}")

        # Get MCTS from input
        if not context.input_stream or not isinstance(context.input_stream, MCTSStream):
            raise CommandError("explore requires MCTSStream input")

        mcts = context.input_stream.mcts

        # Run search until we reach desired depth
        # Heuristic: run simulations proportional to depth
        simulations = depth * 20

        initial_state = f"Question: {mcts.original_question}\n\nLet me reason:"
        mcts.search(initial_state, simulations=simulations)

        metadata = context.input_stream.metadata.copy()
        metadata['explored_depth'] = depth
        metadata['state'] = 'explored'

        return MCTSStream(mcts, metadata=metadata)

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """explore - Explore tree to specific depth

Usage:
  explore [depth]             Explore to depth (default: 5)

Examples:
  ask "problem" | explore 10
  load tree.json | explore 3

Runs enough MCTS simulations to explore to the specified depth.
"""


class SampleCommand(CommandBase):
    """Sample paths from the MCTS tree."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        # Get number of samples
        n = int(args[0]) if args else 1

        # Get strategy
        strategy = kwargs.get('strategy', 'value')
        temperature = float(kwargs.get('temperature', 1.0))

        # Get MCTS from input
        if not context.input_stream or not isinstance(context.input_stream, MCTSStream):
            raise CommandError("sample requires MCTSStream input")

        mcts = context.input_stream.mcts

        if not mcts.root:
            raise CommandError("Cannot sample from empty tree. Run search first.")

        # Sample paths
        paths = mcts.sample(n=n, temperature=temperature, strategy=strategy)

        # Ensure paths is a list
        if not isinstance(paths, list):
            paths = [paths]

        return PathStream(paths, source_mcts=mcts)

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """sample - Sample paths from MCTS tree

Usage:
  sample [N]                  Sample N paths (default: 1)
  sample 5 --strategy value   Sample by value (default)
  sample 5 --strategy visits  Sample by visit count
  sample 5 --strategy diverse Sample diverse paths
  sample 3 --temperature 0.5  Control randomness (0=greedy, higher=random)

Examples:
  ask "problem" | search 100 | sample 5
  load tree.json | sample 3 --strategy diverse

Samples reasoning paths from the tree using specified strategy.
Outputs a PathStream.
"""


class BestCommand(CommandBase):
    """Get the best solution from tree or paths."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("best requires input stream")

        if isinstance(context.input_stream, MCTSStream):
            # Get best path from MCTS tree
            mcts = context.input_stream.mcts
            if not mcts.root:
                raise CommandError("Empty tree")

            solution = mcts.solution
            confidence = mcts.best_value

            metadata = {'confidence': confidence, 'source': 'mcts_best'}
            return SolutionStream([solution], [metadata])

        elif isinstance(context.input_stream, PathStream):
            # Get best path from sampled paths
            paths = list(context.input_stream.paths)
            if not paths:
                raise CommandError("No paths to select from")

            # Find path with highest value
            best_path = max(paths, key=lambda p: p.total_value / max(p.total_visits, 1))

            metadata = {
                'value': best_path.total_value / max(best_path.total_visits, 1),
                'visits': best_path.total_visits,
                'source': 'path_best'
            }
            return SolutionStream([best_path.final_state], [metadata])

        else:
            raise CommandError(f"best cannot process {type(context.input_stream).__name__}")

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """best - Get the best solution

Usage:
  best                        Get best solution from tree or paths

Examples:
  ask "problem" | search 100 | best
  load tree.json | sample 10 | best

Returns the highest-value solution from the input.
Works with MCTSStream or PathStream inputs.
Outputs a SolutionStream.
"""


class WorstCommand(CommandBase):
    """Get the worst solution (for debugging)."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("worst requires input stream")

        if isinstance(context.input_stream, PathStream):
            paths = list(context.input_stream.paths)
            if not paths:
                raise CommandError("No paths")

            worst_path = min(paths, key=lambda p: p.total_value / max(p.total_visits, 1))

            metadata = {
                'value': worst_path.total_value / max(worst_path.total_visits, 1),
                'visits': worst_path.total_visits
            }
            return SolutionStream([worst_path.final_state], [metadata])

        else:
            raise CommandError("worst only works with PathStream")

    def requires_input(self) -> bool:
        return True


class RandomCommand(CommandBase):
    """Get a random solution."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        import random

        if not context.input_stream:
            raise CommandError("random requires input stream")

        if isinstance(context.input_stream, PathStream):
            paths = list(context.input_stream.paths)
            if not paths:
                raise CommandError("No paths")

            random_path = random.choice(paths)

            metadata = {
                'value': random_path.total_value / max(random_path.total_visits, 1),
                'visits': random_path.total_visits
            }
            return SolutionStream([random_path.final_state], [metadata])

        else:
            raise CommandError("random only works with PathStream")

    def requires_input(self) -> bool:
        return True
