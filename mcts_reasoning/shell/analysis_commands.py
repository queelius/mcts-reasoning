"""
Analysis and introspection commands.

Commands:
- stats: Show tree/path statistics
- tree: Display tree structure
- verify: Verify solution correctness
- consistency: Check consistency across samples
- diff: Compare paths or solutions
"""

from typing import List, Dict, Any

from .command_base import CommandBase, CommandContext, CommandError
from .streams import Stream, MCTSStream, PathStream, SolutionStream, TextStream, StatsStream


class StatsCommand(CommandBase):
    """Show statistics about the input stream."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("stats requires input stream")

        stats = {}

        if isinstance(context.input_stream, MCTSStream):
            mcts = context.input_stream.mcts
            stats = mcts.stats

            # Add derived stats
            if hasattr(mcts, 'reasoning_depth'):
                stats['reasoning_depth'] = mcts.reasoning_depth
            if hasattr(mcts, 'exploration_breadth'):
                stats['exploration_breadth'] = mcts.exploration_breadth

        elif isinstance(context.input_stream, PathStream):
            paths = list(context.input_stream.paths)

            values = [p.total_value / max(p.total_visits, 1) for p in paths]
            visits = [p.total_visits for p in paths]
            lengths = [p.length for p in paths]

            stats = {
                'num_paths': len(paths),
                'value': {
                    'mean': sum(values) / len(values) if values else 0,
                    'min': min(values) if values else 0,
                    'max': max(values) if values else 0,
                },
                'visits': {
                    'mean': sum(visits) / len(visits) if visits else 0,
                    'min': min(visits) if visits else 0,
                    'max': max(visits) if visits else 0,
                },
                'length': {
                    'mean': sum(lengths) / len(lengths) if lengths else 0,
                    'min': min(lengths) if lengths else 0,
                    'max': max(lengths) if lengths else 0,
                }
            }

        elif isinstance(context.input_stream, SolutionStream):
            solutions = context.input_stream.solutions
            stats = {
                'num_solutions': len(solutions),
                'total_chars': sum(len(s) for s in solutions),
                'avg_length': sum(len(s) for s in solutions) / len(solutions) if solutions else 0,
                'unique_solutions': len(set(solutions))
            }

        else:
            stats = {'type': type(context.input_stream).__name__}

        return StatsStream(stats)

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """stats - Show statistics

Usage:
  stats                       Show stream statistics

Examples:
  ask "problem" | search 100 | stats
  sample 50 | stats
  load tree.json | stats

Displays detailed statistics about the input stream.
"""


class TreeCommand(CommandBase):
    """Display tree structure."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("tree requires input stream")

        if not isinstance(context.input_stream, MCTSStream):
            raise CommandError("tree only works with MCTSStream")

        mcts = context.input_stream.mcts

        if not mcts.root:
            return TextStream(["Empty tree"])

        # Get max depth to display
        max_depth = int(args[0]) if args else None

        lines = []

        def print_node(node, depth=0, prefix="", is_last=True):
            if max_depth is not None and depth > max_depth:
                return

            # Node info
            avg_value = node.value / max(node.visits, 1)
            connector = "└── " if is_last else "├── "

            # Truncate state
            state_preview = node.state[:60].replace('\n', ' ')
            if len(node.state) > 60:
                state_preview += "..."

            node_info = f"{connector}{state_preview}"
            node_stats = f" [v={avg_value:.3f}, n={node.visits}, d={depth}]"

            lines.append(f"{prefix}{node_info}{node_stats}")

            # Children
            if node.children and (max_depth is None or depth < max_depth):
                extension = "    " if is_last else "│   "
                for i, child in enumerate(node.children):
                    print_node(child, depth + 1, prefix + extension, i == len(node.children) - 1)

        print_node(mcts.root)
        return TextStream(lines)

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """tree - Display tree structure

Usage:
  tree [max_depth]            Display tree (optionally limit depth)

Examples:
  ask "problem" | search 50 | tree
  load tree.json | tree 3

Displays the MCTS tree structure in ASCII format.
"""


class VerifyCommand(CommandBase):
    """Verify solution correctness using LLM."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("verify requires input stream")

        # Get LLM
        llm = context.llm_provider
        if not llm:
            raise CommandError("No LLM provider configured")

        # Get solutions to verify
        solutions = []

        if isinstance(context.input_stream, SolutionStream):
            solutions = context.input_stream.solutions
        elif isinstance(context.input_stream, MCTSStream):
            solutions = [context.input_stream.mcts.solution]
        elif isinstance(context.input_stream, PathStream):
            paths = list(context.input_stream.paths)
            solutions = [p.final_state for p in paths]
        else:
            raise CommandError("verify requires solutions, paths, or MCTS tree")

        # Verify each solution
        results = []

        for i, solution in enumerate(solutions):
            prompt = f"""
Verify if this solution is correct and complete.

Solution:
{solution}

Is this solution:
1. Logically correct?
2. Complete (fully answers the question)?
3. Well-reasoned?

Respond with:
VERDICT: CORRECT/INCORRECT/PARTIAL
CONFIDENCE: 0.0-1.0
REASONING: Brief explanation

Your response:
"""

            try:
                response = llm.generate(prompt, max_tokens=200)

                # Parse response
                verdict = "UNKNOWN"
                confidence = 0.5
                reasoning = response

                if "VERDICT:" in response:
                    verdict_line = [l for l in response.split('\n') if 'VERDICT:' in l][0]
                    verdict = verdict_line.split('VERDICT:')[1].strip().split()[0]

                if "CONFIDENCE:" in response:
                    conf_line = [l for l in response.split('\n') if 'CONFIDENCE:' in l][0]
                    conf_str = conf_line.split('CONFIDENCE:')[1].strip().split()[0]
                    try:
                        confidence = float(conf_str)
                    except:
                        pass

                results.append({
                    'solution_id': i,
                    'verdict': verdict,
                    'confidence': confidence,
                    'reasoning': reasoning
                })

            except Exception as e:
                results.append({
                    'solution_id': i,
                    'verdict': 'ERROR',
                    'confidence': 0.0,
                    'reasoning': str(e)
                })

        # Format output
        lines = ["Verification Results:", ""]

        for r in results:
            lines.append(f"Solution {r['solution_id']}:")
            lines.append(f"  Verdict: {r['verdict']}")
            lines.append(f"  Confidence: {r['confidence']:.2f}")
            lines.append("")

        return TextStream(lines)

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """verify - Verify solution correctness

Usage:
  verify                      Verify solutions using LLM

Examples:
  ask "problem" | search 100 | best | verify
  sample 5 | verify

Uses LLM to verify if solutions are correct and complete.
"""


class ConsistencyCommand(CommandBase):
    """Check consistency across multiple samples."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("consistency requires input stream")

        # Number of samples to check
        n_samples = int(args[0]) if args else 20

        # Only works with MCTSStream
        if not isinstance(context.input_stream, MCTSStream):
            raise CommandError("consistency only works with MCTSStream")

        mcts = context.input_stream.mcts

        if not mcts.root:
            raise CommandError("Empty tree")

        # Use built-in consistency check
        temperature = float(kwargs.get('temperature', 1.0))

        result = mcts.check_consistency(n_samples=n_samples, temperature=temperature)

        # Format output
        lines = [
            "Consistency Check Results:",
            "",
            f"Samples: {n_samples}",
            f"Confidence: {result['confidence']:.2%}",
            f"Support: {result['support']}/{n_samples}",
            "",
            "Most Consistent Solution:",
            "---",
            result['solution'],
            "---",
            "",
            f"Found {len(result['clusters'])} unique solution clusters"
        ]

        return TextStream(lines)

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """consistency - Check solution consistency

Usage:
  consistency [N]             Check across N samples (default: 20)
  consistency 50 --temperature 0.8

Examples:
  ask "problem" | search 100 | consistency 30
  load tree.json | consistency 50

Samples multiple paths and finds the most consistent solution.
Reports confidence based on how often the same solution appears.
"""


class DiffCommand(CommandBase):
    """Compare paths or solutions."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("diff requires input stream")

        if isinstance(context.input_stream, PathStream):
            paths = list(context.input_stream.paths)

            if len(paths) < 2:
                raise CommandError("Need at least 2 paths to compare")

            # Compare first two paths (or specified indices)
            idx1 = int(kwargs.get('path1', 0))
            idx2 = int(kwargs.get('path2', 1))

            if idx1 >= len(paths) or idx2 >= len(paths):
                raise CommandError(f"Invalid path indices: {idx1}, {idx2}")

            path1 = paths[idx1]
            path2 = paths[idx2]

            lines = [
                f"Comparing Path {idx1} vs Path {idx2}",
                "",
                f"Path {idx1}:",
                f"  Length: {path1.length}",
                f"  Value: {path1.total_value / max(path1.total_visits, 1):.3f}",
                f"  Visits: {path1.total_visits}",
                "",
                f"Path {idx2}:",
                f"  Length: {path2.length}",
                f"  Value: {path2.total_value / max(path2.total_visits, 1):.3f}",
                f"  Visits: {path2.total_visits}",
                "",
                "Action Sequences:",
                f"  Path {idx1}: {' -> '.join(str(a) for a in path1.actions[:5])}...",
                f"  Path {idx2}: {' -> '.join(str(a) for a in path2.actions[:5])}...",
                "",
                "Final States:",
                f"--- Path {idx1} ---",
                path1.final_state[:200],
                "",
                f"--- Path {idx2} ---",
                path2.final_state[:200]
            ]

            return TextStream(lines)

        elif isinstance(context.input_stream, SolutionStream):
            solutions = context.input_stream.solutions

            if len(solutions) < 2:
                raise CommandError("Need at least 2 solutions to compare")

            # Simple character-level diff
            sol1 = solutions[0]
            sol2 = solutions[1]

            lines = [
                "Comparing Solutions:",
                "",
                "Solution 1:",
                sol1[:300],
                "",
                "Solution 2:",
                sol2[:300],
                "",
                f"Length difference: {len(sol1) - len(sol2)} chars",
                f"Character overlap: {self._calc_overlap(sol1, sol2):.1%}"
            ]

            return TextStream(lines)

        else:
            raise CommandError("diff only works with PathStream or SolutionStream")

    def _calc_overlap(self, s1: str, s2: str) -> float:
        """Calculate character overlap between strings."""
        if not s1 or not s2:
            return 0.0

        set1 = set(s1.lower().split())
        set2 = set(s2.lower().split())

        if not set1 or not set2:
            return 0.0

        intersection = set1 & set2
        union = set1 | set2

        return len(intersection) / len(union)

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """diff - Compare paths or solutions

Usage:
  diff                        Compare first two items
  diff --path1 0 --path2 2    Compare specific paths

Examples:
  sample 5 | diff
  sample 10 | filter --min-value 0.8 | diff

Compares reasoning paths or solutions.
"""


class ExplainCommand(CommandBase):
    """Explain the reasoning process."""

    def execute(self, args: List[str], kwargs: Dict[str, Any],
                context: CommandContext) -> Stream:
        if not context.input_stream:
            raise CommandError("explain requires input stream")

        if isinstance(context.input_stream, MCTSStream):
            mcts = context.input_stream.mcts

            if hasattr(mcts, 'explain_reasoning'):
                explanation = mcts.explain_reasoning()
            else:
                # Fallback
                explanation = f"MCTS tree with {mcts.stats['total_nodes']} nodes"

            return TextStream([explanation])

        elif isinstance(context.input_stream, PathStream):
            paths = list(context.input_stream.paths)

            lines = [f"Sampled {len(paths)} reasoning paths:", ""]

            for i, path in enumerate(paths[:10]):  # Limit to first 10
                avg_value = path.total_value / max(path.total_visits, 1)
                lines.append(f"\nPath {i+1}:")
                lines.append(f"  Quality: {avg_value:.3f}")
                lines.append(f"  Steps: {path.length}")
                lines.append(f"  Actions: {' -> '.join(str(a) for a in path.actions)}")

            return TextStream(lines)

        else:
            return TextStream([context.input_stream.to_text()])

    def requires_input(self) -> bool:
        return True

    def get_help(self) -> str:
        return """explain - Explain reasoning process

Usage:
  explain                     Show reasoning explanation

Examples:
  ask "problem" | search 100 | explain
  sample 5 | explain

Provides human-readable explanation of the reasoning process.
"""
