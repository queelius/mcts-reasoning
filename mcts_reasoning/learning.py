"""
Learning from Successful Paths

Automatically improves RAG stores by learning which compositional actions
lead to successful solutions.
"""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PathAnalysis:
    """Analysis of a reasoning path."""

    path_length: int
    operations_used: List[str]
    operation_counts: Dict[str, int]
    focuses_used: List[str]
    styles_used: List[str]
    success_score: float
    is_solution: bool


class PathLearner:
    """
    Learns from successful reasoning paths to improve action selection.

    Analyzes which compositional actions led to good solutions and updates
    RAG store weights accordingly.
    """

    def __init__(self, rag_store=None):
        """
        Initialize path learner.

        Args:
            rag_store: CompositionalRAGStore to update
        """
        self.rag_store = rag_store
        self._learned_patterns = []
        self._learning_count = 0

    def analyze_path(self, path: List[Any], final_value: float,
                    is_solution: bool = False) -> PathAnalysis:
        """
        Analyze a reasoning path.

        Args:
            path: List of (action, state) tuples from root to leaf
            final_value: Final value/score of this path
            is_solution: Whether this path led to a finalized solution

        Returns:
            PathAnalysis with extracted patterns
        """
        from mcts_reasoning.compositional.actions import CompositionalAction

        operations = []
        focuses = []
        styles = []

        for action, _ in path:
            if isinstance(action, CompositionalAction):
                operations.append(action.operation.value)
                focuses.append(action.focus.value)
                styles.append(action.style.value)

        # Count operations
        from collections import Counter
        op_counts = dict(Counter(operations))

        return PathAnalysis(
            path_length=len(path),
            operations_used=operations,
            operation_counts=op_counts,
            focuses_used=focuses,
            styles_used=styles,
            success_score=final_value,
            is_solution=is_solution
        )

    def learn_from_path(self, problem: str, path: List[Any],
                       final_value: float, is_solution: bool = False,
                       learning_rate: float = 0.3):
        """
        Learn from a successful path.

        Args:
            problem: Original problem/question
            path: Reasoning path (list of (action, state) tuples)
            final_value: Final value/score
            is_solution: Whether this led to finalized solution
            learning_rate: How much to adjust weights (0-1)
        """
        if self.rag_store is None:
            logger.warning("No RAG store available for learning")
            return

        # Only learn from reasonably good paths
        if final_value < 0.3 and not is_solution:
            logger.debug(f"Skipping low-value path (value={final_value})")
            return

        analysis = self.analyze_path(path, final_value, is_solution)

        # Solutions are extra valuable
        weight_boost = 2.0 if is_solution else 1.0
        base_weight = final_value * learning_rate * weight_boost

        logger.info(f"Learning from path (value={final_value:.3f}, solution={is_solution})")
        logger.info(f"  Operations: {analysis.operation_counts}")

        # Update RAG store with learned patterns
        try:
            from mcts_reasoning.compositional import CognitiveOperation, FocusAspect, ReasoningStyle
            from mcts_reasoning.compositional.rag import CompositionalGuidance

            # Find or create guidance for this problem pattern
            relevant = self.rag_store.retrieve(problem, k=1)

            if relevant:
                # Update existing guidance
                guidance = relevant[0]
                self._update_guidance_weights(guidance, analysis, base_weight)
            else:
                # Create new guidance from this successful path
                # Extract dominant operations
                top_operations = sorted(
                    analysis.operation_counts.items(),
                    key=lambda x: x[1],
                    reverse=True
                )[:3]

                ops = []
                for op_name, count in top_operations:
                    try:
                        ops.append(CognitiveOperation(op_name))
                    except:
                        pass

                # Create guidance
                keywords = self._extract_keywords(problem)
                guidance = CompositionalGuidance(
                    problem_pattern=problem[:50],  # Truncate for pattern
                    problem_keywords=keywords,
                    recommended_operations=ops if ops else None,
                    domain="learned",
                    success_rate=final_value
                )

                self.rag_store.add_guidance(guidance)
                logger.info(f"Created new guidance pattern from successful path")

            self._learning_count += 1
            self._learned_patterns.append({
                'problem': problem[:100],
                'value': final_value,
                'is_solution': is_solution,
                'operations': analysis.operation_counts
            })

        except Exception as e:
            logger.error(f"Failed to learn from path: {e}")

    def _update_guidance_weights(self, guidance, analysis: PathAnalysis,
                                 weight_boost: float):
        """Update guidance weights based on path analysis."""
        # Initialize weights if not present
        if guidance.weights is None:
            guidance.weights = {}

        # Update operation weights
        if 'cognitive_op' not in guidance.weights:
            guidance.weights['cognitive_op'] = {}

        from mcts_reasoning.compositional import CognitiveOperation

        for op_name, count in analysis.operation_counts.items():
            try:
                op = CognitiveOperation(op_name)
                current_weight = guidance.weights['cognitive_op'].get(op, 1.0)
                # Increase weight based on how often this operation was used
                new_weight = current_weight + (count * weight_boost * 0.5)
                guidance.weights['cognitive_op'][op] = new_weight
            except:
                pass

        # Update success rate (running average)
        n = getattr(guidance, '_learning_count', 0)
        old_rate = guidance.success_rate
        guidance.success_rate = (old_rate * n + analysis.success_score) / (n + 1)
        guidance._learning_count = n + 1

    def _extract_keywords(self, problem: str, max_keywords: int = 5) -> List[str]:
        """Extract keywords from problem statement."""
        # Simple keyword extraction (could be improved with NLP)
        import re

        # Remove common words
        common_words = {'what', 'how', 'why', 'when', 'where', 'who', 'the', 'a', 'an',
                       'is', 'are', 'was', 'were', 'be', 'been', 'being',
                       'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                       'can', 'could', 'should', 'may', 'might', 'must'}

        # Extract words
        words = re.findall(r'\b[a-z]+\b', problem.lower())

        # Filter and take most significant
        keywords = [w for w in words if w not in common_words and len(w) > 3]

        return keywords[:max_keywords]

    def learn_from_tree(self, mcts, problem: str,
                       min_value: float = 0.5,
                       learn_from_solutions: bool = True):
        """
        Learn from all good paths in an MCTS tree.

        Args:
            mcts: MCTS instance with completed search
            problem: Original problem
            min_value: Minimum average value to learn from
            learn_from_solutions: Whether to specially weight finalized solutions
        """
        if not mcts.root:
            logger.warning("No tree to learn from")
            return

        from mcts_reasoning.solution_detection import is_finalized_solution

        nodes = mcts.get_all_nodes()
        learned_count = 0
        good_nodes = 0
        solution_nodes = 0

        for node in nodes:
            # Calculate average value
            avg_value = node.value / node.visits if node.visits > 0 else 0

            # Check if solution
            is_solution = is_finalized_solution(node.state)

            if is_solution:
                solution_nodes += 1

            # Learn from good paths and all solutions
            if avg_value >= min_value or (learn_from_solutions and is_solution):
                good_nodes += 1

                # Get path to this node
                path = []
                current = node
                while current.parent is not None:
                    path.insert(0, (current.action_taken, current.state))
                    current = current.parent

                # Only learn from paths with compositional actions
                if path:  # Skip root node with empty path
                    self.learn_from_path(
                        problem,
                        path,
                        avg_value,
                        is_solution=is_solution
                    )
                    learned_count += 1

        logger.info(f"Learned from {learned_count}/{good_nodes} qualifying paths ({solution_nodes} solutions, {len(nodes)} total nodes)")

    def get_stats(self) -> Dict[str, Any]:
        """Get learning statistics."""
        return {
            'learning_count': self._learning_count,
            'patterns_learned': len(self._learned_patterns),
            'recent_patterns': self._learned_patterns[-5:] if self._learned_patterns else []
        }


__all__ = ['PathLearner', 'PathAnalysis']
