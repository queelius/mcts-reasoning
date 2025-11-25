"""
Benchmarking Suite for Standard Datasets

Provides comprehensive benchmarking capabilities for evaluating MCTS reasoning
system performance on standard datasets with quantitative metrics.
"""

from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, asdict
import time
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class BenchmarkProblem:
    """A single benchmark problem."""

    id: str
    category: str
    question: str
    answer: str  # Ground truth answer
    difficulty: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class BenchmarkResult:
    """Results for a single problem."""

    problem_id: str
    category: str
    question: str
    ground_truth: str
    predicted_answer: Optional[str]
    is_correct: bool
    confidence: float
    time_seconds: float
    simulations: int
    tree_stats: Dict[str, Any]
    llm_calls: int
    error: Optional[str] = None

    def to_dict(self):
        return asdict(self)


@dataclass
class BenchmarkSummary:
    """Summary statistics for benchmark run."""

    total_problems: int
    correct: int
    incorrect: int
    errors: int
    accuracy: float
    avg_time: float
    avg_simulations: float
    avg_llm_calls: float
    category_breakdown: Dict[str, Dict[str, Any]]

    def to_dict(self):
        return asdict(self)


class AnswerExtractor:
    """Extracts and normalizes answers from reasoning output."""

    @staticmethod
    def extract_answer(state: str, format_hint: Optional[str] = None) -> str:
        """
        Extract final answer from reasoning state.

        Args:
            state: Reasoning state text
            format_hint: Optional hint about answer format (e.g., "number", "yes/no")

        Returns:
            Extracted answer string
        """
        # Look for common answer markers
        markers = [
            "## Final Answer",
            "Final Answer:",
            "The answer is",
            "Therefore,",
            "**Answer:**",
            "ANSWER:",
        ]

        for marker in markers:
            if marker in state:
                # Extract text after marker
                parts = state.split(marker, 1)
                if len(parts) > 1:
                    answer_section = parts[1].strip()
                    # Take first line or sentence
                    lines = answer_section.split('\n')
                    answer = lines[0].strip()
                    # Clean up markdown formatting
                    answer = answer.replace('**', '').replace('*', '')
                    return answer

        # Fallback: take last non-empty line
        lines = [l.strip() for l in state.split('\n') if l.strip()]
        return lines[-1] if lines else ""

    @staticmethod
    def normalize_answer(answer: str, answer_type: Optional[str] = None) -> str:
        """
        Normalize answer for comparison.

        Args:
            answer: Answer string
            answer_type: Type hint (number, boolean, text, etc.)

        Returns:
            Normalized answer
        """
        answer = answer.strip().lower()

        # Remove common prefixes
        prefixes = ["the answer is", "answer:", "it is", "therefore"]
        for prefix in prefixes:
            if answer.startswith(prefix):
                answer = answer[len(prefix):].strip()

        # Remove punctuation at end
        answer = answer.rstrip('.,;:!?')

        # Normalize numbers
        if answer_type == "number":
            # Extract just the number
            import re
            numbers = re.findall(r'-?\d+\.?\d*', answer)
            if numbers:
                answer = numbers[0]

        return answer.strip()


class AnswerChecker:
    """Checks if predicted answer matches ground truth."""

    @staticmethod
    def check_answer(predicted: str, ground_truth: str,
                    answer_type: Optional[str] = None,
                    tolerance: float = 1e-6) -> bool:
        """
        Check if predicted answer matches ground truth.

        Args:
            predicted: Predicted answer
            ground_truth: Ground truth answer
            answer_type: Type hint for comparison
            tolerance: Numerical tolerance for float comparison

        Returns:
            True if answers match
        """
        # Normalize both
        extractor = AnswerExtractor()
        pred_norm = extractor.normalize_answer(predicted, answer_type)
        truth_norm = extractor.normalize_answer(ground_truth, answer_type)

        # Exact match
        if pred_norm == truth_norm:
            return True

        # Numerical comparison
        if answer_type == "number":
            try:
                pred_val = float(pred_norm)
                truth_val = float(truth_norm)
                return abs(pred_val - truth_val) < tolerance
            except:
                pass

        # Contains match (predicted contains ground truth)
        if truth_norm in pred_norm:
            return True

        return False


class BenchmarkRunner:
    """
    Runs benchmarks on MCTS reasoning system.

    Supports multiple configurations, ablation studies, and detailed reporting.
    """

    def __init__(self, llm_provider=None):
        """
        Initialize benchmark runner.

        Args:
            llm_provider: LLM provider to use (can be set per run)
        """
        self.llm_provider = llm_provider
        self.answer_extractor = AnswerExtractor()
        self.answer_checker = AnswerChecker()
        self._llm_call_counter = 0

    def run_benchmark(
        self,
        problems: List[BenchmarkProblem],
        mcts_config: Dict[str, Any],
        simulations: int = 50,
        max_problems: Optional[int] = None,
        verbose: bool = True
    ) -> tuple[List[BenchmarkResult], BenchmarkSummary]:
        """
        Run benchmark on a set of problems.

        Args:
            problems: List of benchmark problems
            mcts_config: MCTS configuration dict
            simulations: Number of MCTS simulations per problem
            max_problems: Maximum number of problems to run (None = all)
            verbose: Print progress

        Returns:
            (results, summary) tuple
        """
        from mcts_reasoning import ReasoningMCTS

        if max_problems:
            problems = problems[:max_problems]

        results = []

        for i, problem in enumerate(problems):
            if verbose:
                print(f"\n[{i+1}/{len(problems)}] {problem.category}: {problem.question[:60]}...")

            result = self._run_single_problem(
                problem,
                mcts_config,
                simulations,
                verbose=verbose
            )
            results.append(result)

            if verbose:
                status = "✓" if result.is_correct else "✗"
                print(f"  {status} Answer: {result.predicted_answer}")
                print(f"  Time: {result.time_seconds:.1f}s, LLM calls: {result.llm_calls}")

        # Generate summary
        summary = self._generate_summary(results)

        return results, summary

    def _run_single_problem(
        self,
        problem: BenchmarkProblem,
        mcts_config: Dict[str, Any],
        simulations: int,
        verbose: bool = False
    ) -> BenchmarkResult:
        """Run MCTS on a single problem."""
        from mcts_reasoning import ReasoningMCTS

        start_time = time.time()
        self._llm_call_counter = 0

        try:
            # Create MCTS instance with config
            mcts = ReasoningMCTS()

            # Apply configuration
            if 'llm' in mcts_config:
                mcts.with_llm(mcts_config['llm'])
            elif self.llm_provider:
                mcts.with_llm(self.llm_provider)

            mcts.with_question(problem.question)

            # Apply all config options
            if mcts_config.get('compositional', True):
                mcts.with_compositional_actions(enabled=True)

            if 'rag_store' in mcts_config:
                mcts.with_rag_store(mcts_config['rag_store'])

            if mcts_config.get('meta_reasoning', False):
                mcts.with_meta_reasoning(enabled=True)

            if mcts_config.get('reflection', False):
                mcts.with_reflection(enabled=True)

            if mcts_config.get('solution_detection', True):
                mcts.with_solution_detection(enabled=True)

            if mcts_config.get('learning', False):
                mcts.with_learning(enabled=True)

            if mcts_config.get('context_management', False):
                mcts.with_context_config(auto_configure=True)

            # Set MCTS parameters
            mcts.with_exploration(mcts_config.get('exploration', 1.414))
            mcts.with_max_rollout_depth(mcts_config.get('max_depth', 4))

            # Run search
            mcts.search(f"Let's solve: {problem.question}", simulations=simulations)

            # Get solution
            solution = mcts.solution

            # Extract answer
            predicted = self.answer_extractor.extract_answer(solution)

            # Check correctness
            answer_type = problem.metadata.get('answer_type') if problem.metadata else None
            is_correct = self.answer_checker.check_answer(
                predicted,
                problem.answer,
                answer_type=answer_type
            )

            # Get tree stats
            nodes = mcts.get_all_nodes()
            best_node = max(nodes, key=lambda n: n.value/n.visits if n.visits > 0 else 0)

            tree_stats = {
                'total_nodes': len(nodes),
                'max_depth': max(n.depth for n in nodes),
                'best_value': best_node.value / best_node.visits if best_node.visits > 0 else 0,
            }

            # Count LLM calls (rough estimate based on tree size and features)
            llm_calls = len(nodes)  # Base: one per node
            if mcts_config.get('meta_reasoning'):
                llm_calls += len(nodes)  # One suggestion per action
            if mcts_config.get('reflection'):
                llm_calls += len(nodes)  # One critique per action
            if mcts_config.get('solution_detection'):
                llm_calls += len(nodes) // 3  # Periodic detection

            elapsed = time.time() - start_time

            return BenchmarkResult(
                problem_id=problem.id,
                category=problem.category,
                question=problem.question,
                ground_truth=problem.answer,
                predicted_answer=predicted,
                is_correct=is_correct,
                confidence=tree_stats['best_value'],
                time_seconds=elapsed,
                simulations=simulations,
                tree_stats=tree_stats,
                llm_calls=llm_calls,
                error=None
            )

        except Exception as e:
            elapsed = time.time() - start_time
            logger.error(f"Benchmark failed for {problem.id}: {e}")

            return BenchmarkResult(
                problem_id=problem.id,
                category=problem.category,
                question=problem.question,
                ground_truth=problem.answer,
                predicted_answer=None,
                is_correct=False,
                confidence=0.0,
                time_seconds=elapsed,
                simulations=simulations,
                tree_stats={},
                llm_calls=0,
                error=str(e)
            )

    def _generate_summary(self, results: List[BenchmarkResult]) -> BenchmarkSummary:
        """Generate summary statistics."""
        total = len(results)
        correct = sum(1 for r in results if r.is_correct)
        errors = sum(1 for r in results if r.error is not None)
        incorrect = total - correct - errors

        accuracy = correct / total if total > 0 else 0.0

        # Average metrics
        valid_results = [r for r in results if r.error is None]
        avg_time = sum(r.time_seconds for r in valid_results) / len(valid_results) if valid_results else 0
        avg_sims = sum(r.simulations for r in valid_results) / len(valid_results) if valid_results else 0
        avg_calls = sum(r.llm_calls for r in valid_results) / len(valid_results) if valid_results else 0

        # Category breakdown
        categories = {}
        for result in results:
            cat = result.category
            if cat not in categories:
                categories[cat] = {
                    'total': 0,
                    'correct': 0,
                    'accuracy': 0.0
                }

            categories[cat]['total'] += 1
            if result.is_correct:
                categories[cat]['correct'] += 1

        for cat in categories:
            cat_total = categories[cat]['total']
            cat_correct = categories[cat]['correct']
            categories[cat]['accuracy'] = cat_correct / cat_total if cat_total > 0 else 0.0

        return BenchmarkSummary(
            total_problems=total,
            correct=correct,
            incorrect=incorrect,
            errors=errors,
            accuracy=accuracy,
            avg_time=avg_time,
            avg_simulations=avg_sims,
            avg_llm_calls=avg_calls,
            category_breakdown=categories
        )

    def save_results(self, results: List[BenchmarkResult],
                    summary: BenchmarkSummary,
                    output_path: str):
        """Save benchmark results to JSON file."""
        data = {
            'summary': summary.to_dict(),
            'results': [r.to_dict() for r in results]
        }

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2)

        logger.info(f"Results saved to {output_path}")

    def compare_configurations(
        self,
        problems: List[BenchmarkProblem],
        configurations: Dict[str, Dict[str, Any]],
        simulations: int = 50,
        max_problems: Optional[int] = None
    ) -> Dict[str, tuple]:
        """
        Compare multiple MCTS configurations.

        Args:
            problems: List of problems
            configurations: Dict of {name: config} to compare
            simulations: Simulations per problem
            max_problems: Max problems to test

        Returns:
            Dict of {name: (results, summary)} for each configuration
        """
        comparison = {}

        for config_name, config in configurations.items():
            print(f"\n{'='*80}")
            print(f"Running Configuration: {config_name}")
            print(f"{'='*80}")

            results, summary = self.run_benchmark(
                problems,
                config,
                simulations=simulations,
                max_problems=max_problems
            )

            comparison[config_name] = (results, summary)

            print(f"\nResults for {config_name}:")
            print(f"  Accuracy: {summary.accuracy:.1%}")
            print(f"  Avg Time: {summary.avg_time:.1f}s")
            print(f"  Avg LLM Calls: {summary.avg_llm_calls:.0f}")

        return comparison


__all__ = [
    'BenchmarkProblem',
    'BenchmarkResult',
    'BenchmarkSummary',
    'BenchmarkRunner',
    'AnswerExtractor',
    'AnswerChecker'
]
