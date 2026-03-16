"""
BenchRunner: Runs benchmarks against one or more solvers.

Iterates over all problems in a benchmark and collects SolverResults for each
solver.  Returns a BenchReport ready for analysis.
"""

from __future__ import annotations

from .benchmark import Benchmark
from .solver import Solver


class BenchRunner:
    """Orchestrates running a Benchmark against a list of Solvers."""

    def run(self, benchmark: Benchmark, solvers: list[Solver]) -> "BenchReport":
        """
        Run all solvers against all problems.

        Args:
            benchmark: The benchmark to evaluate against.
            solvers: List of solvers to compare.

        Returns:
            BenchReport with results keyed by solver name.
        """
        from .report import BenchReport

        results: dict = {}
        for solver in solvers:
            solver_results = []
            for problem in benchmark.problems():
                result = solver.solve(problem)
                solver_results.append((problem, result))
            results[solver.name] = solver_results

        return BenchReport(benchmark_name=benchmark.name, results=results)
