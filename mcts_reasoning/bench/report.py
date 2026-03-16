"""
BenchReport: Collects and analyses benchmark results.

Provides accuracy breakdowns by domain and difficulty, lift computation
versus a baseline, and serialization to table/JSON/CSV.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from typing import Optional

from .solver import SolverResult
from .benchmark import Problem


@dataclass
class BenchReport:
    """
    Holds raw results and provides analysis helpers.

    ``results`` maps solver name -> list of (Problem, SolverResult) tuples.
    """

    benchmark_name: str
    results: dict[str, list[tuple[Problem, SolverResult]]] = field(
        default_factory=dict
    )

    # ------------------------------------------------------------------
    # Accuracy helpers
    # ------------------------------------------------------------------

    def accuracy(self, solver: str) -> float:
        """Overall fraction of problems answered correctly."""
        pairs = self.results.get(solver, [])
        if not pairs:
            return 0.0
        return sum(1 for _, r in pairs if r.correct) / len(pairs)

    def accuracy_by_domain(self, solver: str) -> dict[str, float]:
        """Per-domain accuracy fractions."""
        pairs = self.results.get(solver, [])
        domains: dict[str, list[bool]] = {}
        for problem, result in pairs:
            domains.setdefault(problem.domain, []).append(result.correct)
        return {
            domain: sum(flags) / len(flags) for domain, flags in domains.items()
        }

    def accuracy_by_difficulty(self, solver: str) -> dict[str, float]:
        """Per-difficulty accuracy fractions."""
        pairs = self.results.get(solver, [])
        buckets: dict[str, list[bool]] = {}
        for problem, result in pairs:
            buckets.setdefault(problem.difficulty, []).append(result.correct)
        return {
            diff: sum(flags) / len(flags) for diff, flags in buckets.items()
        }

    def lift(self, baseline: str, experimental: str) -> float:
        """
        Absolute accuracy lift of *experimental* over *baseline*.

        Positive values mean the experimental solver is better.
        """
        return self.accuracy(experimental) - self.accuracy(baseline)

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_table(self) -> str:
        """
        Render a plain-text comparison table.

        Columns: Solver | N | Accuracy | Avg Score | Avg Time (s)
        """
        if not self.results:
            return f"Benchmark: {self.benchmark_name}\n(no results)"

        header = f"Benchmark: {self.benchmark_name}\n"
        sep = "-" * 65 + "\n"
        row_fmt = "{:<30} {:>5} {:>10} {:>11} {:>13}\n"
        header_row = row_fmt.format(
            "Solver", "N", "Accuracy", "Avg Score", "Avg Time (s)"
        )
        lines = header + sep + header_row + sep

        for solver_name, pairs in self.results.items():
            n = len(pairs)
            if n == 0:
                lines += row_fmt.format(solver_name, 0, "N/A", "N/A", "N/A")
                continue
            acc = self.accuracy(solver_name)
            avg_score = sum(r.score for _, r in pairs) / n
            avg_time = sum(r.time_seconds for _, r in pairs) / n
            lines += row_fmt.format(
                solver_name[:30],
                n,
                f"{acc:.1%}",
                f"{avg_score:.3f}",
                f"{avg_time:.3f}",
            )

        lines += sep
        return lines

    def to_json(self) -> str:
        """Serialise to a JSON string."""
        data: dict = {
            "benchmark_name": self.benchmark_name,
            "solvers": {},
        }
        for solver_name, pairs in self.results.items():
            data["solvers"][solver_name] = {
                "accuracy": self.accuracy(solver_name),
                "accuracy_by_domain": self.accuracy_by_domain(solver_name),
                "accuracy_by_difficulty": self.accuracy_by_difficulty(solver_name),
                "n": len(pairs),
                "results": [
                    {
                        "question": p.question,
                        "ground_truth": p.ground_truth,
                        "domain": p.domain,
                        "difficulty": p.difficulty,
                        "answer": r.answer,
                        "correct": r.correct,
                        "score": r.score,
                        "time_seconds": r.time_seconds,
                        "metadata": r.metadata,
                    }
                    for p, r in pairs
                ],
            }
        return json.dumps(data, indent=2)

    def to_csv(self, path: str) -> None:
        """Write per-problem results to a CSV file at *path*."""
        with open(path, "w", newline="") as fh:
            writer = csv.DictWriter(
                fh,
                fieldnames=[
                    "solver",
                    "question",
                    "ground_truth",
                    "domain",
                    "difficulty",
                    "answer",
                    "correct",
                    "score",
                    "time_seconds",
                ],
            )
            writer.writeheader()
            for solver_name, pairs in self.results.items():
                for problem, result in pairs:
                    writer.writerow(
                        {
                            "solver": solver_name,
                            "question": problem.question,
                            "ground_truth": problem.ground_truth,
                            "domain": problem.domain,
                            "difficulty": problem.difficulty,
                            "answer": result.answer,
                            "correct": result.correct,
                            "score": result.score,
                            "time_seconds": result.time_seconds,
                        }
                    )
