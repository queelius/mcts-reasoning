"""
Tests for the benchmarking framework (v0.6 Tasks 11-12).

All tests use mock solvers — no LLM calls are made.
"""

from __future__ import annotations

import csv
import json
import os
import tempfile

import pytest

from mcts_reasoning.bench import (
    Benchmark,
    BenchReport,
    BenchRunner,
    Problem,
    Solver,
    SolverResult,
)
from mcts_reasoning.bench.benchmarks import (
    ArithmeticChains,
    KnightsAndKnaves,
    get_benchmark,
)


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


class AlwaysCorrectSolver(Solver):
    """Returns the ground truth for every problem."""

    @property
    def name(self) -> str:
        return "always_correct"

    def solve(self, problem: Problem) -> SolverResult:
        return SolverResult(
            answer=problem.ground_truth,
            correct=True,
            score=1.0,
            time_seconds=0.001,
        )


class AlwaysWrongSolver(Solver):
    """Returns a nonsense answer for every problem."""

    @property
    def name(self) -> str:
        return "always_wrong"

    def solve(self, problem: Problem) -> SolverResult:
        return SolverResult(
            answer="WRONG",
            correct=False,
            score=0.0,
            time_seconds=0.001,
        )


class HalfCorrectSolver(Solver):
    """Correct on even-indexed problems (by insertion order), wrong on odd."""

    def __init__(self):
        self._call_count = 0

    @property
    def name(self) -> str:
        return "half_correct"

    def solve(self, problem: Problem) -> SolverResult:
        correct = self._call_count % 2 == 0
        self._call_count += 1
        return SolverResult(
            answer=problem.ground_truth if correct else "WRONG",
            correct=correct,
            score=1.0 if correct else 0.0,
            time_seconds=0.001,
        )


def _make_tiny_benchmark(n: int = 4, domain: str = "test") -> Benchmark:
    """Return a minimal concrete Benchmark with *n* problems."""

    class TinyBenchmark(Benchmark):
        @property
        def name(self) -> str:
            return "tiny"

        def problems(self) -> list[Problem]:
            return [
                Problem(
                    question=f"Q{i}",
                    ground_truth=f"A{i}",
                    domain=domain,
                    difficulty="easy" if i % 2 == 0 else "medium",
                )
                for i in range(n)
            ]

    return TinyBenchmark()


# ---------------------------------------------------------------------------
# BenchRunner tests
# ---------------------------------------------------------------------------


def test_bench_runner_runs_all_problems():
    benchmark = _make_tiny_benchmark(n=4)
    runner = BenchRunner()
    report = runner.run(benchmark, [AlwaysCorrectSolver()])
    pairs = report.results["always_correct"]
    assert len(pairs) == 4


def test_bench_runner_runs_multiple_solvers():
    benchmark = _make_tiny_benchmark(n=3)
    runner = BenchRunner()
    report = runner.run(benchmark, [AlwaysCorrectSolver(), AlwaysWrongSolver()])
    assert "always_correct" in report.results
    assert "always_wrong" in report.results


def test_bench_runner_each_problem_paired():
    """Each entry in results is a (Problem, SolverResult) pair."""
    benchmark = _make_tiny_benchmark(n=2)
    runner = BenchRunner()
    report = runner.run(benchmark, [AlwaysCorrectSolver()])
    for pair in report.results["always_correct"]:
        problem, result = pair
        assert isinstance(problem, Problem)
        assert isinstance(result, SolverResult)


# ---------------------------------------------------------------------------
# BenchReport accuracy tests
# ---------------------------------------------------------------------------


def test_bench_report_accuracy_100():
    benchmark = _make_tiny_benchmark(n=4)
    runner = BenchRunner()
    report = runner.run(benchmark, [AlwaysCorrectSolver()])
    assert report.accuracy("always_correct") == pytest.approx(1.0)


def test_bench_report_accuracy_0():
    benchmark = _make_tiny_benchmark(n=4)
    runner = BenchRunner()
    report = runner.run(benchmark, [AlwaysWrongSolver()])
    assert report.accuracy("always_wrong") == pytest.approx(0.0)


def test_bench_report_accuracy_partial():
    """HalfCorrectSolver should yield 50% accuracy on an even-sized benchmark."""
    benchmark = _make_tiny_benchmark(n=4)
    runner = BenchRunner()
    report = runner.run(benchmark, [HalfCorrectSolver()])
    assert report.accuracy("half_correct") == pytest.approx(0.5)


def test_bench_report_accuracy_unknown_solver():
    report = BenchReport(benchmark_name="x", results={})
    assert report.accuracy("nonexistent") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Accuracy by domain / difficulty
# ---------------------------------------------------------------------------


def test_bench_report_accuracy_by_domain():
    class MixedBenchmark(Benchmark):
        @property
        def name(self):
            return "mixed"

        def problems(self):
            return [
                Problem("Q1", "A", domain="math", difficulty="easy"),
                Problem("Q2", "B", domain="math", difficulty="easy"),
                Problem("Q3", "C", domain="logic", difficulty="hard"),
            ]

    runner = BenchRunner()
    solver = AlwaysCorrectSolver()
    report = runner.run(MixedBenchmark(), [solver])
    by_domain = report.accuracy_by_domain("always_correct")
    assert by_domain["math"] == pytest.approx(1.0)
    assert by_domain["logic"] == pytest.approx(1.0)


def test_bench_report_accuracy_by_difficulty():
    benchmark = _make_tiny_benchmark(n=4)  # alternates easy/medium
    runner = BenchRunner()
    report = runner.run(benchmark, [AlwaysCorrectSolver()])
    by_diff = report.accuracy_by_difficulty("always_correct")
    assert by_diff["easy"] == pytest.approx(1.0)
    assert by_diff["medium"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Lift
# ---------------------------------------------------------------------------


def test_bench_report_lift():
    benchmark = _make_tiny_benchmark(n=4)
    runner = BenchRunner()
    report = runner.run(benchmark, [AlwaysCorrectSolver(), AlwaysWrongSolver()])
    lift_value = report.lift("always_wrong", "always_correct")
    assert lift_value == pytest.approx(1.0)


def test_bench_report_lift_negative():
    benchmark = _make_tiny_benchmark(n=4)
    runner = BenchRunner()
    report = runner.run(benchmark, [AlwaysCorrectSolver(), AlwaysWrongSolver()])
    # wrong compared to correct = negative lift
    assert report.lift("always_correct", "always_wrong") == pytest.approx(-1.0)


def test_bench_report_lift_zero():
    benchmark = _make_tiny_benchmark(n=4)
    runner = BenchRunner()
    report = runner.run(benchmark, [AlwaysCorrectSolver()])
    assert report.lift("always_correct", "always_correct") == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# to_table
# ---------------------------------------------------------------------------


def test_bench_report_to_table():
    benchmark = _make_tiny_benchmark(n=4)
    runner = BenchRunner()
    report = runner.run(benchmark, [AlwaysCorrectSolver(), AlwaysWrongSolver()])
    table = report.to_table()
    assert isinstance(table, str)
    assert "tiny" in table
    assert "always_correct" in table
    assert "always_wrong" in table
    assert "100.0%" in table
    assert "0.0%" in table


def test_bench_report_to_table_empty():
    report = BenchReport(benchmark_name="empty", results={})
    table = report.to_table()
    assert "empty" in table


# ---------------------------------------------------------------------------
# to_json
# ---------------------------------------------------------------------------


def test_bench_report_to_json():
    benchmark = _make_tiny_benchmark(n=2)
    runner = BenchRunner()
    report = runner.run(benchmark, [AlwaysCorrectSolver()])
    payload = json.loads(report.to_json())
    assert payload["benchmark_name"] == "tiny"
    assert "always_correct" in payload["solvers"]
    solver_data = payload["solvers"]["always_correct"]
    assert solver_data["accuracy"] == pytest.approx(1.0)
    assert solver_data["n"] == 2
    assert len(solver_data["results"]) == 2


def test_bench_report_to_json_contains_per_problem_fields():
    benchmark = _make_tiny_benchmark(n=1)
    runner = BenchRunner()
    report = runner.run(benchmark, [AlwaysCorrectSolver()])
    payload = json.loads(report.to_json())
    result = payload["solvers"]["always_correct"]["results"][0]
    assert "question" in result
    assert "ground_truth" in result
    assert "answer" in result
    assert "correct" in result
    assert "score" in result
    assert "time_seconds" in result


# ---------------------------------------------------------------------------
# to_csv
# ---------------------------------------------------------------------------


def test_bench_report_to_csv():
    benchmark = _make_tiny_benchmark(n=3)
    runner = BenchRunner()
    report = runner.run(benchmark, [AlwaysCorrectSolver()])
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", delete=False
    ) as fh:
        csv_path = fh.name

    try:
        report.to_csv(csv_path)
        with open(csv_path, newline="") as fh:
            reader = csv.DictReader(fh)
            rows = list(reader)
        assert len(rows) == 3
        assert rows[0]["solver"] == "always_correct"
        assert rows[0]["correct"] == "True"
    finally:
        os.unlink(csv_path)


# ---------------------------------------------------------------------------
# KnightsAndKnaves benchmark
# ---------------------------------------------------------------------------


def test_knights_benchmark_has_problems():
    bench = KnightsAndKnaves()
    problems = bench.problems()
    assert len(problems) == 15


def test_knights_benchmark_name():
    assert KnightsAndKnaves().name == "knights_and_knaves"


def test_knights_benchmark_problem_structure():
    for p in KnightsAndKnaves().problems():
        assert isinstance(p, Problem)
        assert p.question
        assert p.ground_truth in {
            "A is a knight, B is a knight",
            "A is a knight, B is a knave",
            "A is a knave, B is a knight",
            "A is a knave, B is a knave",
        }, f"Unexpected ground_truth: {p.ground_truth!r}"
        assert p.domain == "logic"
        assert p.difficulty in {"easy", "medium", "hard"}


def test_knights_benchmark_difficulty_distribution():
    problems = KnightsAndKnaves().problems()
    difficulties = [p.difficulty for p in problems]
    assert difficulties.count("easy") == 5
    assert difficulties.count("medium") == 5
    assert difficulties.count("hard") == 5


# ---------------------------------------------------------------------------
# ArithmeticChains benchmark
# ---------------------------------------------------------------------------


def test_arithmetic_benchmark_has_problems():
    bench = ArithmeticChains()
    problems = bench.problems()
    assert len(problems) == 20


def test_arithmetic_benchmark_name():
    assert ArithmeticChains().name == "arithmetic_chains"


def test_arithmetic_benchmark_problem_structure():
    for p in ArithmeticChains().problems():
        assert isinstance(p, Problem)
        assert p.question
        assert p.ground_truth.lstrip("-").isdigit(), (
            f"Expected integer ground_truth, got {p.ground_truth!r}"
        )
        assert p.domain == "arithmetic"
        assert p.difficulty in {"easy", "medium", "hard"}


def test_arithmetic_benchmark_difficulty_distribution():
    problems = ArithmeticChains().problems()
    difficulties = [p.difficulty for p in problems]
    assert difficulties.count("easy") == 7
    assert difficulties.count("medium") == 7
    assert difficulties.count("hard") == 6


def test_arithmetic_benchmark_ground_truth_non_negative_where_expected():
    """Spot-check a few known answers."""
    problems = {p.metadata["expression"]: p for p in ArithmeticChains().problems()}
    assert problems["(3 + 7) * 4"].ground_truth == "40"
    assert problems["(15 - 6) * 3"].ground_truth == "27"
    assert problems["(9 + 11) * (6 - 2)"].ground_truth == "80"


# ---------------------------------------------------------------------------
# get_benchmark factory
# ---------------------------------------------------------------------------


def test_get_benchmark_by_name_knights():
    bench = get_benchmark("knights")
    assert isinstance(bench, KnightsAndKnaves)


def test_get_benchmark_by_name_arithmetic():
    bench = get_benchmark("arithmetic")
    assert isinstance(bench, ArithmeticChains)


def test_get_benchmark_unknown_name_raises():
    with pytest.raises(KeyError, match="Unknown benchmark"):
        get_benchmark("nonexistent")


# ---------------------------------------------------------------------------
# Integration: BenchRunner + built-in benchmark + mock solver
# ---------------------------------------------------------------------------


def test_runner_with_knights_benchmark():
    bench = KnightsAndKnaves()
    runner = BenchRunner()
    report = runner.run(bench, [AlwaysCorrectSolver()])
    assert report.accuracy("always_correct") == pytest.approx(1.0)
    assert len(report.results["always_correct"]) == 15


def test_runner_with_arithmetic_benchmark():
    bench = ArithmeticChains()
    runner = BenchRunner()
    report = runner.run(bench, [AlwaysCorrectSolver()])
    assert report.accuracy("always_correct") == pytest.approx(1.0)
    assert len(report.results["always_correct"]) == 20


def test_full_pipeline_table_and_json():
    bench = _make_tiny_benchmark(n=4)
    runner = BenchRunner()
    report = runner.run(bench, [AlwaysCorrectSolver(), HalfCorrectSolver()])
    table = report.to_table()
    payload = json.loads(report.to_json())

    assert "tiny" in table
    assert payload["benchmark_name"] == "tiny"
    assert payload["solvers"]["always_correct"]["accuracy"] == pytest.approx(1.0)
    assert payload["solvers"]["half_correct"]["accuracy"] == pytest.approx(0.5)
