"""
Tests for the CLI rewrite (v0.6 Task 14).

Tests the argument parser structure and subcommand help output.
Uses subprocess to test the CLI as a user would invoke it.
"""

from __future__ import annotations

import subprocess
import sys

import pytest


def _run_cli(*args: str) -> subprocess.CompletedProcess:
    """Run the CLI via ``python -m mcts_reasoning.cli`` with given args."""
    return subprocess.run(
        [sys.executable, "-m", "mcts_reasoning.cli", *args],
        capture_output=True,
        text=True,
        timeout=30,
    )


# ---------------------------------------------------------------------------
# Top-level help
# ---------------------------------------------------------------------------


def test_cli_help():
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "search" in result.stdout
    assert "explore" in result.stdout
    assert "bench" in result.stdout


def test_cli_no_args_prints_help():
    result = _run_cli()
    assert result.returncode == 0
    # Should print help (contains subcommand names)
    assert "search" in result.stdout


# ---------------------------------------------------------------------------
# search subcommand
# ---------------------------------------------------------------------------


def test_cli_search_help():
    result = _run_cli("search", "--help")
    assert result.returncode == 0
    assert "question" in result.stdout
    assert "--simulations" in result.stdout
    assert "--provider" in result.stdout
    assert "--exploration" in result.stdout
    assert "--json" in result.stdout
    assert "--save" in result.stdout


def test_cli_search_with_mock():
    """Run a real (mock) search and check output."""
    result = _run_cli("search", "What is 2+2?", "--provider", "mock", "--simulations", "3")
    assert result.returncode == 0
    # Should output some answer-related text
    assert "Answer" in result.stdout or "answer" in result.stdout or "{" in result.stdout


def test_cli_search_json_output():
    result = _run_cli(
        "search", "What is 2+2?", "--provider", "mock", "--simulations", "3", "--json"
    )
    assert result.returncode == 0
    # Should be valid JSON
    import json

    data = json.loads(result.stdout)
    assert isinstance(data, dict)
    assert "answer" in data or "error" in data


def test_cli_search_backward_compat():
    """Bare question (no subcommand) should be treated as search."""
    result = _run_cli("What is 2+2?", "--provider", "mock", "--simulations", "3", "--json")
    assert result.returncode == 0
    import json

    data = json.loads(result.stdout)
    assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# explore subcommand
# ---------------------------------------------------------------------------


def test_cli_explore_help():
    result = _run_cli("explore", "--help")
    assert result.returncode == 0
    assert "question" in result.stdout
    assert "--simulations" in result.stdout


def test_cli_explore_with_mock():
    result = _run_cli(
        "explore", "What is 2+2?", "--provider", "mock", "--simulations", "3"
    )
    assert result.returncode == 0
    import json

    data = json.loads(result.stdout)
    assert "tree" in data or "error" in data


# ---------------------------------------------------------------------------
# bench subcommand
# ---------------------------------------------------------------------------


def test_cli_bench_help():
    result = _run_cli("bench", "--help")
    assert result.returncode == 0
    assert "--benchmark" in result.stdout
    assert "--simulations" in result.stdout
    assert "--format" in result.stdout
    assert "--output" in result.stdout


def test_cli_bench_json():
    result = _run_cli(
        "bench",
        "--benchmark", "arithmetic",
        "--provider", "mock",
        "--simulations", "3",
        "--format", "json",
    )
    assert result.returncode == 0
    import json

    data = json.loads(result.stdout)
    assert "benchmark_name" in data or "error" in data


def test_cli_bench_table():
    result = _run_cli(
        "bench",
        "--benchmark", "arithmetic",
        "--provider", "mock",
        "--simulations", "3",
        "--format", "table",
    )
    assert result.returncode == 0
    assert "Solver" in result.stdout or "Benchmark" in result.stdout


def test_cli_bench_csv(tmp_path):
    csv_path = str(tmp_path / "results.csv")
    result = _run_cli(
        "bench",
        "--benchmark", "arithmetic",
        "--provider", "mock",
        "--simulations", "3",
        "--format", "csv",
        "--output", csv_path,
    )
    assert result.returncode == 0
    import csv

    with open(csv_path, newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    assert len(rows) > 0
    assert "solver" in rows[0]


def test_cli_bench_csv_without_output_errors():
    result = _run_cli(
        "bench",
        "--benchmark", "arithmetic",
        "--provider", "mock",
        "--simulations", "3",
        "--format", "csv",
    )
    # Should fail because --output is required for csv
    assert result.returncode != 0


# ---------------------------------------------------------------------------
# Parser structure
# ---------------------------------------------------------------------------


def test_parser_structure():
    """Test that create_parser returns a parser with the expected subcommands."""
    from mcts_reasoning.cli import create_parser

    parser = create_parser()
    # Parsing known subcommands should succeed
    args = parser.parse_args(["search", "question"])
    assert args.command == "search"
    assert args.question == "question"

    args = parser.parse_args(["explore", "question"])
    assert args.command == "explore"
    assert args.question == "question"

    args = parser.parse_args(["bench", "--benchmark", "knights"])
    assert args.command == "bench"
    assert args.benchmark == "knights"


def test_parser_defaults():
    """Test parser defaults for search subcommand."""
    from mcts_reasoning.cli import create_parser

    parser = create_parser()
    args = parser.parse_args(["search", "Q?"])
    assert args.simulations == 10
    assert args.exploration == pytest.approx(1.414)
    assert args.provider is None
    assert args.model is None
    assert args.json is False
