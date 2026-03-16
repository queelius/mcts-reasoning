#!/usr/bin/env python3
"""
MCTS-Reasoning CLI: Run Monte Carlo Tree Search reasoning from the command line.

Subcommands:
    search   -- Run MCTS search on a question
    explore  -- Run MCTS and show the full reasoning tree
    bench    -- Run benchmark comparison (baseline vs MCTS)

Usage:
    mcts-reason search "What is 2+2?"
    mcts-reason search "Solve: 15*7+23" --simulations 20 --provider ollama
    mcts-reason explore "Hard problem" --simulations 30 --json
    mcts-reason bench --benchmark arithmetic --simulations 10,20
    mcts-reason --help

Backward compatibility:
    mcts-reason "What is 2+2?"   (treated as 'search')

Environment variables:
    LLM_PROVIDER: Default provider (openai, anthropic, ollama, mock)
    OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
    OPENAI_API_KEY: OpenAI API key
    ANTHROPIC_API_KEY: Anthropic API key
"""

from __future__ import annotations

import argparse
import json
import sys


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser with search/explore/bench subcommands."""
    parser = argparse.ArgumentParser(
        prog="mcts-reason",
        description="MCTS for LLM reasoning",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            '  mcts-reason search "What is 2+2?"\n'
            '  mcts-reason search "15*7+23" --simulations 20 --provider ollama\n'
            '  mcts-reason explore "Hard problem" --simulations 30 --json\n'
            "  mcts-reason bench --benchmark arithmetic --simulations 10,20\n"
        ),
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # -----------------------------------------------------------------
    # search
    # -----------------------------------------------------------------
    search_p = subparsers.add_parser(
        "search",
        help="Run MCTS search on a question",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    search_p.add_argument("question", help="The question to solve")
    search_p.add_argument(
        "--provider",
        default=None,
        help="LLM provider (openai, anthropic, ollama, mock; default: auto-detect)",
    )
    search_p.add_argument("--model", default=None, help="Model name")
    search_p.add_argument(
        "--simulations",
        type=int,
        default=10,
        help="Number of simulations (default: 10)",
    )
    search_p.add_argument(
        "--exploration",
        type=float,
        default=1.414,
        help="UCB1 exploration constant (default: 1.414)",
    )
    search_p.add_argument("--json", action="store_true", help="JSON output")
    search_p.add_argument(
        "--base-url",
        default=None,
        help="Provider base URL (for Ollama or OpenAI-compatible APIs)",
    )
    search_p.add_argument(
        "--save", default=None, metavar="FILE", help="Save search state to file"
    )

    # -----------------------------------------------------------------
    # explore
    # -----------------------------------------------------------------
    explore_p = subparsers.add_parser(
        "explore",
        help="Run MCTS and show the full reasoning tree",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    explore_p.add_argument("question", help="The question to explore")
    explore_p.add_argument("--provider", default=None)
    explore_p.add_argument("--model", default=None)
    explore_p.add_argument("--simulations", type=int, default=10)
    explore_p.add_argument("--json", action="store_true", help="JSON output (default)")
    explore_p.add_argument("--base-url", default=None)

    # -----------------------------------------------------------------
    # bench
    # -----------------------------------------------------------------
    bench_p = subparsers.add_parser(
        "bench",
        help="Run benchmark comparison (baseline vs MCTS)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    bench_p.add_argument(
        "--benchmark",
        default="knights",
        help="Benchmark name (knights, arithmetic; default: knights)",
    )
    bench_p.add_argument("--provider", default=None)
    bench_p.add_argument("--model", default=None)
    bench_p.add_argument(
        "--simulations",
        default="10",
        help="Comma-separated simulation counts (default: 10)",
    )
    bench_p.add_argument(
        "--format",
        choices=["table", "json", "csv"],
        default="table",
        help="Output format (default: table)",
    )
    bench_p.add_argument(
        "--output",
        default=None,
        metavar="FILE",
        help="Output file (required for csv format)",
    )
    bench_p.add_argument("--base-url", default=None)

    return parser


# ---------------------------------------------------------------------------
# Subcommand handlers
# ---------------------------------------------------------------------------


def _handle_search(args) -> None:
    from .server.tools import mcts_search_impl

    provider_name = args.provider or "auto"
    result = mcts_search_impl(
        args.question,
        provider_name,
        args.model,
        args.simulations,
        args.exploration,
    )

    if args.json or "error" in result:
        print(json.dumps(result, indent=2))
    else:
        if result.get("answer"):
            print(f"Answer: {result['answer']}")
            print(f"Confidence: {result.get('confidence', 0):.1%}")
            print(f"Simulations: {result['simulations']}")
            print(f"Terminal states: {result['terminal_states']}")
            print(f"Tree nodes: {result['tree_nodes']}")
        else:
            print("No answer found.")
            print(json.dumps(result, indent=2))


def _handle_explore(args) -> None:
    from .server.tools import mcts_explore_impl

    provider_name = args.provider or "auto"
    result = mcts_explore_impl(
        args.question,
        provider_name,
        args.model,
        args.simulations,
    )
    print(json.dumps(result, indent=2))


def _handle_bench(args) -> None:
    from .server.tools import mcts_bench_impl

    provider_name = args.provider or "auto"
    sim_list = [int(s.strip()) for s in args.simulations.split(",")]
    result = mcts_bench_impl(args.benchmark, provider_name, args.model, sim_list)

    if "error" in result:
        print(json.dumps(result, indent=2), file=sys.stderr)
        sys.exit(1)

    fmt = args.format

    if fmt == "json":
        print(json.dumps(result, indent=2))

    elif fmt == "csv":
        if not args.output:
            print("Error: --output is required for csv format", file=sys.stderr)
            sys.exit(1)
        _write_bench_csv(result, args.output)
        print(f"Results written to {args.output}")

    else:
        # table (default)
        _print_bench_table(result)


def _print_bench_table(data: dict) -> None:
    """Render a plain-text comparison table from a bench report dict."""
    print(f"Benchmark: {data['benchmark_name']}")
    sep = "-" * 65
    print(sep)
    row = "{:<30} {:>5} {:>10} {:>11} {:>13}"
    print(row.format("Solver", "N", "Accuracy", "Avg Score", "Avg Time (s)"))
    print(sep)
    for solver_name, solver_data in data.get("solvers", {}).items():
        n = solver_data["n"]
        acc = solver_data["accuracy"]
        results = solver_data.get("results", [])
        avg_score = sum(r["score"] for r in results) / n if n else 0.0
        avg_time = sum(r["time_seconds"] for r in results) / n if n else 0.0
        print(
            row.format(
                solver_name[:30],
                n,
                f"{acc:.1%}",
                f"{avg_score:.3f}",
                f"{avg_time:.3f}",
            )
        )
    print(sep)


def _write_bench_csv(data: dict, path: str) -> None:
    """Write per-problem benchmark results to a CSV file."""
    import csv

    fieldnames = [
        "solver",
        "question",
        "ground_truth",
        "domain",
        "difficulty",
        "answer",
        "correct",
        "score",
        "time_seconds",
    ]
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for solver_name, solver_data in data.get("solvers", {}).items():
            for result in solver_data.get("results", []):
                writer.writerow(
                    {
                        "solver": solver_name,
                        "question": result.get("question", ""),
                        "ground_truth": result.get("ground_truth", ""),
                        "domain": result.get("domain", ""),
                        "difficulty": result.get("difficulty", ""),
                        "answer": result.get("answer", ""),
                        "correct": result.get("correct", ""),
                        "score": result.get("score", ""),
                        "time_seconds": result.get("time_seconds", ""),
                    }
                )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point for the mcts-reason CLI."""
    parser = create_parser()

    # Backward compat: if the first positional arg is not a known subcommand,
    # treat the invocation as ``mcts-reason search <question>``.
    # We must do this *before* parse_args() because argparse will reject
    # unknown subcommand names.
    known_commands = {"search", "explore", "bench", "--help", "-h"}
    if len(sys.argv) > 1 and sys.argv[1] not in known_commands:
        sys.argv.insert(1, "search")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    handlers = {
        "search": _handle_search,
        "explore": _handle_explore,
        "bench": _handle_bench,
    }
    handler = handlers.get(args.command)
    if handler:
        handler(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
