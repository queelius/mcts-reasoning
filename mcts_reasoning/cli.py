#!/usr/bin/env python3
"""
MCTS-Reasoning CLI: Run Monte Carlo Tree Search reasoning from the command line.

Usage:
    mcts-reason "What is 2+2?"
    mcts-reason "Solve: 15*7+23" --simulations 20 --provider ollama
    mcts-reason --help

Environment variables:
    LLM_PROVIDER: Default provider (openai, anthropic, ollama, mock)
    OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
    OPENAI_API_KEY: OpenAI API key
    ANTHROPIC_API_KEY: Anthropic API key
"""

import argparse
import json
import sys
from typing import Optional

from .mcts import MCTS, SearchResult
from .generator import LLMGenerator
from .evaluator import (
    NumericEvaluator,
    ProcessEvaluator,
    LLMEvaluator,
    MockEvaluator,
)
from .sampling import PathSampler
from .compositional.providers import get_llm, MockLLMProvider


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser."""
    parser = argparse.ArgumentParser(
        prog="mcts-reason",
        description="Run MCTS-based reasoning on a question",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  mcts-reason "What is 2+2?"
  mcts-reason "Solve: 15*7+23" --answer 128
  mcts-reason "Explain photosynthesis" --simulations 20
  mcts-reason "Complex problem" --provider ollama --model llama3.2
        """,
    )

    parser.add_argument(
        "question",
        nargs="?",
        help="Question to solve (or use --question)",
    )

    parser.add_argument(
        "-q", "--question",
        dest="question_flag",
        help="Question to solve (alternative to positional arg)",
    )

    # Provider settings
    parser.add_argument(
        "-p", "--provider",
        choices=["openai", "anthropic", "ollama", "mock"],
        default=None,
        help="LLM provider (default: auto-detect)",
    )

    parser.add_argument(
        "-m", "--model",
        default=None,
        help="Model name (provider-specific)",
    )

    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)",
    )

    # Search settings
    parser.add_argument(
        "-n", "--simulations",
        type=int,
        default=10,
        help="Number of MCTS simulations (default: 10)",
    )

    parser.add_argument(
        "-d", "--max-depth",
        type=int,
        default=5,
        help="Maximum rollout depth (default: 5)",
    )

    parser.add_argument(
        "-b", "--branching",
        type=int,
        default=3,
        help="Maximum children per node (default: 3)",
    )

    parser.add_argument(
        "-c", "--exploration",
        type=float,
        default=1.414,
        help="UCB1 exploration constant (default: 1.414)",
    )

    # Evaluation settings
    parser.add_argument(
        "-a", "--answer",
        type=float,
        default=None,
        help="Ground truth numeric answer (for evaluation)",
    )

    # Output settings
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output (show tree structure)",
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )

    parser.add_argument(
        "--sample",
        type=int,
        default=0,
        metavar="N",
        help="Sample N diverse paths from tree",
    )

    parser.add_argument(
        "--consistency",
        action="store_true",
        help="Show answer consistency score",
    )

    return parser


def get_provider(args):
    """Get LLM provider based on args."""
    try:
        if args.provider:
            kwargs = {"model": args.model} if args.model else {}
            llm = get_llm(args.provider, **kwargs)
        else:
            llm = get_llm()

        # Check availability
        if not llm.is_available():
            print(f"Warning: {llm.get_provider_name()} not available, using mock",
                  file=sys.stderr)
            return MockLLMProvider()
        return llm

    except Exception as e:
        print(f"Warning: Could not initialize provider: {e}", file=sys.stderr)
        print("Using mock provider", file=sys.stderr)
        return MockLLMProvider()


def get_evaluator(args, llm):
    """Get evaluator based on args."""
    if args.answer is not None:
        # Use numeric evaluator with ground truth
        return ProcessEvaluator(
            answer_evaluator=NumericEvaluator(
                ground_truth=args.answer,
                rel_tol=0.01,
            ),
            answer_weight=0.7,
            process_weight=0.3,
        )
    else:
        # Use process evaluator only
        return ProcessEvaluator()


def run_search(args, question: str) -> dict:
    """Run MCTS search and return results dict."""
    # Get provider
    llm = get_provider(args)

    # Create generator and evaluator
    generator = LLMGenerator(
        llm=llm,
        temperature=args.temperature,
        max_tokens=500,
    )
    evaluator = get_evaluator(args, llm)

    # Create MCTS
    mcts = MCTS(
        generator=generator,
        evaluator=evaluator,
        exploration_constant=args.exploration,
        max_children_per_node=args.branching,
        max_rollout_depth=args.max_depth,
    )

    # Run search
    result = mcts.search(
        question=question,
        simulations=args.simulations,
    )

    # Build results dict
    output = {
        "question": question,
        "answer": result.best_answer,
        "confidence": result.confidence,
        "simulations": result.simulations,
        "terminal_states": len(result.terminal_states),
        "stats": result.stats,
        "provider": llm.get_provider_name(),
    }

    # Add ground truth comparison if provided
    if args.answer is not None:
        output["ground_truth"] = args.answer
        try:
            predicted = float(result.best_answer.replace("$", "").replace(",", ""))
            output["correct"] = abs(predicted - args.answer) < 0.01 * abs(args.answer)
        except (ValueError, TypeError, AttributeError):
            output["correct"] = False

    # Add sampling results if requested
    if args.sample > 0:
        sampler = PathSampler(result.root)
        paths = sampler.sample(n=args.sample, strategy="diverse")
        output["sampled_paths"] = [
            {
                "answer": p.answer,
                "value": p.value,
                "steps": p.steps,
            }
            for p in paths
        ]

    # Add consistency if requested
    if args.consistency:
        sampler = PathSampler(result.root)
        output["consistency"] = sampler.consistency_score()
        output["answer_distribution"] = {
            str(k): {"count": v["count"], "avg_value": v["avg_value"]}
            for k, v in sampler.get_answer_distribution().items()
        }

    return output, result


def print_results(output: dict, result: SearchResult, args):
    """Print results in human-readable format."""
    print("=" * 60)
    print("MCTS-Reasoning Results")
    print("=" * 60)
    print(f"Question: {output['question']}")
    print(f"Provider: {output['provider']}")
    print()
    print(f"Best Answer: {output['answer']}")
    print(f"Confidence: {output['confidence']:.1%}")
    print(f"Simulations: {output['simulations']}")
    print(f"Terminal States: {output['terminal_states']}")

    if "ground_truth" in output:
        status = "CORRECT" if output.get("correct") else "INCORRECT"
        print(f"Ground Truth: {output['ground_truth']} ({status})")

    if "consistency" in output:
        print(f"Consistency: {output['consistency']:.1%}")

    # Tree stats
    if args.verbose:
        print()
        print("Tree Statistics:")
        stats = output["stats"]
        print(f"  Total nodes: {stats['total_nodes']}")
        print(f"  Max depth: {stats['max_depth']}")

    # Answer distribution
    if args.consistency and "answer_distribution" in output:
        print()
        print("Answer Distribution:")
        for answer, info in output["answer_distribution"].items():
            print(f"  {answer}: {info['count']} occurrences (avg value: {info['avg_value']:.2f})")

    # Sampled paths
    if args.sample > 0 and "sampled_paths" in output:
        print()
        print(f"Sampled Paths ({len(output['sampled_paths'])} diverse):")
        for i, path in enumerate(output["sampled_paths"], 1):
            print(f"\n  Path {i} (answer: {path['answer']}, value: {path['value']:.2f}):")
            for j, step in enumerate(path["steps"][:3], 1):
                step_preview = step[:100].replace("\n", " ")
                print(f"    Step {j}: {step_preview}...")

    # Verbose tree visualization
    if args.verbose:
        print()
        print("Tree Structure (top 3 levels):")
        print("-" * 40)
        _print_tree(result.root, max_depth=3)


def _print_tree(node, depth=0, max_depth=3, prefix=""):
    """Print tree structure."""
    if depth > max_depth:
        return

    visits = node.visits
    value = node.average_value
    terminal = "T" if node.is_terminal else ""
    answer = f" [{node.answer}]" if node.answer else ""

    if depth == 0:
        print(f"Root (visits={visits}, value={value:.2f})")
    else:
        state_preview = node.state[-40:].replace("\n", " ")
        print(f"{prefix}├── ...{state_preview} (v={visits}, val={value:.2f}){terminal}{answer}")

    for i, child in enumerate(node.children):
        is_last = i == len(node.children) - 1
        child_prefix = prefix + ("    " if is_last else "│   ")
        _print_tree(child, depth + 1, max_depth, child_prefix)


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Get question from args
    question = args.question or args.question_flag
    if not question:
        parser.print_help()
        print("\nError: Question is required", file=sys.stderr)
        sys.exit(1)

    # Run search
    output, result = run_search(args, question)

    # Output results
    if args.json:
        # JSON output - exclude non-serializable fields
        json_output = {k: v for k, v in output.items() if k != "stats"}
        json_output["stats"] = {
            k: v for k, v in output["stats"].items()
            if k not in ("best_answer",)  # Already in output
        }
        print(json.dumps(json_output, indent=2))
    else:
        print_results(output, result, args)


if __name__ == "__main__":
    main()
