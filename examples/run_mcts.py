#!/usr/bin/env python3
"""
Example: Run MCTS on a real math problem.

Usage:
    python examples/run_mcts.py [--provider ollama|openai|anthropic|mock]

Environment variables:
    OLLAMA_BASE_URL: Ollama server URL (default: http://localhost:11434)
    OPENAI_API_KEY: OpenAI API key
    ANTHROPIC_API_KEY: Anthropic API key
"""

import argparse
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcts_reasoning import (
    MCTS,
    LLMGenerator,
    NumericEvaluator,
    ProcessEvaluator,
    get_llm,
    MockLLMProvider,
)


def main():
    parser = argparse.ArgumentParser(description="Run MCTS on a math problem")
    parser.add_argument(
        "--provider",
        choices=["ollama", "openai", "anthropic", "mock"],
        default=None,
        help="LLM provider (default: auto-detect)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name (provider-specific)",
    )
    parser.add_argument(
        "--simulations",
        type=int,
        default=10,
        help="Number of MCTS simulations (default: 10)",
    )
    parser.add_argument(
        "--question",
        default="What is 15 * 7 + 23?",
        help="Question to solve",
    )
    parser.add_argument(
        "--answer",
        type=float,
        default=None,
        help="Ground truth answer (for evaluation)",
    )
    args = parser.parse_args()

    # Get LLM provider
    print("=" * 60)
    print("MCTS-Reasoning Demo")
    print("=" * 60)

    try:
        if args.provider:
            kwargs = {"model": args.model} if args.model else {}
            llm = get_llm(args.provider, **kwargs)
        else:
            llm = get_llm()
        print(f"Provider: {llm.get_provider_name()}")
    except Exception as e:
        print(f"Could not initialize LLM provider: {e}")
        print("Falling back to mock provider...")
        llm = MockLLMProvider()

    # Check if provider is available
    print("Checking availability...")
    if not llm.is_available():
        print(f"Provider {llm.get_provider_name()} is not available")
        print("Falling back to mock provider...")
        llm = MockLLMProvider()

    print(f"Using: {llm.get_provider_name()}")
    print()

    # Set up generator
    generator = LLMGenerator(
        llm=llm,
        temperature=0.7,
        max_tokens=500,
    )

    # Set up evaluator
    if args.answer is not None:
        # Use numeric evaluator with ground truth
        evaluator = ProcessEvaluator(
            answer_evaluator=NumericEvaluator(
                ground_truth=args.answer,
                rel_tol=0.01,
            ),
            answer_weight=0.7,
            process_weight=0.3,
        )
        print(f"Ground truth: {args.answer}")
    else:
        # Calculate ground truth for default question
        if args.question == "What is 15 * 7 + 23?":
            ground_truth = 15 * 7 + 23  # = 128
            evaluator = ProcessEvaluator(
                answer_evaluator=NumericEvaluator(
                    ground_truth=ground_truth,
                    rel_tol=0.01,
                ),
                answer_weight=0.7,
                process_weight=0.3,
            )
            print(f"Ground truth: {ground_truth}")
        else:
            # No ground truth - use process evaluator only
            evaluator = ProcessEvaluator()
            print("No ground truth provided - evaluating process only")

    print()

    # Create MCTS
    mcts = MCTS(
        generator=generator,
        evaluator=evaluator,
        exploration_constant=1.414,
        max_children_per_node=3,
        max_rollout_depth=5,
    )

    # Run search
    print(f"Question: {args.question}")
    print(f"Running {args.simulations} simulations...")
    print()

    result = mcts.search(
        question=args.question,
        simulations=args.simulations,
    )

    # Show results
    print("=" * 60)
    print("Results")
    print("=" * 60)
    print(f"Best answer: {result.best_answer}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Simulations: {result.simulations}")
    print(f"Terminal states found: {len(result.terminal_states)}")
    print()

    # Show tree stats
    stats = result.stats
    print("Tree Statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Terminal states: {stats['terminal_states_found']}")
    print()

    # Show best path
    if result.best_answer:
        print("Best reasoning path:")
        print("-" * 40)
        # Find the terminal node with the answer
        best_node = None
        best_value = -1
        for node in mcts.root.children:
            terminals = _find_terminals(node)
            for t in terminals:
                if t.answer == result.best_answer and t.average_value > best_value:
                    best_node = t
                    best_value = t.average_value

        if best_node:
            path = best_node.path_from_root()
            for i, node in enumerate(path):
                if i == 0:
                    continue  # Skip root
                # Show just the added content
                prev_state = path[i - 1].state
                added = node.state[len(prev_state):].strip()
                print(f"Step {i}: {added[:200]}{'...' if len(added) > 200 else ''}")
                print()

    # Visualize tree (simplified)
    print("Tree structure (top 3 levels):")
    print("-" * 40)
    _print_tree(mcts.root, max_depth=3)


def _find_terminals(node):
    """Find all terminal nodes under this node."""
    terminals = []
    if node.is_terminal:
        terminals.append(node)
    for child in node.children:
        terminals.extend(_find_terminals(child))
    return terminals


def _print_tree(node, depth=0, max_depth=3, prefix=""):
    """Print tree structure."""
    if depth > max_depth:
        return

    # Node info
    visits = node.visits
    value = node.average_value
    terminal = "T" if node.is_terminal else ""
    answer = f" [{node.answer}]" if node.answer else ""

    if depth == 0:
        print(f"Root (visits={visits}, value={value:.2f})")
    else:
        state_preview = node.state[-50:].replace("\n", " ")
        print(f"{prefix}├── ...{state_preview} (v={visits}, val={value:.2f}){terminal}{answer}")

    # Children
    for i, child in enumerate(node.children):
        is_last = i == len(node.children) - 1
        child_prefix = prefix + ("    " if is_last else "│   ")
        _print_tree(child, depth + 1, max_depth, child_prefix)


if __name__ == "__main__":
    main()
