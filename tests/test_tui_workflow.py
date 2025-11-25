#!/usr/bin/env python3
"""
Simulate the exact TUI workflow to debug the issue.
"""

from mcts_reasoning.compositional.providers import OllamaProvider
from mcts_reasoning.reasoning import ReasoningMCTS

def test_greeting_question():
    """Test with greeting (should not work well)."""
    print("=" * 70)
    print("Test 1: Greeting Question (Expected to fail)")
    print("=" * 70)

    # Create provider
    provider = OllamaProvider(
        model="llama3.2",
        base_url="http://192.168.0.225:11434"
    )

    # Create MCTS (mimicking TUI's initialize_mcts)
    mcts = (
        ReasoningMCTS()
        .with_llm(provider)
        .with_question("hi")
        .with_exploration(1.414)
        .with_max_rollout_depth(5)
        .with_compositional_actions(enabled=True)
    )

    # Initial state (mimicking TUI)
    question = "hi"
    initial_state = f"Question: {question}\n\nLet me think about this systematically."

    print(f"Question: {question}")
    print(f"Initial state:\n{initial_state}")
    print()

    # Run search
    print("Running 10 simulations...")
    mcts.search(initial_state, simulations=10)

    # Show stats
    stats = mcts.stats
    print(f"\nTree statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Best value: {stats['best_value']:.3f}")

    # Show solution
    solution = mcts.solution
    print(f"\nBest solution ({len(solution)} chars):")
    print("=" * 70)
    print(solution)
    print("=" * 70)
    print()


def test_reasoning_question():
    """Test with actual reasoning question (should work well)."""
    print("\n" + "=" * 70)
    print("Test 2: Reasoning Question (Should work)")
    print("=" * 70)

    # Create provider
    provider = OllamaProvider(
        model="llama3.2",
        base_url="http://192.168.0.225:11434"
    )

    # Create MCTS
    mcts = (
        ReasoningMCTS()
        .with_llm(provider)
        .with_question("What are the prime numbers less than 20?")
        .with_exploration(1.414)
        .with_max_rollout_depth(5)
        .with_compositional_actions(enabled=True)
    )

    # Initial state
    question = "What are the prime numbers less than 20?"
    initial_state = f"Question: {question}\n\nLet me think about this systematically."

    print(f"Question: {question}")
    print(f"Initial state:\n{initial_state}")
    print()

    # Run search
    print("Running 10 simulations...")
    mcts.search(initial_state, simulations=10)

    # Show stats
    stats = mcts.stats
    print(f"\nTree statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Best value: {stats['best_value']:.3f}")

    # Show solution
    solution = mcts.solution
    print(f"\nBest solution ({len(solution)} chars):")
    print("=" * 70)
    print(solution)
    print("=" * 70)
    print()


def test_with_simple_actions():
    """Test with simple actions instead of compositional."""
    print("\n" + "=" * 70)
    print("Test 3: Simple Actions (No compositional)")
    print("=" * 70)

    # Create provider
    provider = OllamaProvider(
        model="llama3.2",
        base_url="http://192.168.0.225:11434"
    )

    # Create MCTS without compositional actions
    mcts = (
        ReasoningMCTS()
        .with_llm(provider)
        .with_question("What are the prime numbers less than 20?")
        .with_exploration(1.414)
        .with_max_rollout_depth(5)
        .with_compositional_actions(enabled=False)  # Disable compositional
        .with_actions([
            "Think about the definition of prime numbers",
            "List numbers from 2 to 19",
            "Check each number for primality",
            "Provide the final answer"
        ])
    )

    # Initial state
    question = "What are the prime numbers less than 20?"
    initial_state = f"Question: {question}\n\nLet me solve this step by step."

    print(f"Question: {question}")
    print(f"Using simple actions (not compositional)")
    print()

    # Run search
    print("Running 10 simulations...")
    mcts.search(initial_state, simulations=10)

    # Show stats
    stats = mcts.stats
    print(f"\nTree statistics:")
    print(f"  Total nodes: {stats['total_nodes']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Best value: {stats['best_value']:.3f}")

    # Show solution
    solution = mcts.solution
    print(f"\nBest solution ({len(solution)} chars):")
    print("=" * 70)
    print(solution)
    print("=" * 70)
    print()


if __name__ == "__main__":
    import time

    # Test 1: Greeting (bad question)
    test_greeting_question()
    time.sleep(1)

    # Test 2: Actual reasoning question with compositional actions
    test_reasoning_question()
    time.sleep(1)

    # Test 3: Simple actions
    test_with_simple_actions()
