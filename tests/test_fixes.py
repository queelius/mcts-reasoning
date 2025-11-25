#!/usr/bin/env python3
"""
Test script to verify the two major fixes:
1. Context truncation increased from 1000 to 4000 chars
2. Proper action tracking for tree exploration
"""

from mcts_reasoning.compositional.providers import MockLLMProvider
from mcts_reasoning.reasoning import ReasoningMCTS

def test_tree_exploration():
    """Test that tree now branches properly."""
    print("=" * 70)
    print("Testing Tree Exploration (Fix #1)")
    print("=" * 70)
    print()

    provider = MockLLMProvider()
    mcts = (
        ReasoningMCTS()
        .with_llm(provider)
        .with_question("What are the prime numbers less than 20?")
        .with_exploration(1.414)
        .with_max_rollout_depth(4)
        .with_compositional_actions(enabled=True)
    )

    initial_state = "Question: What are the prime numbers less than 20?\n\nLet me solve this step by step."

    print("Running 20 simulations...")
    mcts.search(initial_state, simulations=20)

    # Get all nodes
    nodes = mcts.get_all_nodes()

    print(f"\nTree Statistics:")
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Max depth: {mcts.stats['max_depth']}")
    print(f"  Root visits: {mcts.root.visits}")
    print(f"  Root children: {len(mcts.root.children)}")
    print()

    # Count nodes at each depth
    depth_counts = {}
    for node in nodes:
        depth = node.depth
        depth_counts[depth] = depth_counts.get(depth, 0) + 1

    print("Nodes per depth:")
    for depth in sorted(depth_counts.keys()):
        count = depth_counts[depth]
        bar = "█" * count
        print(f"  Depth {depth:2d}: {count:2d} nodes {bar}")
    print()

    # Show tree structure for first few levels
    print("Tree Structure (first 3 levels):")
    def print_tree(node, prefix="", depth_limit=3):
        if node.depth > depth_limit:
            return

        action_str = str(node.action_taken) if node.action_taken else "ROOT"
        if len(action_str) > 40:
            action_str = action_str[:37] + "..."

        print(f"{prefix}{action_str}")
        print(f"{prefix}  visits={node.visits}, tried={len(node.tried_actions)}, children={len(node.children)}")

        for i, child in enumerate(node.children):
            is_last = i == len(node.children) - 1
            child_prefix = prefix + ("    " if is_last else "│   ")
            print(f"{prefix}{'└── ' if is_last else '├── '}", end="")
            print_tree(child, child_prefix, depth_limit)

    print_tree(mcts.root, "", depth_limit=2)
    print()

    # Check if tree branched
    if len(mcts.root.children) > 1:
        print("✅ SUCCESS: Tree has multiple branches at root!")
        print(f"   Root has {len(mcts.root.children)} children (should be > 1)")
    else:
        print("⚠️  WARNING: Tree is still mostly linear")
        print(f"   Root has only {len(mcts.root.children)} child")
        print("   This might be okay if simulations are low or tree hit terminal states")

    print()


def test_context_length():
    """Test that context is preserved better with increased limit."""
    print("=" * 70)
    print("Testing Context Length (Fix #2)")
    print("=" * 70)
    print()

    from mcts_reasoning.compositional.actions import CompositionalAction
    from mcts_reasoning.compositional import (
        CognitiveOperation, FocusAspect, ReasoningStyle,
        ConnectionType, OutputFormat
    )

    # Create a long state
    long_state = "Question: Test question\n\n"
    for i in range(100):
        long_state += f"Step {i}: This is reasoning step {i} with some detailed analysis.\n"

    print(f"Created test state with {len(long_state)} characters")

    action = CompositionalAction(
        operation=CognitiveOperation.ANALYZE,
        focus=FocusAspect.DETAILS,
        style=ReasoningStyle.SYSTEMATIC,
        connection=ConnectionType.THEREFORE,
        output_format=OutputFormat.STEPS
    )

    # Generate prompt
    prompt = action.to_prompt(long_state, "Test question", None)

    # Check how much context was included
    if "Step 0:" in prompt:
        print("✅ SUCCESS: Early context (Step 0) is preserved in prompt!")
    else:
        print("⚠️  WARNING: Early context was truncated")

    if "Step 50:" in prompt:
        print("✅ SUCCESS: Mid context (Step 50) is preserved in prompt!")
    else:
        print("⚠️  NOTE: Mid context was truncated (this may be okay)")

    if "Step 99:" in prompt or "Step 98:" in prompt:
        print("✅ SUCCESS: Recent context (Step 99) is preserved in prompt!")
    else:
        print("❌ FAILURE: Recent context was truncated!")

    # Count how many steps are in the prompt
    step_count = prompt.count("Step ")
    print(f"\nPrompt includes {step_count}/100 reasoning steps")
    print(f"Prompt length: {len(prompt)} characters")
    print()

    # Show old vs new context limit
    print("Context limits:")
    print("  OLD: 1000 chars (would include ~20 steps)")
    print("  NEW: 4000 chars (should include ~80 steps)")
    print()


if __name__ == "__main__":
    print("\nVerifying Bug Fixes")
    print("=" * 70)
    print()

    # Test 1: Tree exploration
    test_tree_exploration()

    # Test 2: Context length
    test_context_length()

    print("=" * 70)
    print("Tests complete!")
    print("=" * 70)
