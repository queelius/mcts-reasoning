#!/usr/bin/env python3
"""Test automatic solution detection and finalization."""

from mcts_reasoning.reasoning import ReasoningMCTS
from mcts_reasoning.compositional.providers import MockLLMProvider
from mcts_reasoning.solution_detection import is_finalized_solution

def test_solution_finalization():
    """Test that solutions are automatically detected and finalized."""
    print("Testing Automatic Solution Detection and Finalization")
    print("=" * 70)

    # Create mock LLM with solution-like responses
    solution_response = """Let me solve this step by step.

First, I'll identify the prime numbers less than 20:
2, 3, 5, 7, 11, 13, 17, 19

Therefore, the prime numbers less than 20 are: 2, 3, 5, 7, 11, 13, 17, 19.

Final Answer: There are 8 prime numbers less than 20."""

    llm = MockLLMProvider(responses={
        "judge": "VERDICT: SOLUTION\nCONFIDENCE: 0.9\nREASONING: Complete answer provided\nREFINEMENT_NEEDED: NO",
        "finalize": """## Final Answer

The prime numbers less than 20 are: **2, 3, 5, 7, 11, 13, 17, 19**

There are a total of **8 prime numbers** less than 20.

## Key Reasoning

- Prime numbers are divisible only by 1 and themselves
- Checked all numbers from 2 to 19
- Found 8 numbers meeting the criteria""",
        "": solution_response  # Default for other prompts
    })

    # Create MCTS with solution detection enabled
    mcts = (
        ReasoningMCTS()
        .with_llm(llm)
        .with_question("What are the prime numbers less than 20?")
        .with_compositional_actions(enabled=True)
        .with_solution_detection(enabled=True, threshold=0.7)
        .with_exploration(1.414)
        .with_max_rollout_depth(3)
    )

    print("\nConfiguration:")
    print(f"  Solution detection: {'Enabled' if mcts.solution_detector else 'Disabled'}")
    print(f"  Auto-finalize: {mcts.auto_finalize_solutions}")
    print(f"  Detection threshold: {mcts.solution_detector.threshold if mcts.solution_detector else 'N/A'}")
    print()

    # Run search
    print("Running search (solution detection should trigger)...")
    initial_state = "Let's find all prime numbers less than 20."
    mcts.search(initial_state, simulations=15)

    # Analyze results
    nodes = mcts.get_all_nodes()
    print(f"\nTree Statistics:")
    print(f"  Total nodes: {len(nodes)}")

    # Find finalized solutions
    finalized_nodes = []
    for i, node in enumerate(nodes):
        if is_finalized_solution(node.state):
            finalized_nodes.append((i, node))

    print(f"  Finalized solution nodes: {len(finalized_nodes)}")

    # Show finalized solutions
    if finalized_nodes:
        print(f"\nFinalized Solutions:")
        for idx, node in finalized_nodes:
            print(f"\n  Node {idx}:")
            print(f"    Depth: {node.depth}")
            print(f"    Visits: {node.visits}")
            print(f"    Value: {node.value:.3f}")
            print(f"    Is terminal: {node.is_leaf}")
            print(f"    State preview:")
            state_lines = node.state.split('\n')[:10]  # First 10 lines
            for line in state_lines:
                print(f"      {line}")
            if len(node.state.split('\n')) > 10:
                print(f"      ... ({len(node.state.split('\n')) - 10} more lines)")

    # Check solution detector stats
    if mcts.solution_detector:
        print(f"\nSolution Detector Statistics:")
        print(f"  Judgment count: {mcts.solution_detector._judgment_count}")

    # Check solution finalizer stats
    if mcts.solution_finalizer:
        stats = mcts.solution_finalizer.get_stats()
        print(f"\nSolution Finalizer Statistics:")
        print(f"  Finalization count: {stats['finalization_count']}")

    # Verify finalized nodes are terminal
    print(f"\nVerifying Terminal Status:")
    all_terminal = True
    for idx, node in finalized_nodes:
        is_terminal = mcts._is_terminal_state(node.state)
        status = "✓" if is_terminal else "✗"
        print(f"  Node {idx}: {status} {'Terminal' if is_terminal else 'Not terminal (ERROR!)'}")
        if not is_terminal:
            all_terminal = False

    # Final summary
    print("\n" + "=" * 70)
    if finalized_nodes:
        print("✅ Solution finalization worked!")
        print(f"   {len(finalized_nodes)} solution(s) detected and finalized")
        if all_terminal:
            print("   All finalized nodes are properly marked as terminal")
        else:
            print("   ⚠️  Some finalized nodes not marked terminal (check _is_terminal_state)")
    else:
        print("⚠️  No solutions were finalized")
        print("   This might mean:")
        print("   - LLM didn't produce solution-like responses")
        print("   - Solution detector threshold too high")
        print("   - Not enough simulations to reach solution")
    print("=" * 70)

if __name__ == "__main__":
    test_solution_finalization()
