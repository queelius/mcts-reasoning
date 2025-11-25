#!/usr/bin/env python3
"""
Test Meta-Reasoning: LLM Suggests Next Action

This tests the meta-reasoning feature where the LLM analyzes the current
reasoning state and suggests which cognitive operation would be most productive next.
"""

from mcts_reasoning.reasoning import ReasoningMCTS
from mcts_reasoning.compositional.providers import MockLLMProvider
from mcts_reasoning.compositional.rag import CompositionalRAGStore


def test_meta_reasoning_mock():
    """Test meta-reasoning with mock LLM."""
    print("=" * 80)
    print("TESTING META-REASONING WITH MOCK LLM")
    print("=" * 80)

    # Create mock LLM with meta-reasoning responses
    llm = MockLLMProvider(responses={
        "evaluate": "0.7",
        "quality": "0.75",
        # Meta-reasoning response
        "OPERATION": """OPERATION: analyze
FOCUS: problem structure
STYLE: systematic
CONFIDENCE: 0.85
REASONING: The problem needs systematic decomposition into prime checking steps."""
    })

    # Create RAG store
    rag_store = CompositionalRAGStore()

    print("\n[Test 1] Meta-reasoning enabled")
    print("─" * 80)

    mcts = (
        ReasoningMCTS()
        .with_llm(llm)
        .with_question("Find all prime numbers less than 20")
        .with_compositional_actions(enabled=True)
        .with_rag_store(rag_store)
        .with_meta_reasoning(enabled=True, bias_strength=3.0)  # Enable meta-reasoning!
        .with_exploration(1.414)
        .with_max_rollout_depth(3)
    )

    print(f"Meta-reasoner enabled: {mcts.meta_reasoner is not None}")
    print(f"Bias strength: {mcts.meta_bias_strength}")
    print()

    # Run search
    print("Running search with meta-reasoning...")
    mcts.search("Let's find prime numbers...", simulations=10)

    # Check meta-reasoner stats
    if mcts.meta_reasoner:
        stats = mcts.meta_reasoner.get_stats()
        print(f"\nMeta-Reasoning Statistics:")
        print(f"  Suggestions made: {stats['suggestion_count']}")
        print(f"  Average confidence: {stats['average_confidence']:.3f}")

        if stats['most_suggested']:
            op, count = stats['most_suggested']
            print(f"  Most suggested operation: {op} ({count} times)")

        if stats.get('operation_distribution'):
            print(f"  Operation distribution: {stats['operation_distribution']}")

        if stats.get('recent_suggestions'):
            print(f"\n  Recent suggestions:")
            for i, sugg in enumerate(stats['recent_suggestions'][:3], 1):
                print(f"    {i}. {sugg['operation']} (confidence={sugg['confidence']:.2f})")
                print(f"       Reasoning: {sugg['reasoning'][:80]}...")

    # Compare to baseline without meta-reasoning
    print("\n[Test 2] Baseline without meta-reasoning")
    print("─" * 80)

    mcts_baseline = (
        ReasoningMCTS()
        .with_llm(llm)
        .with_question("Find all prime numbers less than 20")
        .with_compositional_actions(enabled=True)
        .with_rag_store(rag_store)
        # NO meta-reasoning
        .with_exploration(1.414)
        .with_max_rollout_depth(3)
    )

    print(f"Meta-reasoner enabled: {mcts_baseline.meta_reasoner is not None}")
    print()

    print("Running baseline search...")
    mcts_baseline.search("Let's find prime numbers...", simulations=10)

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    nodes_meta = mcts.get_all_nodes()
    nodes_baseline = mcts_baseline.get_all_nodes()

    best_value_meta = max(n.value/n.visits if n.visits > 0 else 0 for n in nodes_meta)
    best_value_baseline = max(n.value/n.visits if n.visits > 0 else 0 for n in nodes_baseline)

    print(f"\nWith Meta-Reasoning:")
    print(f"  Total nodes: {len(nodes_meta)}")
    print(f"  Best value: {best_value_meta:.3f}")
    print(f"  Meta-reasoning suggestions: {stats['suggestion_count']}")

    print(f"\nBaseline (no meta-reasoning):")
    print(f"  Total nodes: {len(nodes_baseline)}")
    print(f"  Best value: {best_value_baseline:.3f}")

    # Success criteria
    if stats['suggestion_count'] > 0:
        print("\n✅ Meta-reasoning test PASSED - Suggestions being made")
    else:
        print("\n⚠️  Meta-reasoning not making suggestions")

    print("=" * 80)


def test_meta_reasoning_with_real_llm():
    """Test meta-reasoning with real LLM (if available)."""
    import os

    # Check for API keys
    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("\n⚠️  No API keys found - skipping real LLM meta-reasoning test")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test with real LLM")
        return

    print("\n" + "=" * 80)
    print("TESTING META-REASONING WITH REAL LLM")
    print("=" * 80)

    from mcts_reasoning.compositional.providers import get_llm
    from mcts_reasoning.compositional.rag import get_math_compositional_rag

    # Get real LLM
    provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"
    llm = get_llm(provider)
    print(f"✓ Connected to {provider} LLM\n")

    # Use math RAG
    rag_store = get_math_compositional_rag()

    # Test problem: Quadratic equation
    problem = "Solve the equation x^2 - 7x + 12 = 0"

    print(f"Problem: {problem}")
    print("─" * 80)

    mcts = (
        ReasoningMCTS()
        .with_llm(llm)
        .with_question(problem)
        .with_compositional_actions(enabled=True)
        .with_rag_store(rag_store)
        .with_meta_reasoning(enabled=True, temperature=0.2, bias_strength=4.0)
        .with_solution_detection(enabled=True, threshold=0.75)
        .with_exploration(1.414)
        .with_max_rollout_depth(4)
    )

    print("Running search with meta-reasoning...")
    mcts.search(f"Let's solve: {problem}", simulations=20)

    # Results
    nodes = mcts.get_all_nodes()
    best_node = max(nodes, key=lambda n: n.value/n.visits if n.visits > 0 else 0)

    print(f"\nResults:")
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Best value: {best_node.value/best_node.visits:.3f}")

    # Meta-reasoning stats
    if mcts.meta_reasoner:
        stats = mcts.meta_reasoner.get_stats()
        print(f"\nMeta-Reasoning Performance:")
        print(f"  Suggestions made: {stats['suggestion_count']}")
        print(f"  Average confidence: {stats['average_confidence']:.3f}")

        if stats.get('operation_distribution'):
            print(f"  Operation usage:")
            for op, count in sorted(stats['operation_distribution'].items(),
                                   key=lambda x: x[1], reverse=True):
                print(f"    {op}: {count} times")

        print(f"\n  Sample suggestions:")
        for i, sugg in enumerate(stats['recent_suggestions'][:5], 1):
            print(f"    {i}. {sugg['operation']} (conf={sugg['confidence']:.2f})")
            print(f"       → {sugg['reasoning']}")

    # Show solution if found
    from mcts_reasoning.solution_detection import is_finalized_solution
    solutions = [n for n in nodes if is_finalized_solution(n.state)]

    if solutions:
        print(f"\n✓ Found {len(solutions)} finalized solutions")
        best_solution = max(solutions, key=lambda n: n.value/n.visits if n.visits > 0 else 0)
        print(f"\n  Best Solution Preview:")
        state_lines = best_solution.state.split('\n')
        for line in state_lines[:10]:
            print(f"    {line}")

    print("\n" + "=" * 80)


def main():
    """Run all meta-reasoning tests."""
    print("\n" + "=" * 80)
    print("META-REASONING TEST SUITE")
    print("=" * 80)

    # Always run mock test
    test_meta_reasoning_mock()

    # Run real LLM test if available
    test_meta_reasoning_with_real_llm()

    print("\n" + "=" * 80)
    print("ALL META-REASONING TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
