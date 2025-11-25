#!/usr/bin/env python3
"""
Test MCTS reasoning system on real math and logic problems.

This validates:
- Solution detection and finalization
- Learning from successful paths
- RAG-guided action selection
- Context management
- All features working together
"""

from mcts_reasoning.reasoning import ReasoningMCTS
from mcts_reasoning.compositional.rag import CompositionalRAGStore, get_math_compositional_rag
from mcts_reasoning.compositional.providers import MockLLMProvider, get_llm
from mcts_reasoning.solution_detection import is_finalized_solution
import os


# Test problems organized by category
MATH_PROBLEMS = {
    "arithmetic": [
        "What is 15% of 240?",
        "If a store offers a 30% discount on a $150 item, what is the final price?",
    ],

    "algebra": [
        "Solve for x: 2x + 5 = 17",
        "Solve the equation x^2 - 5x + 6 = 0",
        "If y = 3x + 2 and x = 4, what is y?",
    ],

    "number_theory": [
        "Find all prime numbers less than 30",
        "What are the factors of 48?",
        "Is 97 a prime number?",
    ],

    "logic": [
        "If all cats are mammals and all mammals are animals, are all cats animals?",
        "In a group of 100 people, 60 like coffee, 50 like tea, and 30 like both. How many like neither?",
    ],

    "word_problems": [
        "A train travels 120 miles in 2 hours. What is its average speed?",
        "If 3 apples cost $2.40, how much do 5 apples cost?",
    ]
}


def test_with_mock_llm():
    """Test with mock LLM for fast validation."""
    print("=" * 80)
    print("TESTING WITH MOCK LLM (Fast Validation)")
    print("=" * 80)

    # Create mock LLM with realistic responses
    llm = MockLLMProvider(responses={
        "evaluate": "0.75",
        "quality": "0.8",
        "judge": "VERDICT: SOLUTION\nCONFIDENCE: 0.85\nREASONING: Complete answer with correct calculation\nREFINEMENT_NEEDED: NO",
        "finalize": """## Final Answer
The answer is 36.

## Key Reasoning
- Applied basic arithmetic operation
- Verified calculation accuracy
- Presented clear result"""
    })

    # Create RAG store (starts empty, will learn)
    rag_store = CompositionalRAGStore()

    # Test one problem from each category
    test_problems = [
        ("arithmetic", "What is 15% of 240?"),
        ("algebra", "Solve for x: 2x + 5 = 17"),
        ("number_theory", "Find all prime numbers less than 30"),
    ]

    total_solutions = 0
    total_learning_events = 0

    for category, problem in test_problems:
        print(f"\n{'─' * 80}")
        print(f"Problem [{category}]: {problem}")
        print('─' * 80)

        mcts = (
            ReasoningMCTS()
            .with_llm(llm)
            .with_question(problem)
            .with_compositional_actions(enabled=True)
            .with_rag_store(rag_store)
            .with_learning(enabled=True, auto_learn=True)
            .with_solution_detection(enabled=True, threshold=0.7)
            .with_exploration(1.414)
            .with_max_rollout_depth(3)
        )

        # Run search
        mcts.search(f"Let's solve: {problem}", simulations=15)

        # Check results
        nodes = mcts.get_all_nodes()
        solutions = [n for n in nodes if is_finalized_solution(n.state)]

        print(f"\nResults:")
        print(f"  Total nodes: {len(nodes)}")
        print(f"  Solutions found: {len(solutions)}")
        print(f"  Best value: {max(n.value/n.visits if n.visits > 0 else 0 for n in nodes):.3f}")

        if mcts.path_learner:
            stats = mcts.path_learner.get_stats()
            print(f"  Patterns learned: {stats['learning_count']}")
            total_learning_events += stats['learning_count']

        total_solutions += len(solutions)

    print(f"\n{'=' * 80}")
    print(f"MOCK LLM TEST SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total solutions finalized: {total_solutions}")
    print(f"Total learning events: {total_learning_events}")
    print(f"RAG store size: {len(rag_store)} guidance entries")

    if total_solutions > 0 and total_learning_events > 0:
        print("\n✅ Mock LLM test PASSED - System functioning correctly")
    else:
        print("\n⚠️  Mock LLM test showed limited activity")

    # Assert that the system produced some activity (solutions or learning)
    assert total_solutions > 0 or total_learning_events > 0, \
        "Mock LLM test should produce at least some solutions or learning events"

    # Optionally verify RAG store was created
    assert isinstance(rag_store, CompositionalRAGStore), \
        "RAG store should be a CompositionalRAGStore instance"


def test_with_real_llm(provider: str = None):
    """Test with real LLM (requires API key)."""

    # Detect provider
    if provider is None:
        if os.getenv("OPENAI_API_KEY"):
            provider = "openai"
        elif os.getenv("ANTHROPIC_API_KEY"):
            provider = "anthropic"
        else:
            print("\n⚠️  No API keys found - skipping real LLM test")
            print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test with real LLM")
            return

    print(f"\n{'=' * 80}")
    print(f"TESTING WITH REAL LLM ({provider.upper()})")
    print(f"{'=' * 80}")

    try:
        # Get real LLM
        llm = get_llm(provider)
        print(f"✓ Connected to {provider} LLM")
    except Exception as e:
        print(f"✗ Failed to connect to {provider}: {e}")
        return

    # Start with math RAG patterns
    rag_store = get_math_compositional_rag()
    print(f"✓ Loaded math RAG store ({len(rag_store)} initial patterns)")

    # Test challenging problems
    test_problems = [
        ("algebra", "Solve the quadratic equation x^2 - 5x + 6 = 0"),
        ("logic", "If all cats are mammals and all mammals are animals, are all cats animals?"),
        ("number_theory", "Find all prime numbers between 20 and 40"),
    ]

    results = []

    for category, problem in test_problems:
        print(f"\n{'─' * 80}")
        print(f"Problem [{category}]: {problem}")
        print('─' * 80)

        mcts = (
            ReasoningMCTS()
            .with_llm(llm)
            .with_question(problem)
            .with_compositional_actions(enabled=True)
            .with_rag_store(rag_store)
            .with_learning(enabled=True, auto_learn=True)
            .with_solution_detection(enabled=True, threshold=0.75)
            .with_context_config(auto_configure=True)
            .with_exploration(1.414)
            .with_max_rollout_depth(4)
        )

        # Run search
        print("Running MCTS search...")
        mcts.search(f"Let's solve: {problem}", simulations=30)

        # Analyze results
        nodes = mcts.get_all_nodes()
        solutions = [n for n in nodes if is_finalized_solution(n.state)]
        best_node = max(nodes, key=lambda n: n.value/n.visits if n.visits > 0 else 0)

        result = {
            'category': category,
            'problem': problem,
            'total_nodes': len(nodes),
            'solutions_found': len(solutions),
            'best_value': best_node.value / best_node.visits if best_node.visits > 0 else 0,
            'max_depth': max(n.depth for n in nodes),
        }

        if mcts.path_learner:
            stats = mcts.path_learner.get_stats()
            result['patterns_learned'] = stats['learning_count']

        if mcts.context_manager:
            ctx_stats = mcts.context_manager.get_stats()
            result['summarizations'] = ctx_stats['summarization_count']

        if mcts.solution_detector:
            result['judgments'] = mcts.solution_detector._judgment_count

        results.append(result)

        # Print results
        print(f"\nResults:")
        print(f"  Total nodes: {result['total_nodes']}")
        print(f"  Max depth: {result['max_depth']}")
        print(f"  Solutions found: {result['solutions_found']}")
        print(f"  Best value: {result['best_value']:.3f}")

        if 'patterns_learned' in result:
            print(f"  Patterns learned: {result['patterns_learned']}")
        if 'summarizations' in result:
            print(f"  Context summarizations: {result['summarizations']}")
        if 'judgments' in result:
            print(f"  Solution judgments: {result['judgments']}")

        # Show best solution if found
        if solutions:
            best_solution = max(solutions, key=lambda n: n.value/n.visits if n.visits > 0 else 0)
            print(f"\n  Best Solution Preview:")
            # Extract final answer section
            state = best_solution.state
            if "## Final Answer" in state:
                answer_part = state.split("## Final Answer")[1].split("\n\n")[0]
                preview = answer_part[:200]
                print(f"    {preview}...")

    # Summary
    print(f"\n{'=' * 80}")
    print(f"REAL LLM TEST SUMMARY")
    print(f"{'=' * 80}")

    total_solutions = sum(r['solutions_found'] for r in results)
    total_patterns = sum(r.get('patterns_learned', 0) for r in results)
    avg_value = sum(r['best_value'] for r in results) / len(results)

    print(f"\nProblems tested: {len(results)}")
    print(f"Total solutions: {total_solutions}")
    print(f"Total patterns learned: {total_patterns}")
    print(f"Average best value: {avg_value:.3f}")
    print(f"Final RAG store size: {len(rag_store)} patterns")

    # Per-category breakdown
    print(f"\nPer-Category Results:")
    for r in results:
        print(f"  {r['category']:15s} - Solutions: {r['solutions_found']:2d}, "
              f"Best value: {r['best_value']:.3f}, "
              f"Learned: {r.get('patterns_learned', 0):2d}")

    # Success criteria
    success = (
        total_solutions >= len(results) * 0.5 and  # At least 50% of problems got solutions
        avg_value >= 0.6  # Average quality is reasonable
    )

    if success:
        print(f"\n✅ Real LLM test PASSED - System performing well")
    else:
        print(f"\n⚠️  Real LLM test showed mixed results")
        if total_solutions < len(results) * 0.5:
            print(f"   - Low solution rate: {total_solutions}/{len(results)}")
        if avg_value < 0.6:
            print(f"   - Low average quality: {avg_value:.3f}")


def main():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("MCTS REASONING - REAL PROBLEM TEST SUITE")
    print("=" * 80)

    # Always run mock test (no longer returns value)
    print("\n[1/2] Running mock LLM tests...")
    test_with_mock_llm()

    # Run real LLM test if available
    print("\n[2/2] Running real LLM tests...")
    test_with_real_llm()

    print("\n" + "=" * 80)
    print("ALL TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
