#!/usr/bin/env python3
"""
Test Reflection and Critique: Self-Improvement Through Self-Evaluation

This tests the reflection feature where the LLM critiques its own reasoning
and suggests improvements, creating a feedback loop for quality.
"""

from mcts_reasoning.reasoning import ReasoningMCTS
from mcts_reasoning.compositional.providers import MockLLMProvider
from mcts_reasoning.reflection import ReflectionCritic, ReflectiveRefinementLoop


def test_reflection_mock():
    """Test reflection with mock LLM."""
    print("=" * 80)
    print("TESTING REFLECTION WITH MOCK LLM")
    print("=" * 80)

    # Create mock LLM with reflection responses
    llm = MockLLMProvider(responses={
        "evaluate": "0.65",
        "quality": "0.7",
        # Critique response
        "QUALITY_SCORE": """QUALITY_SCORE: 0.55
STRENGTHS:
- Systematic approach
- Clear reasoning structure
WEAKNESSES:
- Missing verification step
- Could be more efficient
SUGGESTIONS:
- Add verification of the result
- Consider using a more direct method
NEEDS_REFINEMENT: YES
REASONING: The reasoning is sound but incomplete and could be more efficient.""",
        # Refinement response
        "REFINED": "Improved reasoning with verification step added and more efficient approach."
    })

    print("\n[Test 1] Basic Reflection Critic")
    print("─" * 80)

    critic = ReflectionCritic(llm, temperature=0.3)

    test_reasoning = """
    Let's solve this step by step:
    1. First, identify what the problem is asking
    2. Break it down into smaller parts
    3. Solve each part
    """

    critique = critic.critique(test_reasoning, "Solve x^2 - 5x + 6 = 0")

    print(f"Critique Results:")
    print(f"  Quality score: {critique.quality_score:.2f}")
    print(f"  Strengths: {len(critique.strengths)}")
    for s in critique.strengths:
        print(f"    - {s}")
    print(f"  Weaknesses: {len(critique.weaknesses)}")
    for w in critique.weaknesses:
        print(f"    - {w}")
    print(f"  Suggestions: {len(critique.suggestions)}")
    for s in critique.suggestions:
        print(f"    - {s}")
    print(f"  Needs refinement: {critique.needs_refinement}")

    # Test refinement
    if critique.needs_refinement:
        print(f"\n  Refining reasoning...")
        refined = critic.refine(test_reasoning, critique, "Solve x^2 - 5x + 6 = 0")
        print(f"  Refined reasoning created: {len(refined)} characters")
        print(f"  Preview: {refined[:100]}...")

    print("\n[Test 2] Reflective Refinement Loop")
    print("─" * 80)

    loop = ReflectiveRefinementLoop(llm, max_iterations=3, quality_threshold=0.8)

    initial_reasoning = "Let's solve the equation by factoring..."

    final_reasoning, critiques = loop.refine_iteratively(
        initial_reasoning,
        "Solve x^2 - 5x + 6 = 0"
    )

    print(f"Refinement Loop Results:")
    print(f"  Iterations: {len(critiques)}")
    print(f"  Final quality: {critiques[-1].quality_score:.2f}")
    print(f"  Quality progression: {[c.quality_score for c in critiques]}")

    print("\n[Test 3] Integration with MCTS")
    print("─" * 80)

    mcts = (
        ReasoningMCTS()
        .with_llm(llm)
        .with_question("Find all prime numbers less than 20")
        .with_compositional_actions(enabled=True)
        .with_reflection(enabled=True, quality_threshold=0.6)  # Enable reflection!
        .with_exploration(1.414)
        .with_max_rollout_depth(3)
    )

    print(f"Reflection enabled: {mcts.reflection_critic is not None}")
    print(f"Quality threshold: {mcts.reflection_threshold}")
    print()

    # Run search
    print("Running search with reflection...")
    mcts.search("Let's find prime numbers...", simulations=10)

    # Check reflection stats
    if mcts.reflection_critic:
        stats = mcts.reflection_critic.get_stats()
        print(f"\nReflection Statistics:")
        print(f"  Critiques made: {stats['critique_count']}")
        print(f"  Average quality: {stats['average_quality']:.3f}")
        print(f"  Refinement rate: {stats['refinement_rate']:.1%}")

        if stats.get('recent_critiques'):
            print(f"\n  Recent critiques (last 3):")
            for i, c in enumerate(stats['recent_critiques'][:3], 1):
                print(f"    {i}. Quality: {c['quality_score']:.2f}, "
                      f"Refined: {c['needs_refinement']}, "
                      f"Weaknesses: {c['weaknesses_count']}")

    # Compare to baseline
    print("\n[Test 4] Baseline without reflection")
    print("─" * 80)

    mcts_baseline = (
        ReasoningMCTS()
        .with_llm(llm)
        .with_question("Find all prime numbers less than 20")
        .with_compositional_actions(enabled=True)
        # NO reflection
        .with_exploration(1.414)
        .with_max_rollout_depth(3)
    )

    print(f"Reflection enabled: {mcts_baseline.reflection_critic is not None}")
    print()

    print("Running baseline search...")
    mcts_baseline.search("Let's find prime numbers...", simulations=10)

    # Compare
    print("\n" + "=" * 80)
    print("COMPARISON")
    print("=" * 80)

    nodes_reflected = mcts.get_all_nodes()
    nodes_baseline = mcts_baseline.get_all_nodes()

    best_value_reflected = max(n.value/n.visits if n.visits > 0 else 0 for n in nodes_reflected)
    best_value_baseline = max(n.value/n.visits if n.visits > 0 else 0 for n in nodes_baseline)

    print(f"\nWith Reflection:")
    print(f"  Total nodes: {len(nodes_reflected)}")
    print(f"  Best value: {best_value_reflected:.3f}")
    print(f"  Critiques: {stats['critique_count']}")
    print(f"  Refinements: {int(stats['refinement_rate'] * stats['critique_count'])}")

    print(f"\nBaseline (no reflection):")
    print(f"  Total nodes: {len(nodes_baseline)}")
    print(f"  Best value: {best_value_baseline:.3f}")

    # Success criteria
    if stats['critique_count'] > 0:
        print("\n✅ Reflection test PASSED - Critiques being made")
    else:
        print("\n⚠️  Reflection not making critiques")

    print("=" * 80)


def test_reflection_with_real_llm():
    """Test reflection with real LLM (if available)."""
    import os

    if not (os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY")):
        print("\n⚠️  No API keys found - skipping real LLM reflection test")
        print("   Set OPENAI_API_KEY or ANTHROPIC_API_KEY to test with real LLM")
        return

    print("\n" + "=" * 80)
    print("TESTING REFLECTION WITH REAL LLM")
    print("=" * 80)

    from mcts_reasoning.compositional.providers import get_llm
    from mcts_reasoning.compositional.rag import get_math_compositional_rag

    # Get real LLM
    provider = "openai" if os.getenv("OPENAI_API_KEY") else "anthropic"
    llm = get_llm(provider)
    print(f"✓ Connected to {provider} LLM\n")

    # Test problem
    problem = "Find the sum of all even numbers from 1 to 20"

    print(f"Problem: {problem}")
    print("─" * 80)

    # Test reflection loop standalone
    print("\n[Standalone Reflective Refinement Loop]")
    loop = ReflectiveRefinementLoop(llm, max_iterations=2, quality_threshold=0.85)

    initial_reasoning = """
    Let's solve this problem:
    The even numbers from 1 to 20 are: 2, 4, 6, 8, 10, 12, 14, 16, 18, 20.
    Sum = 2 + 4 + 6 + 8 + 10 + 12 + 14 + 16 + 18 + 20
    """

    final, critiques = loop.refine_iteratively(initial_reasoning, problem)

    print(f"\nRefinement Results:")
    print(f"  Iterations: {len(critiques)}")
    print(f"  Quality progression: {[f'{c.quality_score:.2f}' for c in critiques]}")

    for i, c in enumerate(critiques, 1):
        print(f"\n  Iteration {i}:")
        print(f"    Quality: {c.quality_score:.2f}")
        print(f"    Weaknesses: {len(c.weaknesses)}")
        for w in c.weaknesses[:2]:
            print(f"      - {w}")
        print(f"    Suggestions: {len(c.suggestions)}")
        for s in c.suggestions[:2]:
            print(f"      - {s}")

    # Test integrated with MCTS
    print("\n[Integrated with MCTS]")

    rag_store = get_math_compositional_rag()

    mcts = (
        ReasoningMCTS()
        .with_llm(llm)
        .with_question(problem)
        .with_compositional_actions(enabled=True)
        .with_rag_store(rag_store)
        .with_reflection(enabled=True, temperature=0.3, quality_threshold=0.7)
        .with_solution_detection(enabled=True)
        .with_exploration(1.414)
        .with_max_rollout_depth(3)
    )

    print("Running MCTS with reflection...")
    mcts.search(f"Let's solve: {problem}", simulations=15)

    # Results
    nodes = mcts.get_all_nodes()
    best_node = max(nodes, key=lambda n: n.value/n.visits if n.visits > 0 else 0)

    print(f"\nMCTS Results:")
    print(f"  Total nodes: {len(nodes)}")
    print(f"  Best value: {best_node.value/best_node.visits:.3f}")

    # Reflection stats
    if mcts.reflection_critic:
        stats = mcts.reflection_critic.get_stats()
        print(f"\nReflection Performance:")
        print(f"  Critiques made: {stats['critique_count']}")
        print(f"  Average quality: {stats['average_quality']:.3f}")
        print(f"  Refinement rate: {stats['refinement_rate']:.1%}")

        print(f"\n  Sample critiques:")
        for i, c in enumerate(stats['recent_critiques'][:3], 1):
            print(f"    {i}. Quality: {c['quality_score']:.2f}, "
                  f"Needs refinement: {c['needs_refinement']}")

    print("\n" + "=" * 80)


def main():
    """Run all reflection tests."""
    print("\n" + "=" * 80)
    print("REFLECTION TEST SUITE")
    print("=" * 80)

    # Always run mock test
    test_reflection_mock()

    # Run real LLM test if available
    test_reflection_with_real_llm()

    print("\n" + "=" * 80)
    print("ALL REFLECTION TESTS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
