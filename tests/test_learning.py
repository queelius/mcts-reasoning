#!/usr/bin/env python3
"""Test automatic learning from successful paths."""

from mcts_reasoning.reasoning import ReasoningMCTS
from mcts_reasoning.compositional.providers import MockLLMProvider
from mcts_reasoning.compositional.rag import CompositionalRAGStore

def test_learning():
    """Test that system learns from successful reasoning paths."""
    print("Testing Automatic Learning from Successful Paths")
    print("=" * 70)

    # Create mock LLM with good evaluation scores
    llm = MockLLMProvider(responses={
        "evaluate": "0.85",  # High quality scores for learning
        "quality": "0.9"
    })

    # Create RAG store
    rag_store = CompositionalRAGStore()

    print(f"\nInitial RAG store: {len(rag_store)} guidance entries")

    # Create MCTS with learning enabled
    mcts = (
        ReasoningMCTS()
        .with_llm(llm)
        .with_question("Find all prime numbers less than 30")
        .with_compositional_actions(enabled=True)
        .with_rag_store(rag_store)
        .with_learning(enabled=True, auto_learn=True)  # Enable automatic learning
        .with_exploration(1.414)
        .with_max_rollout_depth(3)
    )

    print(f"Learning enabled: {mcts.path_learner is not None}")
    print(f"Auto-learn: {mcts.auto_learn}")
    print()

    # Run first search
    print("Running first search...")
    mcts.search("Let's identify primes...", simulations=20)

    # Check learning stats
    if mcts.path_learner:
        stats = mcts.path_learner.get_stats()
        print(f"\nLearning Statistics (after first search):")
        print(f"  Patterns learned: {stats['patterns_learned']}")
        print(f"  Learning iterations: {stats['learning_count']}")

        if stats['recent_patterns']:
            print(f"\n  Recent learned patterns:")
            for pattern in stats['recent_patterns'][:3]:
                print(f"    - Problem: {pattern['problem'][:60]}...")
                print(f"      Value: {pattern['value']:.3f}")
                print(f"      Is solution: {pattern['is_solution']}")
                print(f"      Operations: {pattern['operations']}")
                print()

    # Check RAG store growth
    print(f"\nRAG store after learning: {len(rag_store)} guidance entries")

    # Run second search on similar problem
    print("\nRunning second search on similar problem...")
    mcts2 = (
        ReasoningMCTS()
        .with_llm(llm)
        .with_question("What are the prime numbers between 20 and 40?")
        .with_compositional_actions(enabled=True)
        .with_rag_store(rag_store)  # Use learned RAG store
        .with_learning(enabled=True, auto_learn=True)
        .with_exploration(1.414)
        .with_max_rollout_depth(3)
    )

    mcts2.search("Let's find primes...", simulations=20)

    # Check if learning accumulated
    if mcts2.path_learner:
        stats2 = mcts2.path_learner.get_stats()
        print(f"\nLearning Statistics (after second search):")
        print(f"  Total patterns: {stats2['patterns_learned']}")
        print(f"  Learning iterations: {stats2['learning_count']}")

    print(f"\nFinal RAG store: {len(rag_store)} guidance entries")

    # Show learned guidance
    if rag_store.guidance:
        print(f"\nLearned Guidance Patterns:")
        for i, guidance in enumerate(rag_store.guidance[:3], 1):
            print(f"\n  Pattern {i}:")
            print(f"    Problem pattern: {guidance.problem_pattern}")
            print(f"    Keywords: {guidance.problem_keywords}")
            print(f"    Success rate: {guidance.success_rate:.3f}")
            if guidance.recommended_operations:
                ops = [op.value for op in guidance.recommended_operations]
                print(f"    Recommended operations: {ops}")
            if guidance.weights and 'cognitive_op' in guidance.weights:
                print(f"    Operation weights: {dict(list(guidance.weights['cognitive_op'].items())[:3])}")

    # Verify learning occurred
    print("\n" + "=" * 70)
    if stats['learning_count'] > 0:
        print("✅ Learning system working!")
        print(f"   System learned {stats['learning_count']} patterns from search")
        print(f"   RAG store now has {len(rag_store)} guidance entries")
    else:
        print("⚠️  No learning occurred")
        print("   This might be expected if no paths had sufficient value")
    print("=" * 70)

if __name__ == "__main__":
    test_learning()
