#!/usr/bin/env python3
"""
Demonstration of compositional prompting features with MCTS.

Shows how to use the advanced compositional action space with
cognitive operations, focus aspects, reasoning styles, etc.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts_reasoning import (
    ReasoningMCTS,
    MockLLMProvider,
    get_llm,
    CognitiveOperation,
    FocusAspect,
    ReasoningStyle,
    ComposingPrompt
)


def demo_composing_prompt():
    """Demonstrate ComposingPrompt fluid API."""
    print("=" * 60)
    print("ComposingPrompt Demonstration")
    print("=" * 60)

    # Example 1: Build a structured reasoning prompt
    prompt = (
        ComposingPrompt()
        .cognitive_op(CognitiveOperation.DECOMPOSE)
        .focus(FocusAspect.STRUCTURE)
        .style(ReasoningStyle.SYSTEMATIC)
        .problem_context("Find all prime numbers less than 20")
        .build()
    )

    print("\nExample 1: Decompose with systematic style")
    print("-" * 40)
    print(prompt)

    # Example 2: Analyze with critical thinking
    prompt2 = (
        ComposingPrompt()
        .cognitive_op(CognitiveOperation.ANALYZE)
        .focus(FocusAspect.ASSUMPTIONS)
        .style(ReasoningStyle.CRITICAL)
        .problem_context("Is this algorithm correct for all inputs?")
        .build()
    )

    print("\n\nExample 2: Critical analysis")
    print("-" * 40)
    print(prompt2)

    # Example 3: Get action vector
    prompt3 = ComposingPrompt().cognitive_op(CognitiveOperation.VERIFY).focus(FocusAspect.CORRECTNESS)
    vector = prompt3.get_action_vector()

    print("\n\nExample 3: Action vector")
    print("-" * 40)
    print(f"Vector: {vector}")


def demo_mcts_with_compositional():
    """Demonstrate MCTS with compositional actions."""
    print("\n\n" + "=" * 60)
    print("MCTS with Compositional Actions")
    print("=" * 60)

    # Setup LLM
    llm = MockLLMProvider({
        "decompose": "Let me break this down: we need to find primes < 20",
        "analyze": "Primes are numbers divisible only by 1 and themselves",
        "generate": "Testing: 2, 3, 5, 7, 11, 13, 17, 19",
        "verify": "All numbers checked. These are all the primes.",
        "synthesize": "Final answer: {2, 3, 5, 7, 11, 13, 17, 19}",
        "terminal": "YES",
        "quality": "0.90"
    })

    # Create MCTS with compositional actions
    question = "Find all prime numbers less than 20"

    print(f"\nQuestion: {question}")
    print("Enabling compositional action space...")

    mcts = (
        ReasoningMCTS()
        .with_llm(llm)
        .with_question(question)
        .with_exploration(1.414)
        .with_compositional_actions(enabled=True)  # Enable compositional!
        .with_max_rollout_depth(4)
    )

    print("\nRunning MCTS search with compositional actions...")

    # Run search
    initial_state = "Let's find all prime numbers less than 20."
    mcts.search(initial_state, simulations=30)

    # Display results
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)

    stats = mcts.stats
    print(f"\nTree statistics:")
    print(f"  Total nodes explored: {stats['total_nodes']}")
    print(f"  Maximum depth: {stats['max_depth']}")
    print(f"  Best value: {stats['best_value']:.3f}")

    print(f"\nBest solution:")
    print("-" * 40)
    solution = mcts.solution
    print(solution[-500:] if len(solution) > 500 else solution)

    # Sample diverse solutions
    print("\n" + "=" * 40)
    print("SAMPLING DIVERSE SOLUTIONS")
    print("=" * 40)

    print("\nSampling 3 diverse reasoning paths...")
    paths = mcts.sample(n=3, strategy="diverse", temperature=1.5)

    for i, path in enumerate(paths, 1):
        print(f"\nPath {i} (length={path.length}, value={path.total_value:.2f}):")
        print(f"  Final state: {path.final_state[-200:]}")

    # Check consistency
    print("\n" + "=" * 40)
    print("CONSISTENCY CHECK")
    print("=" * 40)

    print("\nChecking consistency across 10 samples...")
    result = mcts.check_consistency(n_samples=10, temperature=1.0)

    print(f"\nMost consistent solution:")
    print(f"  Confidence: {result['confidence']:.1%}")
    print(f"  Support: {result['support']}/{result['total_samples']} samples")
    print(f"  Number of clusters: {len(result['clusters'])}")
    print(f"\nSolution: {result['solution'][-300:]}")


def demo_weighted_sampling():
    """Demonstrate weighted action sampling."""
    print("\n\n" + "=" * 60)
    print("Weighted Action Sampling")
    print("=" * 60)

    # Define weights to bias toward certain operations
    weights = {
        'cognitive_op': {
            CognitiveOperation.DECOMPOSE: 3.0,  # Strongly prefer decomposition
            CognitiveOperation.VERIFY: 2.0,     # Prefer verification
            CognitiveOperation.ANALYZE: 1.0,    # Normal preference
        },
        'focus': {
            FocusAspect.STRUCTURE: 2.0,
            FocusAspect.CORRECTNESS: 2.0,
        },
        'style': {
            ReasoningStyle.SYSTEMATIC: 3.0,     # Strongly prefer systematic
            ReasoningStyle.FORMAL: 1.5,
        }
    }

    print("\nWeighted sampling (biased toward decompose + systematic):")
    print("-" * 40)

    # Sample 5 actions with weights
    for i in range(5):
        prompt = ComposingPrompt.sample_action(weights)
        vector = prompt.get_action_vector()
        print(f"\nSample {i+1}: ω={vector['omega']}, φ={vector['phi']}, σ={vector['sigma']}")


def main():
    """Run all demonstrations."""
    demo_composing_prompt()
    demo_mcts_with_compositional()
    demo_weighted_sampling()

    print("\n\n" + "=" * 60)
    print("✅ Compositional prompting demo completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
