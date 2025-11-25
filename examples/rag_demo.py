#!/usr/bin/env python3
"""
Demonstration of RAG (Retrieval-Augmented Generation) features.

Shows how to use:
1. Example-based few-shot learning
2. Compositional RAG (type a): Problem → Compositional dimensions
3. Solution RAG (type b): Problem → Full solution examples
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts_reasoning import ReasoningMCTS, MockLLMProvider, ComposingPrompt
from mcts_reasoning.compositional.examples import (
    Example, ExampleSet, get_math_examples
)
from mcts_reasoning.compositional.rag import (
    CompositionalRAGStore,
    SolutionRAGStore,
    get_math_compositional_rag,
    get_coding_compositional_rag
)
from mcts_reasoning.compositional import (
    CognitiveOperation, FocusAspect, ReasoningStyle
)


def demo_examples():
    """Demonstrate Example and ExampleSet functionality."""
    print("=" * 70)
    print("EXAMPLE-BASED FEW-SHOT LEARNING")
    print("=" * 70)

    # Create an example set
    examples = ExampleSet()

    # Add examples
    examples.add_from_dict(
        problem="What is 7 × 8?",
        reasoning_steps=[
            "7 × 8 = 7 × (4 + 4)",
            "= 7 × 4 + 7 × 4",
            "= 28 + 28",
            "= 56"
        ],
        solution="56",
        domain="arithmetic"
    )

    examples.add_from_dict(
        problem="What is 12 × 15?",
        reasoning_steps=[
            "12 × 15 = 12 × (10 + 5)",
            "= 12 × 10 + 12 × 5",
            "= 120 + 60",
            "= 180"
        ],
        solution="180",
        domain="arithmetic"
    )

    print(f"\nCreated example set with {len(examples)} examples")

    # Retrieve similar examples
    query = "What is 9 × 11?"
    print(f"\nQuery: {query}")
    print("\nRetrieving similar examples:")

    similar = examples.retrieve_similar(query, k=2, method='keyword')
    for i, ex in enumerate(similar, 1):
        print(f"\n--- Example {i} ---")
        print(ex.to_prompt_string())

    # Generate few-shot prompt
    print("\n" + "=" * 50)
    print("FEW-SHOT PROMPT")
    print("=" * 50)
    prompt = examples.to_few_shot_prompt(n_examples=2, query=query)
    print(prompt)


def demo_compositional_rag():
    """Demonstrate Compositional RAG (type a)."""
    print("\n\n" + "=" * 70)
    print("COMPOSITIONAL RAG (Type A)")
    print("Problem → Compositional Dimensions")
    print("=" * 70)

    # Get predefined math RAG store
    rag_store = get_math_compositional_rag()
    print(f"\nLoaded RAG store with {len(rag_store)} guidance entries")

    # Test with different problems
    problems = [
        "Solve x² + 3x + 2 = 0",
        "Find all prime numbers between 20 and 30",
        "Calculate 145 × 37"
    ]

    for problem in problems:
        print(f"\n{'='*50}")
        print(f"Problem: {problem}")
        print("-" * 50)

        # Retrieve guidance
        guidance = rag_store.retrieve(problem, k=1)

        if guidance:
            g = guidance[0]
            print(f"Domain: {g.domain}")
            print(f"Recommended operations: {[op.value for op in (g.recommended_operations or [])]}")
            print(f"Recommended focuses: {[f.value for f in (g.recommended_focuses or [])]}")
            print(f"Recommended styles: {[s.value for s in (g.recommended_styles or [])]}")

            # Get weights
            weights = rag_store.get_recommended_weights(problem)
            print(f"\nGenerated weights: {weights}")
        else:
            print("No matching guidance found")


def demo_solution_rag():
    """Demonstrate Solution RAG (type b)."""
    print("\n\n" + "=" * 70)
    print("SOLUTION RAG (Type B)")
    print("Problem → Complete Solution Examples")
    print("=" * 70)

    # Create solution RAG store
    rag_store = SolutionRAGStore()

    # Add some examples
    rag_store.add(
        problem="Solve the quadratic equation x² - 5x + 6 = 0",
        reasoning_steps=[
            "Factor the quadratic: (x - 2)(x - 3) = 0",
            "Set each factor to zero: x - 2 = 0 or x - 3 = 0",
            "Solve for x: x = 2 or x = 3"
        ],
        solution="x = 2 or x = 3",
        domain="algebra"
    )

    rag_store.add(
        problem="Solve x² + 4x + 4 = 0",
        reasoning_steps=[
            "Factor as perfect square: (x + 2)² = 0",
            "Take square root: x + 2 = 0",
            "Solve: x = -2"
        ],
        solution="x = -2 (double root)",
        domain="algebra"
    )

    print(f"\nCreated RAG store with {len(rag_store)} examples")

    # Query for similar problems
    query = "Solve x² + 6x + 9 = 0"
    print(f"\nQuery: {query}")
    print("\nRetrieving similar solution examples:")

    similar_examples = rag_store.retrieve(query, k=2, method='keyword')
    for i, ex in enumerate(similar_examples, 1):
        print(f"\n--- Example {i} ---")
        print(ex.to_prompt_string())

    # Generate few-shot prompt
    print("\n" + "=" * 50)
    print("FEW-SHOT PROMPT FOR NEW PROBLEM")
    print("=" * 50)
    few_shot_prompt = rag_store.to_few_shot_prompt(query, n_examples=2)
    print(few_shot_prompt)


def demo_composing_prompt_with_examples():
    """Demonstrate ComposingPrompt with examples."""
    print("\n\n" + "=" * 70)
    print("COMPOSING PROMPT WITH EXAMPLES")
    print("=" * 70)

    # Get math examples
    examples = get_math_examples()

    # Create a prompt with examples
    prompt = (
        ComposingPrompt()
        .cognitive_op(CognitiveOperation.DECOMPOSE)
        .focus(FocusAspect.STRUCTURE)
        .style(ReasoningStyle.SYSTEMATIC)
        .problem_context("What is 25 × 48?")
        .with_examples(list(examples.retrieve_similar("multiply", k=2)))
        .build()
    )

    print("\nGenerated prompt:")
    print("-" * 50)
    print(prompt)


def demo_mcts_with_rag():
    """Demonstrate MCTS with RAG guidance."""
    print("\n\n" + "=" * 70)
    print("MCTS WITH RAG GUIDANCE")
    print("=" * 70)

    # Setup LLM
    llm = MockLLMProvider({
        "decompose": "Let's break this down step by step",
        "analyze": "The key insight is factoring",
        "factor": "We can factor as (x - 2)(x - 3) = 0",
        "solve": "Therefore x = 2 or x = 3",
        "verify": "Checking: 2² - 5(2) + 6 = 0 ✓",
        "terminal": "YES",
        "quality": "0.92"
    })

    # Get RAG store
    rag_store = get_math_compositional_rag()

    # Create MCTS with RAG
    question = "Solve x² - 5x + 6 = 0"

    print(f"\nQuestion: {question}")
    print("Using RAG guidance from math problems database...")

    mcts = (
        ReasoningMCTS()
        .with_llm(llm)
        .with_question(question)
        .with_compositional_actions(enabled=True)
        .with_rag_store(rag_store)  # ← Enable RAG!
        .with_exploration(1.414)
        .with_max_rollout_depth(4)
    )

    print("\nRunning MCTS search...")
    mcts.search("Let's solve this quadratic equation:", simulations=20)

    # Display results
    stats = mcts.stats
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    print(f"\nTree statistics:")
    print(f"  Nodes explored: {stats['total_nodes']}")
    print(f"  Max depth: {stats['max_depth']}")
    print(f"  Best value: {stats['best_value']:.3f}")

    print(f"\nBest solution:")
    print("-" * 50)
    solution = mcts.solution
    print(solution[-400:] if len(solution) > 400 else solution)


def demo_composing_prompt_with_rag_guidance():
    """Demonstrate ComposingPrompt with RAG guidance."""
    print("\n\n" + "=" * 70)
    print("COMPOSING PROMPT WITH RAG GUIDANCE")
    print("=" * 70)

    # Get RAG store
    rag_store = get_coding_compositional_rag()

    # Use RAG to guide prompt creation
    problem = "Optimize this sorting algorithm for better performance"

    prompt = (
        ComposingPrompt()
        .problem_context(problem)
        .with_rag_guidance(rag_store)  # ← Let RAG set the dimensions!
        .build()
    )

    print(f"\nProblem: {problem}")
    print("\nRAG-guided prompt:")
    print("-" * 50)
    print(prompt)
    print("\nAction vector:", prompt.get_action_vector())


def main():
    """Run all RAG demonstrations."""
    demo_examples()
    demo_compositional_rag()
    demo_solution_rag()
    demo_composing_prompt_with_examples()
    demo_composing_prompt_with_rag_guidance()
    demo_mcts_with_rag()

    print("\n\n" + "=" * 70)
    print("✅ RAG demonstration completed!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("1. Examples enable traditional few-shot learning")
    print("2. Compositional RAG (type a) guides which reasoning dimensions to use")
    print("3. Solution RAG (type b) provides complete worked examples")
    print("4. Both can be integrated seamlessly with MCTS and ComposingPrompt")


if __name__ == "__main__":
    main()
