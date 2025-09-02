#!/usr/bin/env python3
"""
Basic usage example of MCTS-Reasoning with fluent API.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts_reasoning import ReasoningMCTS, MockLLMAdapter


def main():
    print("=" * 60)
    print("MCTS-Reasoning: Basic Usage Example")
    print("=" * 60)
    
    # Setup mock LLM for demonstration
    llm = MockLLMAdapter({
        "analyze": "Breaking down the problem into factors...",
        "solve": "Applying the distributive property: 37 * 43 = 37 * (40 + 3) = 1480 + 111 = 1591",
        "verify": "Checking: 1591 / 37 = 43 ✓",
        "terminal": "YES",
        "quality": "0.95"
    })
    
    # Create MCTS with fluent API
    question = "What is 37 * 43?"
    
    mcts = (
        ReasoningMCTS()
        .with_llm(llm)
        .with_question(question)
        .with_exploration(1.414)
        .with_max_rollout_depth(3)
        .with_metadata(domain="arithmetic", difficulty="easy")
    )
    
    print(f"\nQuestion: {question}")
    print("\nRunning MCTS search...")
    
    # Run search
    initial_state = f"Question: {question}\n\nLet's solve this step by step:"
    mcts.search(initial_state, simulations=20)
    
    # Display results
    print("\n" + "=" * 40)
    print("RESULTS")
    print("=" * 40)
    
    # Access properties
    print(f"\nTree statistics:")
    stats = mcts.stats
    for key, value in stats.items():
        if key != 'metadata':
            print(f"  {key}: {value}")
    
    print(f"\nBest solution value: {mcts.best_value:.3f}")
    print(f"Reasoning depth: {mcts.reasoning_depth}")
    
    # Get solution
    solution, confidence = mcts.solution_with_confidence
    print(f"\nSolution (confidence={confidence:.2%}):")
    print("-" * 40)
    print(solution[-500:] if len(solution) > 500 else solution)
    
    # Save tree
    print("\n" + "=" * 40)
    print("PERSISTENCE")
    print("=" * 40)
    
    mcts.save("example_tree.json")
    print("Tree saved to example_tree.json")
    
    # Load and verify
    mcts2 = ReasoningMCTS.load("example_tree.json")
    print(f"Tree loaded: {mcts2}")
    
    print("\n✅ Basic usage example completed!")


if __name__ == "__main__":
    main()