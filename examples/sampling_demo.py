#!/usr/bin/env python3
"""
Demonstration of various sampling strategies in MCTS-Reasoning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts_reasoning import ReasoningMCTS, MockLLMAdapter


def main():
    print("=" * 60)
    print("MCTS-Reasoning: Sampling Strategies Demo")
    print("=" * 60)
    
    # Setup mock LLM with varied responses
    llm = MockLLMAdapter()
    
    # Create and run MCTS
    question = "What are the key factors in designing a scalable web service?"
    
    mcts = (
        ReasoningMCTS()
        .with_llm(llm)
        .with_question(question)
        .with_exploration(2.0)  # High exploration
        .search(f"Question: {question}\n\nKey factors:", simulations=50)
    )
    
    print(f"\nQuestion: {question}")
    print(f"Tree size: {mcts.stats['total_nodes']} nodes")
    
    # ========== Temperature-based Sampling ==========
    print("\n" + "=" * 40)
    print("1. TEMPERATURE-BASED SAMPLING")
    print("=" * 40)
    
    # Greedy (temperature = 0)
    print("\nGreedy (temperature=0):")
    path = mcts.sample(n=1, temperature=0.0, strategy="value")
    print(f"  Value: {path.total_value:.3f}")
    print(f"  Length: {path.length} steps")
    
    # Medium temperature
    print("\nMedium randomness (temperature=0.5):")
    paths = mcts.sample(n=3, temperature=0.5, strategy="value")
    for i, path in enumerate(paths, 1):
        print(f"  Path {i}: value={path.total_value:.3f}, length={path.length}")
    
    # High temperature
    print("\nHigh randomness (temperature=2.0):")
    paths = mcts.sample(n=3, temperature=2.0, strategy="value")
    for i, path in enumerate(paths, 1):
        print(f"  Path {i}: value={path.total_value:.3f}, length={path.length}")
    
    # ========== Visit-based Sampling ==========
    print("\n" + "=" * 40)
    print("2. VISIT-BASED SAMPLING (MCTS-style)")
    print("=" * 40)
    
    paths = mcts.sample(n=5, strategy="visits")
    print("\nSampling proportional to visit counts:")
    for i, path in enumerate(paths, 1):
        print(f"  Path {i}: visits={path.total_visits}, value={path.total_value:.3f}")
    
    # ========== Diverse Sampling ==========
    print("\n" + "=" * 40)
    print("3. DIVERSE SAMPLING")
    print("=" * 40)
    
    diverse_paths = mcts.sample(n=3, strategy="diverse")
    print("\nDiverse paths (syntactically different):")
    for i, path in enumerate(diverse_paths, 1):
        print(f"\nPath {i}:")
        print(f"  Actions: {path.actions[:3]}...")
        print(f"  Value: {path.total_value:.3f}")
    
    # ========== Top-K Solutions ==========
    print("\n" + "=" * 40)
    print("4. TOP-K SOLUTIONS")
    print("=" * 40)
    
    top_solutions = mcts.get_top_solutions(k=3)
    print("\nTop 3 solutions by value:")
    for i, solution in enumerate(top_solutions, 1):
        preview = solution[-100:] if len(solution) > 100 else solution
        print(f"\n{i}. {preview}")
    
    # ========== Consistency Checking ==========
    print("\n" + "=" * 40)
    print("5. CONSISTENCY CHECKING")
    print("=" * 40)
    
    consistency = mcts.check_consistency(n_samples=10, temperature=0.7)
    
    print(f"\nConsistency analysis (10 samples):")
    print(f"  Most consistent solution appears in {consistency['support']}/{consistency['total_samples']} samples")
    print(f"  Confidence: {consistency['confidence']:.1%}")
    print(f"  Number of unique solutions: {len(consistency['clusters'])}")
    
    # Show cluster distribution
    print("\nSolution clusters:")
    for i, cluster in enumerate(consistency['clusters'][:3], 1):
        print(f"  Cluster {i}: {cluster['count']} occurrences")
    
    # ========== Comparison of Strategies ==========
    print("\n" + "=" * 40)
    print("6. STRATEGY COMPARISON")
    print("=" * 40)
    
    print("\nComparing average values across strategies:")
    
    # Value-based
    value_paths = mcts.sample(n=10, temperature=1.0, strategy="value")
    avg_value = sum(p.total_value for p in value_paths) / len(value_paths)
    print(f"  Value-based (temp=1.0): {avg_value:.3f}")
    
    # Visit-based
    visit_paths = mcts.sample(n=10, strategy="visits")
    avg_visits = sum(p.total_value for p in visit_paths) / len(visit_paths)
    print(f"  Visit-based:            {avg_visits:.3f}")
    
    # Diverse
    diverse_paths = mcts.sample(n=5, strategy="diverse")
    avg_diverse = sum(p.total_value for p in diverse_paths) / len(diverse_paths)
    print(f"  Diverse:                {avg_diverse:.3f}")
    
    print("\n✅ Sampling demo completed!")


if __name__ == "__main__":
    main()