#!/usr/bin/env python3
"""
Example of using consistency checking to find robust solutions.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts_reasoning import ReasoningMCTS, MockLLMAdapter
import json


def main():
    print("=" * 60)
    print("MCTS-Reasoning: Consistency Checking Example")
    print("=" * 60)
    
    # Setup mock LLM with some variability
    llm = MockLLMAdapter({
        "river": "We need to get everyone across safely",
        "constraint": "The boat holds at most 2 people",
        "analyze": "Father or mother must be present when with opposite gender children",
        "solve": "1) Father and son cross, 2) Father returns alone...",
        "alternative": "1) Mother and daughter cross first instead...",
        "verify": "All constraints satisfied",
        "terminal": "YES",
        "0.": "0.85"  # Variable quality scores
    })
    
    # Problem with multiple valid solutions
    question = """
    A family needs to cross a river. The family consists of a father, mother, 
    two sons, and two daughters. They have a boat that can hold at most 2 people.
    The father cannot be with the daughters without the mother present.
    The mother cannot be with the sons without the father present.
    How can they all cross the river?
    """
    
    print(f"\nProblem: River crossing puzzle")
    print("(This has multiple valid solution sequences)")
    
    # Create MCTS
    mcts = (
        ReasoningMCTS()
        .with_llm(llm)
        .with_question(question)
        .with_exploration(1.5)  # Moderate exploration
        .with_max_rollout_depth(4)
    )
    
    # Run search with more simulations for diversity
    print("\nRunning MCTS search (100 simulations)...")
    initial_state = f"Problem: {question}\n\nLet's solve this step by step:"
    mcts.search(initial_state, simulations=100)
    
    print(f"Tree size: {mcts.stats['total_nodes']} nodes")
    print(f"Max depth: {mcts.reasoning_depth} steps")
    
    # ========== Single Best Solution ==========
    print("\n" + "=" * 40)
    print("SINGLE BEST SOLUTION")
    print("=" * 40)
    
    solution = mcts.solution
    print("\nBest solution (by value):")
    print("-" * 40)
    print(solution[-300:] if len(solution) > 300 else solution)
    print(f"\nValue: {mcts.best_value:.3f}")
    
    # ========== Consistency Analysis ==========
    print("\n" + "=" * 40)
    print("CONSISTENCY ANALYSIS")
    print("=" * 40)
    
    # Check with different temperatures
    temperatures = [0.5, 1.0, 1.5]
    
    for temp in temperatures:
        print(f"\n--- Temperature = {temp} ---")
        
        result = mcts.check_consistency(n_samples=20, temperature=temp)
        
        print(f"Samples: {result['total_samples']}")
        print(f"Unique solutions: {len(result['clusters'])}")
        print(f"Most common: appears {result['support']} times ({result['confidence']:.1%})")
        
        # Show solution distribution
        print("\nSolution distribution:")
        for i, cluster in enumerate(result['clusters'][:3], 1):
            pct = cluster['count'] / result['total_samples'] * 100
            print(f"  Solution {i}: {cluster['count']} times ({pct:.0f}%)")
    
    # ========== Robust Solution ==========
    print("\n" + "=" * 40)
    print("ROBUST SOLUTION (Most Consistent)")
    print("=" * 40)
    
    # Get the most robust solution across different sampling strategies
    robust_result = mcts.check_consistency(n_samples=30, temperature=0.8)
    
    print(f"\nAfter 30 samples at temperature 0.8:")
    print(f"Most robust solution (appears {robust_result['support']} times):")
    print("-" * 40)
    print(robust_result['solution'][-400:] if len(robust_result['solution']) > 400 
          else robust_result['solution'])
    print("-" * 40)
    print(f"Confidence: {robust_result['confidence']:.1%}")
    
    # ========== Save Results ==========
    print("\n" + "=" * 40)
    print("SAVING ANALYSIS")
    print("=" * 40)
    
    # Save tree and analysis
    mcts.save("consistency_tree.json")
    
    # Save consistency analysis
    analysis = {
        "question": question,
        "tree_stats": mcts.stats,
        "consistency_result": {
            "solution": robust_result['solution'],
            "confidence": robust_result['confidence'],
            "support": robust_result['support'],
            "total_samples": robust_result['total_samples'],
            "num_clusters": len(robust_result['clusters'])
        }
    }
    
    with open("consistency_analysis.json", "w") as f:
        json.dump(analysis, f, indent=2)
    
    print("Tree saved to: consistency_tree.json")
    print("Analysis saved to: consistency_analysis.json")
    
    # ========== Export to Markdown ==========
    print("\n" + "=" * 40)
    print("MARKDOWN EXPORT")
    print("=" * 40)
    
    markdown = mcts.to_markdown()
    with open("consistency_report.md", "w") as f:
        f.write(markdown)
    
    print("Report exported to: consistency_report.md")
    
    print("\n✅ Consistency checking example completed!")
    print("\nKey insights:")
    print("- Multiple valid solutions exist for this problem")
    print("- Lower temperature favors exploitation (less diversity)")
    print("- Higher temperature explores more alternatives")
    print("- Consistency checking helps find robust solutions")


if __name__ == "__main__":
    main()