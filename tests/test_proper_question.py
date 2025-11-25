#!/usr/bin/env python3
"""Test with a proper reasoning question."""

from mcts_reasoning.compositional.providers import OllamaProvider
from mcts_reasoning.reasoning import ReasoningMCTS

# Create provider
provider = OllamaProvider(
    model="llama3.2",
    base_url="http://192.168.0.225:11434"
)

print(f"Provider: {provider.get_provider_name()}")
print()

# Test with an actual reasoning question
print("=" * 70)
print("Testing with proper reasoning question (5 simulations)")
print("=" * 70)

mcts = (
    ReasoningMCTS()
    .with_llm(provider)
    .with_question("What are the prime numbers less than 20?")
    .with_exploration(1.414)
    .with_max_rollout_depth(4)
    .with_compositional_actions(enabled=True)
)

question = "What are the prime numbers less than 20?"
initial_state = f"Question: {question}\n\nLet me think about this systematically."

print(f"Question: {question}")
print()

print("Running search (this may take 30-60 seconds)...")
mcts.search(initial_state, simulations=5)

stats = mcts.stats
print(f"\nStats:")
print(f"  Nodes: {stats['total_nodes']}")
print(f"  Depth: {stats['max_depth']}")
print(f"  Best value: {stats['best_value']:.3f}")

solution = mcts.solution
print(f"\nSolution ({len(solution)} chars):")
print("=" * 70)
print(solution)
print("=" * 70)
