#!/usr/bin/env python3
"""Quick test with fewer simulations."""

from mcts_reasoning.compositional.providers import OllamaProvider
from mcts_reasoning.reasoning import ReasoningMCTS

# Create provider
provider = OllamaProvider(
    model="llama3.2",
    base_url="http://192.168.0.225:11434"
)

print(f"Provider: {provider.get_provider_name()}")
print()

# Test with a greeting (like the user did)
print("=" * 70)
print("Testing with greeting 'hi' (3 simulations)")
print("=" * 70)

mcts = (
    ReasoningMCTS()
    .with_llm(provider)
    .with_question("hi")
    .with_exploration(1.414)
    .with_max_rollout_depth(3)  # Shorter for speed
    .with_compositional_actions(enabled=True)
)

initial_state = "Question: hi\n\nLet me think about this systematically."
print(f"Initial state: {initial_state}\n")

print("Running search...")
mcts.search(initial_state, simulations=3)

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
