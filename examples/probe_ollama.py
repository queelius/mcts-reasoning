#!/usr/bin/env python3
"""
Example: Probing and Using Remote Ollama Endpoints

This example demonstrates:
1. How to probe a remote Ollama endpoint for available models
2. How to create an OllamaProvider connected to a remote endpoint
3. How to verify the connection works
4. How to use the provider with MCTS reasoning
"""

from mcts_reasoning.compositional.providers import OllamaProvider, get_llm
from mcts_reasoning.reasoning import ReasoningMCTS

def probe_and_list_models(endpoint_url: str):
    """Probe an Ollama endpoint and list available models."""
    print(f"Probing Ollama endpoint: {endpoint_url}")
    print("=" * 60)

    # Probe the endpoint
    result = OllamaProvider.probe_endpoint(endpoint_url)

    if not result['available']:
        print(f"❌ Endpoint not available: {result['error']}")
        return None

    print(f"✅ Endpoint available!")
    print(f"Models found: {result['model_count']}")
    print()

    models = result['models']
    if models:
        print("Available models:")
        for i, model in enumerate(models, 1):
            print(f"  {i:2d}. {model}")
        print()

    return models

def create_and_test_provider(endpoint_url: str, model_name: str):
    """Create an OllamaProvider and test it."""
    print(f"Creating OllamaProvider:")
    print(f"  Endpoint: {endpoint_url}")
    print(f"  Model: {model_name}")
    print("=" * 60)

    # Create provider
    provider = OllamaProvider(model=model_name, base_url=endpoint_url)

    # Check if available
    if not provider.is_available():
        print("❌ Provider not available")
        return None

    print(f"✅ Provider created: {provider.get_provider_name()}")
    print(f"   Base URL: {provider.base_url}")
    print()

    # Test basic generation
    print("Testing basic generation...")
    try:
        response = provider.generate("What is 2+2?", max_tokens=50, temperature=0.0)
        print(f"✅ Response: {response[:100]}")
        print()
    except Exception as e:
        print(f"❌ Generation failed: {e}")
        return None

    return provider

def use_with_mcts(provider: OllamaProvider, question: str):
    """Use the provider with MCTS reasoning."""
    print("Using provider with MCTS reasoning")
    print("=" * 60)

    # Create MCTS instance
    mcts = (
        ReasoningMCTS()
        .with_llm(provider)
        .with_question(question)
        .with_exploration(1.414)
        .with_max_rollout_depth(3)  # Keep it short for demo
    )

    # Run search
    print(f"Question: {question}")
    print("\nRunning MCTS search (10 simulations)...")

    initial_state = f"Question: {question}\n\nLet me think about this step by step."
    mcts.search(initial_state, simulations=10)

    # Show results
    print(f"\nSearch complete!")
    print(f"  Total nodes: {mcts.stats['total_nodes']}")
    print(f"  Max depth: {mcts.stats['max_depth']}")
    print(f"  Best value: {mcts.stats['best_value']:.3f}")

    # Get best solution
    solution = mcts.solution
    print(f"\n{'=' * 60}")
    print("Best solution:")
    print(f"{'=' * 60}")
    print(solution)
    print()

def main():
    # Configuration - adjust these for your setup
    REMOTE_ENDPOINT = "http://192.168.0.225:11434"
    # You can also try local: "http://localhost:11434"

    print("\n" + "=" * 70)
    print("Ollama Endpoint Probing Example")
    print("=" * 70)
    print()

    # Step 1: Probe endpoint
    models = probe_and_list_models(REMOTE_ENDPOINT)
    if not models:
        print("No models found. Exiting.")
        return

    # Step 2: Choose a model (use first available, or specify one)
    # Common choices: llama3.2, llama3.1, mistral, phi4-mini, etc.
    chosen_model = "llama3.2:latest"  # or models[0] for first available

    if chosen_model not in models:
        print(f"Warning: {chosen_model} not in available models.")
        print(f"Using first available model: {models[0]}")
        chosen_model = models[0]

    print(f"Using model: {chosen_model}")
    print()

    # Step 3: Create and test provider
    provider = create_and_test_provider(REMOTE_ENDPOINT, chosen_model)
    if not provider:
        print("Failed to create provider. Exiting.")
        return

    # Step 4: Use with MCTS
    question = "What are the prime numbers less than 20?"
    use_with_mcts(provider, question)

    # Step 5: Show how to get model info
    print("=" * 60)
    print("Getting model information...")
    print("=" * 60)

    info = provider.get_model_info()
    if info:
        print(f"Model: {info.get('modelfile', 'N/A')[:50]}...")
        print(f"Template length: {len(info.get('template', ''))} chars")

        params = info.get('parameters', {})
        if params:
            print("\nSome parameters:")
            for key, value in list(params.items())[:5]:
                print(f"  {key}: {value}")
    else:
        print("No model info available")

    print("\n" + "=" * 70)
    print("Example complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
