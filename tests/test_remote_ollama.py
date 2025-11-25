#!/usr/bin/env python3
"""
Test remote Ollama connection directly.
"""

from mcts_reasoning.compositional.providers import OllamaProvider

def test_remote_generation():
    """Test that we can actually generate text from the remote Ollama."""
    print("Testing remote Ollama connection...")
    print("=" * 60)

    # Create provider
    provider = OllamaProvider(
        model="llama3.2",
        base_url="http://192.168.0.225:11434"
    )

    print(f"Provider: {provider.get_provider_name()}")
    print(f"Base URL: {provider.base_url}")
    print(f"Model: {provider.model}")
    print()

    # Test availability
    print("Checking availability...")
    available = provider.is_available()
    print(f"Available: {available}")

    if not available:
        print("❌ Provider not available!")
        return

    print()

    # Test simple generation
    print("Testing simple generation...")
    print("Prompt: 'What is 2+2? Answer briefly.'")
    print()

    try:
        response = provider.generate(
            "What is 2+2? Answer briefly.",
            max_tokens=100,
            temperature=0.7
        )
        print(f"Response ({len(response)} chars):")
        print("-" * 60)
        print(response)
        print("-" * 60)
        print()
        print("✅ Generation successful!")

    except Exception as e:
        print(f"❌ Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Test reasoning-style generation
    print()
    print("=" * 60)
    print("Testing reasoning-style generation...")
    print("Prompt: 'Think step by step: What are the prime numbers less than 20?'")
    print()

    try:
        response = provider.generate(
            "Think step by step: What are the prime numbers less than 20?",
            max_tokens=500,
            temperature=0.7
        )
        print(f"Response ({len(response)} chars):")
        print("-" * 60)
        print(response)
        print("-" * 60)
        print()
        print("✅ Reasoning generation successful!")

    except Exception as e:
        print(f"❌ Reasoning generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_remote_generation()
