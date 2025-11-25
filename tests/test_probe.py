#!/usr/bin/env python3
"""
Test script for endpoint probing functionality.

This tests the new general interface for probing LLM provider endpoints,
with a focus on Ollama endpoints.
"""

from mcts_reasoning.compositional.providers import (
    OllamaProvider,
    OpenAIProvider,
    AnthropicProvider,
    MockLLMProvider
)

def test_ollama_probe():
    """Test probing a remote Ollama endpoint."""
    print("=" * 60)
    print("Testing Ollama Endpoint Probing")
    print("=" * 60)

    # Test with remote endpoint
    remote_url = "http://192.168.0.225:11434"
    print(f"\nProbing: {remote_url}")

    result = OllamaProvider.probe_endpoint(remote_url)

    print(f"Available: {result.get('available', False)}")

    if result.get('available'):
        print(f"Models found: {result.get('model_count', 0)}")
        print("\nModels:")
        for i, model in enumerate(result.get('models', []), 1):
            print(f"  {i}. {model}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

    print()
    # Assert that we got a valid result structure (doesn't require endpoint to be available)
    assert isinstance(result, dict), "Probe result should be a dictionary"
    assert 'available' in result, "Probe result should contain 'available' key"

def test_local_ollama_probe():
    """Test probing local Ollama endpoint."""
    print("=" * 60)
    print("Testing Local Ollama Endpoint Probing")
    print("=" * 60)

    local_url = "http://localhost:11434"
    print(f"\nProbing: {local_url}")

    result = OllamaProvider.probe_endpoint(local_url, timeout=2.0)

    print(f"Available: {result.get('available', False)}")

    if result.get('available'):
        print(f"Models found: {result.get('model_count', 0)}")
        print("\nModels:")
        for i, model in enumerate(result.get('models', []), 1):
            print(f"  {i}. {model}")
    else:
        print(f"Error: {result.get('error', 'Unknown error')}")

    print()

def test_other_providers():
    """Test that other providers correctly report not supporting probing."""
    print("=" * 60)
    print("Testing Other Providers (should not support probing)")
    print("=" * 60)

    providers = [
        ("OpenAI", OpenAIProvider),
        ("Anthropic", AnthropicProvider),
        ("Mock", MockLLMProvider),
    ]

    for name, provider_class in providers:
        print(f"\n{name}:")
        result = provider_class.probe_endpoint("http://example.com")
        print(f"  Available: {result.get('available', False)}")
        print(f"  Error: {result.get('error', 'N/A')}")

    print()

def test_ollama_instance_methods():
    """Test OllamaProvider instance methods."""
    print("=" * 60)
    print("Testing OllamaProvider Instance Methods")
    print("=" * 60)

    # Create provider pointing to remote endpoint
    remote_url = "http://192.168.0.225:11434"
    print(f"\nCreating OllamaProvider with base_url={remote_url}")

    try:
        provider = OllamaProvider(model="llama3.2", base_url=remote_url)
        print(f"Provider created: {provider.get_provider_name()}")
        print(f"Base URL: {provider.base_url}")

        # Test is_available
        print(f"\nTesting is_available()...")
        available = provider.is_available()
        print(f"Available: {available}")

        if available:
            # Test list_models
            print(f"\nTesting list_models()...")
            models = provider.list_models()
            print(f"Found {len(models)} models:")
            for model in models[:5]:  # Show first 5
                name = model.get('name', 'unknown')
                size = model.get('size', 0) / (1024**3)
                print(f"  - {name} ({size:.1f}GB)")

            if len(models) > 5:
                print(f"  ... and {len(models) - 5} more")

            # Test get_model_info
            print(f"\nTesting get_model_info()...")
            info = provider.get_model_info()
            if info:
                print(f"Model info retrieved for: {provider.model}")
                print(f"  Keys: {list(info.keys())[:5]}")
            else:
                print(f"No info available for model: {provider.model}")

    except Exception as e:
        print(f"Error: {e}")

    print()

if __name__ == "__main__":
    print("\nEndpoint Probing Interface Test Suite")
    print("=" * 60)
    print()

    # Test remote Ollama (no longer returns value)
    test_ollama_probe()

    # Test local Ollama (might not be available)
    test_local_ollama_probe()

    # Test other providers
    test_other_providers()

    # Test instance methods (will check availability internally)
    test_ollama_instance_methods()

    print("\n" + "=" * 60)
    print("Test suite complete!")
    print("=" * 60)
