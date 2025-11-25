#!/usr/bin/env python3
"""Check what model is actually being used."""

from mcts_reasoning.compositional.providers import OllamaProvider

# Create with just "llama3.2" (no :latest)
provider = OllamaProvider(
    model="llama3.2",
    base_url="http://192.168.0.225:11434"
)

print(f"Provider name: {provider.get_provider_name()}")
print(f"Model attribute: {provider.model}")
print(f"Base URL: {provider.base_url}")
print()

# List available models
models = provider.list_models()
print(f"Available models on endpoint:")
for m in models:
    name = m.get('name', 'unknown')
    if 'llama3.2' in name.lower():
        print(f"  - {name} ⭐")
    else:
        print(f"  - {name}")
