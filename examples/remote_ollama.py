"""
Example: Using MCTS-Reasoning with a remote Ollama server

This example shows how to connect to a remote Ollama instance
and use it for reasoning tasks.
"""

from mcts_reasoning import ReasoningMCTS, OllamaProvider

# Method 1: Direct instantiation with base_url
print("=" * 60)
print("Method 1: Direct instantiation")
print("=" * 60)

# Note: Use a model that's actually available on your server
# Check available models first with list_models()
ollama = OllamaProvider(
    model="llama3.2:latest",  # Using llama3.2 instead of llama2
    base_url="http://192.168.0.225:11434"
)

print(f"Provider: {ollama.get_provider_name()}")
print(f"Base URL: {ollama.base_url}")
print(f"Model: {ollama.model}")

# Check if server is available
if ollama.is_available():
    print("✓ Server is available")

    # List available models
    models = ollama.list_models()
    if models:
        print("\nAvailable models:")
        for model in models:
            name = model.get("name", "unknown")
            size = model.get("size", 0) / (1024**3)  # GB
            print(f"  - {name} ({size:.1f}GB)")
else:
    print("✗ Server is not available")

print("\n" + "=" * 60)
print("Method 2: Using environment variables")
print("=" * 60)

# Set environment variables:
# export OLLAMA_BASE_URL=http://192.168.0.225:11434
# export OLLAMA_MODEL=llama2

# Then in Python:
# ollama = OllamaProvider()  # Reads from environment

print("""
Set these environment variables:
  export OLLAMA_BASE_URL=http://192.168.0.225:11434
  export OLLAMA_MODEL=llama2

Then create provider:
  ollama = OllamaProvider()  # Auto-detects from env
""")

print("\n" + "=" * 60)
print("Method 3: Using config file")
print("=" * 60)

from mcts_reasoning import get_config

config = get_config()

# Set Ollama config
config.set_provider_config("ollama", {
    "model": "llama3.2:latest",  # Use a model available on your server
    "base_url": "http://192.168.0.225:11434",
    "temperature": 0.7
}, save=True)

print(f"Config saved to: {config.config_file}")
print("\nNow you can use the TUI:")
print("  python mcts_tui.py")
print("  > /model ollama llama2")
print("  > /models")
print("  > /model-info")

print("\n" + "=" * 60)
print("Method 4: Using MCTS with remote Ollama")
print("=" * 60)

# Create MCTS with remote Ollama (if server is available)
if ollama.is_available():
    question = "What are the first 5 prime numbers?"

    mcts = (
        ReasoningMCTS()
        .with_llm(ollama)
        .with_question(question)
        .with_exploration(1.414)
        .with_max_rollout_depth(3)
    )

    print(f"\nQuestion: {question}")
    print("Running 10 simulations...")

    initial_state = "Let me think about this systematically."
    mcts.search(initial_state, simulations=10)

    print(f"\nSolution: {mcts.solution}")
    print(f"Confidence: {mcts.best_value:.2%}")
else:
    print("\nSkipping MCTS example - server not available")

print("\n" + "=" * 60)
print("TUI Examples")
print("=" * 60)

print("""
# Switch to remote Ollama with base_url parameter
/model ollama llama3.2:latest base_url=http://192.168.0.225:11434

# Show current model
/model

# List available models on the server
/models

# Show detailed model information
/model-info

# Switch to a different model on the same server
# (use a model name from the /models list)
/model ollama mistral:latest

# Ask a question
/ask What is the sum of the first 10 prime numbers?

# Run search
/search 50

# View solution
/solution
""")
