# Remote Ollama Setup Guide

This guide shows how to use MCTS-Reasoning with a remote Ollama server.

## Quick Start

Your remote Ollama server at `192.168.0.225:11434` is already configured and has many models available!

### Using the TUI (Recommended)

1. **Start the TUI:**
   ```bash
   python mcts_tui.py
   ```

2. **Connect to your remote server:**
   ```
   > /model ollama llama3.2:latest base_url=http://192.168.0.225:11434
   ```

3. **List available models:**
   ```
   > /models
   ```

4. **Show model info:**
   ```
   > /model-info
   ```

5. **Start reasoning:**
   ```
   > /ask What is the sum of the first 10 prime numbers?
   > /search 50
   > /solution
   ```

### Available Models on Your Server

Your server has 33+ models including:
- **qwen3** series: 0.6b, 1.7b, 4b, 8b, 14b, 30b
- **llama3** series: 3.1, 3.2
- **deepseek-r1** series: 1.5b, 7b, 8b, 14b
- **gemma3** series: 1b, 4b, 12b
- **phi4**: standard and mini
- **mistral**: latest
- **gpt-oss**: 12.8GB
- And many more!

## Configuration Methods

### Method 1: Environment Variables

Set these in your shell:
```bash
export OLLAMA_BASE_URL=http://192.168.0.225:11434
export OLLAMA_MODEL=llama3.2:latest
export LLM_PROVIDER=ollama
```

Then start the TUI:
```bash
python mcts_tui.py
# It will automatically connect to your remote server!
```

### Method 2: Config File

Edit `~/.mcts-reasoning/config.json`:
```json
{
  "default_provider": "ollama",
  "providers": {
    "ollama": {
      "model": "llama3.2:latest",
      "base_url": "http://192.168.0.225:11434",
      "temperature": 0.7
    }
  }
}
```

### Method 3: Programmatic

```python
from mcts_reasoning import ReasoningMCTS, OllamaProvider

# Create provider
ollama = OllamaProvider(
    model="llama3.2:latest",
    base_url="http://192.168.0.225:11434"
)

# Check available models
models = ollama.list_models()
for model in models:
    print(f"  - {model['name']}")

# Use with MCTS
mcts = (
    ReasoningMCTS()
    .with_llm(ollama)
    .with_question("Your question here")
    .with_exploration(1.414)
)

mcts.search("Let me think...", simulations=50)
print(f"Solution: {mcts.solution}")
```

## TUI Commands

### Model Management
- `/model` - Show current model
- `/models` - List available models on the server
- `/model-info [model]` - Show detailed model information
- `/model ollama <model-name>` - Switch to a different model

### Configuration
- `/temperature 0.7` - Set generation temperature
- `/exploration 1.414` - Set MCTS exploration constant

### Reasoning
- `/ask <question>` - Start a new reasoning session
- `/search <n>` - Run N simulations
- `/solution` - Show best solution
- `/tree` - Visualize search tree
- `/sample <n>` - Sample N diverse reasoning paths
- `/consistency <n>` - Check solution consistency

### Session Management
- `/save [filename]` - Save session
- `/load <filename>` - Load session
- `/status` - Show current status

## Example Session

```bash
$ python mcts_tui.py

> /model ollama llama3.2:latest base_url=http://192.168.0.225:11434
✓ Switched to Ollama-llama3.2:latest

> /models
Available models on http://192.168.0.225:11434:
  1. qwen2.5vl:latest (5.6GB)
  2. gemma3:270m (0.3GB)
  3. llama3.2:latest (1.9GB)
  [... more models ...]

> /model-info
Model: llama3.2:latest
Template: ...
Parameters:
  num_ctx: 8192
  temperature: 0.8
  ...

> /ask What are the first 10 prime numbers?

> /search 30
✓ Completed 30 simulations.

Tree statistics:
  Total nodes: 30
  Max depth: 8
  Best value: 0.945

> /solution
✓ Best solution (confidence=94.50%):

The first 10 prime numbers are:
2, 3, 5, 7, 11, 13, 17, 19, 23, 29

These are numbers greater than 1 that are only divisible by 1 and themselves.

> /sample 3
✓ Sampled 3 diverse reasoning paths:

--- Path 1 (length=6, value=94.50) ---
[... reasoning path 1 ...]

--- Path 2 (length=8, value=92.10) ---
[... reasoning path 2 ...]

--- Path 3 (length=7, value=89.30) ---
[... reasoning path 3 ...]

> /save prime_session.json
✓ Session saved to /home/user/.mcts-reasoning/sessions/prime_session.json
```

## Tips

1. **Start with smaller models** for faster iteration:
   - `gemma3:1b` (0.8GB) - Very fast
   - `llama3.2:latest` (1.9GB) - Good balance
   - `mistral:latest` (3.8GB) - Higher quality

2. **Use /models to explore** what's available on your server

3. **Check model info** before using with `/model-info <model-name>`

4. **Adjust temperature** based on task:
   - Low (0.1-0.3): Deterministic, factual tasks
   - Medium (0.5-0.9): Balanced reasoning
   - High (1.0-1.5): Creative, exploratory tasks

5. **Save your sessions** for later analysis:
   ```
   /save my_reasoning_session.json
   ```

## Troubleshooting

**Server not available:**
```
> /model ollama llama3.2:latest base_url=http://192.168.0.225:11434
✗ Failed to switch to ollama
```
- Check if the Ollama server is running
- Verify the IP address and port
- Check firewall settings

**Model not found (404 error):**
```
RuntimeError: Ollama generation failed: 404 Client Error
```
- Use `/models` to see what's actually available
- Model names must match exactly (including `:latest` suffix)
- Example: Use `llama3.2:latest`, not just `llama3.2`

**Config not persisting:**
- Config is automatically saved to `~/.mcts-reasoning/config.json`
- Check file permissions
- Verify the directory exists

## Next Steps

1. Try the example script: `python examples/remote_ollama.py`
2. Experiment with different models for your use case
3. Use MCP integration for tool access (see docs/MCP_INTEGRATION.md)
4. Save and share interesting reasoning sessions

Enjoy exploring reasoning with your remote Ollama server!
