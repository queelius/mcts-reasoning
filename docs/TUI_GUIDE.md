# TUI User Guide

The MCTS-Reasoning TUI provides a Claude Code-style interactive interface for reasoning with Monte Carlo Tree Search.

## Installation

```bash
# Install with rich formatting support
pip install -e ".[tui]"

# Or without rich (plain text mode)
pip install -e .
```

## Quick Start

```bash
# Run the TUI
python mcts_tui.py

# Or if installed
mcts-reasoning-tui

# Load a saved session
python mcts_tui.py --load session.json

# Disable rich formatting
python mcts_tui.py --no-rich
```

## Basic Workflow

1. **Configure your LLM** (default is mock):
   ```
   > /model openai gpt-4
   ```

2. **Ask a question**:
   ```
   > /ask What is the sum of all prime numbers less than 100?
   ```

3. **Run simulations**:
   ```
   > /search 50
   ```

4. **View the solution**:
   ```
   > /solution
   ```

5. **Explore alternatives**:
   ```
   > /sample 5
   > /consistency 20
   ```

## Commands

### Session Management

| Command | Description | Example |
|---------|-------------|---------|
| `/ask <question>` | Start a new reasoning session | `/ask Solve x^2 + 5x + 6 = 0` |
| `/search <n>` | Run N simulations | `/search 100` |
| `/continue <n>` | Continue search (alias) | `/continue 50` |
| `/solution` | Show best solution | `/solution` |
| `/save [filename]` | Save session | `/save my_session.json` |
| `/load <filename>` | Load session | `/load my_session.json` |
| `/status` | Show current status | `/status` |

### Configuration

| Command | Description | Example |
|---------|-------------|---------|
| `/model [provider] [model] [key=value...]` | Switch LLM or show current | `/model anthropic claude-3-5-sonnet-20241022` |
| `/models` | List available models | `/models` |
| `/model-info [model]` | Show model information | `/model-info` |
| `/temperature <value>` | Set temperature (0.0-2.0) | `/temperature 0.8` |
| `/temp <value>` | Alias for temperature | `/temp 0.5` |
| `/exploration <value>` | Set exploration constant | `/exploration 1.414` |

**Supported providers:**
- `openai` - OpenAI GPT models
- `anthropic` - Anthropic Claude models
- `ollama` - Local or remote Ollama models
- `mock` - Mock LLM for testing

**Using remote Ollama:**
```
/model ollama llama2 base_url=http://192.168.0.225:11434
```

**Environment variables:**
- `OLLAMA_BASE_URL` - Default Ollama server URL
- `OLLAMA_MODEL` - Default Ollama model
- `LLM_PROVIDER` - Default provider (openai, anthropic, ollama, mock)

### Analysis

| Command | Description | Example |
|---------|-------------|---------|
| `/tree` | Visualize search tree | `/tree` |
| `/sample <n>` | Sample N diverse paths | `/sample 10` |
| `/consistency [n]` | Check solution consistency | `/consistency 20` |

### MCP Tools

| Command | Description | Example |
|---------|-------------|---------|
| `/mcp-enable` | Enable MCP integration | `/mcp-enable` |
| `/mcp-connect <name> <type>` | Connect to MCP server | `/mcp-connect python python` |
| `/mcp-list` | List connected servers | `/mcp-list` |
| `/mcp-tools` | Show available tools | `/mcp-tools` |

**Supported MCP server types:**
- `python` - Python code execution
- `web` - Web search
- `filesystem` - File operations

### Other

| Command | Description |
|---------|-------------|
| `/help` | Show help |
| `/exit` | Exit TUI |
| `/quit` | Exit TUI (alias) |

## Example Session

```
============================================================
MCTS-Reasoning TUI
============================================================

Interactive reasoning with Monte Carlo Tree Search
Type /help for available commands

Current Configuration:
Provider:     mock
Model:        default
Temperature:  0.7

> /model openai gpt-4
✓ Switched to OpenAI-gpt-4

> /temperature 0.3
✓ Temperature set to 0.3

> /ask Find all prime numbers less than 20

✓ Starting reasoning session for:
Find all prime numbers less than 20

Use /search <n> to run simulations.

> /search 50

✓ Completed 50 simulations.

Tree statistics:
  Total nodes: 50
  Max depth: 12
  Best value: 0.892

> /solution

✓ Best solution (confidence=89.20%):

The prime numbers less than 20 are:
2, 3, 5, 7, 11, 13, 17, 19

These are all the numbers less than 20 that are only divisible
by 1 and themselves.

> /sample 3

✓ Sampled 3 diverse reasoning paths:

--- Path 1 (length=8, value=89.20) ---
[solution 1 text...]

--- Path 2 (length=6, value=87.50) ---
[solution 2 text...]

--- Path 3 (length=10, value=85.30) ---
[solution 3 text...]

> /tree

MCTS Search Tree:
Root (visits=50, value=42.15)
├── Analyze the problem (visits=25, value=22.30, ucb=1.245)
│   ├── Check divisibility (visits=12, value=10.70, ucb=1.189)
│   └── List numbers 2-19 (visits=8, value=7.10, ucb=1.156)
├── Break down into subproblems (visits=15, value=13.40, ucb=1.198)
│   └── Define prime (visits=10, value=8.92, ucb=1.167)
└── Apply a specific technique (visits=10, value=6.45, ucb=1.134)

Tree Statistics:
Total Nodes:   50
Max Depth:     12
Best Value:    0.892

> /consistency 20

✓ Consistency check (20 samples):
  Confidence: 85.0%
  Support: 17/20 samples
  Clusters: 3

Most consistent solution:
2, 3, 5, 7, 11, 13, 17, 19

> /save prime_session.json

✓ Session saved to /home/user/.mcts-reasoning/sessions/prime_session.json

> /exit

Goodbye!
```

## Tips

1. **Start with lower simulations** to get quick results, then increase:
   ```
   > /search 20    # Quick exploration
   > /search 50    # Deeper search
   > /search 100   # Thorough search
   ```

2. **Use temperature for exploration**:
   - Low (0.1-0.3): More deterministic, focused
   - Medium (0.5-0.9): Balanced
   - High (1.0-1.5): More exploratory, creative

3. **Check consistency** for important decisions:
   ```
   > /consistency 50   # Sample 50 paths
   ```

4. **Enable MCP for code problems**:
   ```
   > /mcp-enable
   > /mcp-connect python python
   > /ask Write a function to check if a number is prime
   ```

5. **Save sessions** to resume later:
   ```
   > /save my_work.json
   # Later...
   > /load my_work.json
   > /continue 50
   ```

## Keyboard Shortcuts

### History Navigation
- **↑ (Up Arrow)** - Previous command
- **↓ (Down Arrow)** - Next command
- **Ctrl+R** - Search command history (reverse search)

### Completion
- **Tab** - Auto-complete commands and arguments
- **↑/↓** (in completion menu) - Navigate options

### Editing
- **Ctrl+A** - Beginning of line
- **Ctrl+E** - End of line
- **Ctrl+K** - Delete to end of line
- **Ctrl+U** - Delete to beginning of line
- **Ctrl+W** - Delete word backwards

### Control
- **Ctrl+C** - Cancel current input (doesn't exit)
- **Ctrl+D** (empty line) - Exit TUI
- **Ctrl+L** - Clear screen
- **/exit** or **/quit** - Exit TUI

See [PROMPT_FEATURES.md](PROMPT_FEATURES.md) for detailed prompt documentation.

## Configuration Files

**Session files** are saved to:
```
~/.mcts-reasoning/sessions/
```

Each session includes:
- Current question and state
- MCTS tree (separate .tree.json file)
- Configuration (model, temperature, etc.)
- Command history

**Global configuration** is stored in:
```
~/.mcts-reasoning/config.json
```

This file includes:
- Default provider and model
- Provider-specific settings (e.g., Ollama base_url)
- MCTS parameters (exploration constant, max depth)
- TUI preferences
- Recently used models

Example config.json:
```json
{
  "default_provider": "ollama",
  "providers": {
    "ollama": {
      "model": "llama2",
      "base_url": "http://192.168.0.225:11434",
      "temperature": 0.7
    },
    "openai": {
      "model": "gpt-4",
      "temperature": 0.7
    }
  },
  "mcts": {
    "exploration_constant": 1.414,
    "max_rollout_depth": 5,
    "use_compositional": true
  },
  "recent_models": [
    {"provider": "ollama", "model": "llama2"},
    {"provider": "openai", "model": "gpt-4"}
  ]
}
```

Configuration is automatically saved when you switch models or change settings.

## Troubleshooting

**No rich formatting:**
```bash
pip install rich
```

**LLM provider errors:**
```
> /model openai gpt-4
✗ Error: Failed to switch to openai/gpt-4

# Check environment:
echo $OPENAI_API_KEY
export OPENAI_API_KEY=your-key-here
```

**Out of memory with large trees:**
- Reduce simulations
- Save and restart session
- Use smaller max_rollout_depth

**MCP tools not working:**
```
> /mcp-enable
✗ Error: Failed to enable MCP

# MCP support may not be installed
pip install -e ".[mcp]"
```

## Future Features

- [ ] Command history with Up/Down arrows
- [ ] Tab completion for commands
- [ ] Interactive tree navigation
- [ ] Real-time progress indicators
- [ ] Export to various formats (markdown, HTML)
- [ ] Diff viewer for solution comparison
- [ ] Replay saved sessions step-by-step
