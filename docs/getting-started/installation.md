# Installation

This guide covers installing MCTS-Reasoning and its dependencies.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- Git (for source installation)

## Installation Options

### Option 1: Install from Source (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/mcts-reasoning.git
cd mcts-reasoning

# Basic installation
pip install -e .
```

### Option 2: Install with Extras

MCTS-Reasoning supports several optional dependency groups:

#### TUI Support (Recommended for Interactive Use)

```bash
pip install -e ".[tui]"
```

This includes:
- `rich` - Beautiful terminal formatting
- `prompt_toolkit` - Advanced input with history and completion

#### LLM Provider Support

Install specific LLM provider dependencies:

```bash
# OpenAI
pip install -e ".[openai]"

# Anthropic
pip install -e ".[anthropic]"

# Ollama
pip install -e ".[ollama]"
```

#### Development Dependencies

```bash
pip install -e ".[dev]"
```

This includes:
- `pytest` - Testing framework
- `pytest-cov` - Coverage reporting
- `black` - Code formatter
- `flake8` - Linter
- `mypy` - Type checker

#### All Dependencies

```bash
pip install -e ".[all]"
```

This installs everything: TUI support and all LLM providers.

## Verifying Installation

After installation, verify everything is working:

```bash
# Check that the package is installed
python -c "import mcts_reasoning; print(mcts_reasoning.__version__)"

# Try running the TUI (if installed with [tui])
mcts-tui --help

# Try running the shell (if installed with [tui])
mcts-shell --help
```

## LLM Provider Setup

### OpenAI

```bash
# Set your API key
export OPENAI_API_KEY=your-key-here

# Set provider (optional, auto-detected)
export LLM_PROVIDER=openai
```

### Anthropic

```bash
# Set your API key
export ANTHROPIC_API_KEY=your-key-here

# Set provider (optional, auto-detected)
export LLM_PROVIDER=anthropic
```

### Ollama

```bash
# Install Ollama (if not already installed)
# Visit: https://ollama.ai

# Pull a model
ollama pull llama2

# Set provider
export LLM_PROVIDER=ollama
export OLLAMA_MODEL=llama2

# For remote Ollama server
export OLLAMA_BASE_URL=http://your-server:11434
```

See [Remote Ollama Setup](../guides/remote-ollama-setup.md) for detailed remote configuration.

### Mock Provider (For Testing)

```bash
# No setup required
export LLM_PROVIDER=mock
```

## Configuration File

MCTS-Reasoning stores configuration in `~/.mcts-reasoning/config.json`. This file is created automatically on first run.

To view or edit configuration:

```bash
# View config
cat ~/.mcts-reasoning/config.json

# Edit with your favorite editor
nano ~/.mcts-reasoning/config.json
```

## Troubleshooting

### Import Errors

If you get import errors, ensure you're in the correct directory:

```bash
# Install from the repository root
cd /path/to/mcts-reasoning
pip install -e .
```

### Missing Dependencies

If you get "module not found" errors:

```bash
# Install with all dependencies
pip install -e ".[all]"
```

### Permission Issues

If you encounter permission errors:

```bash
# Use --user flag
pip install --user -e .

# Or use a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e ".[all]"
```

### API Key Issues

Ensure your environment variables are set:

```bash
# Check environment variables
echo $OPENAI_API_KEY
echo $ANTHROPIC_API_KEY
echo $LLM_PROVIDER

# Add to your shell profile for persistence
echo 'export OPENAI_API_KEY=your-key' >> ~/.bashrc
source ~/.bashrc
```

## Next Steps

- [Quick Start Guide](quick-start.md) - Get started with your first reasoning session
- [Examples](examples.md) - Explore code examples
- [Shell Guide](../guides/shell-guide.md) - Learn the interactive shell
- [TUI Guide](../guides/tui-guide.md) - Learn the terminal interface
