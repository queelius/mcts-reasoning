# MCTS-Reasoning

**Monte Carlo Tree Search for LLM-based reasoning with advanced compositional prompting and MCP integration.**

Combines systematic tree search with sophisticated prompt engineering to enable structured, exploratory reasoning across multiple solution paths.

---

## What is MCTS-Reasoning?

MCTS-Reasoning brings the power of Monte Carlo Tree Search to Large Language Model reasoning. Instead of generating a single reasoning path, it:

- Explores multiple solution approaches simultaneously
- Balances exploration of new paths with exploitation of promising ones
- Uses advanced compositional prompting for rich reasoning strategies
- Integrates external tools via Model Context Protocol (MCP)
- Provides both interactive and programmatic interfaces

## Key Features

### Core Capabilities

- **Pure MCTS Implementation**: Clean, canonical MCTS with proper Selection, Expansion, Rollout, and Backpropagation phases
- **Advanced Compositional Prompting**: Rich action space with 5 dimensions enabling 30,000+ action combinations
- **MCP Tool Integration**: Automatic access to external tools (Python, web search, file operations, custom tools)
- **Multiple Sampling Strategies**: Value-based, visit-based, diverse, and top-K sampling
- **Consistency Checking**: Validate solutions across multiple reasoning paths
- **Smart Termination**: Hybrid pattern + LLM-based detection of complete reasoning

### Advanced Features

- **Solution Detection**: Automatic detection and finalization of complete solutions
- **Meta-Reasoning**: LLM-guided action selection based on reasoning state
- **Reflection & Critique**: Self-improvement through iterative refinement
- **Learning System**: Automatic pattern extraction from successful reasoning paths
- **Context Management**: Intelligent summarization to prevent context overflow
- **Benchmarking Suite**: Quantitative evaluation on standard datasets

### User Experience

- **Interactive Shell (mcts-shell)**: Claude Code-style terminal interface with stateful sessions
- **Non-Interactive CLI (mcts)**: Command-line interface for scripting and automation
- **Fluent API**: Chainable method calls for intuitive configuration
- **JSON Serialization**: Save/load search trees for analysis and resumption
- **Rich Diagnostics**: Tree visualization, statistics, and analysis tools

### LLM Support

- **Unified Provider System**: Seamless support for OpenAI, Anthropic, Ollama, and mock LLMs
- **Auto-detection**: Automatically detect provider from environment variables
- **Flexible Configuration**: Per-provider settings for model, temperature, and endpoints

---

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/mcts-reasoning.git
cd mcts-reasoning

# Install with TUI support (recommended)
pip install -e ".[tui]"

# Or install everything
pip install -e ".[all]"
```

See [Installation Guide](getting-started/installation.md) for detailed instructions.

### Interactive Shell (Recommended)

The easiest way to get started:

```bash
# Run the interactive shell
mcts-shell
```

Then use commands like:

```
> ask What is the sum of all prime numbers less than 100?
> search 50
> solution
> sample 5
> verify
> export markdown report.md
```

**Note**: Slash prefix is optional - both `ask` and `/ask` work.

### Non-Interactive CLI

For scripting and automation:

```bash
# Single command execution
mcts ask "What is 2+2?" --search 50

# Work with saved sessions
mcts verify --session my_session.json
mcts export json output.json --session my_session.json
```

See the [TUI Guide](guides/tui-guide.md) for complete command reference.

### Programmatic Usage

```python
from mcts_reasoning import ReasoningMCTS, get_llm

# Auto-detect LLM from environment
llm = get_llm()

# Create MCTS with fluent API
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("What is the optimal algorithm for sorting 1 million integers?")
    .with_exploration(1.414)
    .with_compositional_actions(enabled=True)
    .with_max_rollout_depth(5)
)

# Run search
mcts.search("Let's think about this step by step...", simulations=50)

# Get best solution
print(f"Solution: {mcts.solution}")
print(f"Confidence: {mcts.best_value:.2%}")
```

See [Quick Start Guide](getting-started/quick-start.md) for more examples.

---

## Documentation Structure

### Getting Started
- [Installation](getting-started/installation.md) - Setup and dependencies
- [Quick Start](getting-started/quick-start.md) - Your first MCTS reasoning session
- [Examples](getting-started/examples.md) - Complete code examples

### User Guides
- [Interactive Shell (mcts-shell)](guides/tui-guide.md) - Terminal user interface guide
- [Non-Interactive CLI (mcts)](guides/cli-guide.md) - Command-line interface for automation
- [Remote Ollama Setup](guides/remote-ollama-setup.md) - Using remote Ollama servers

### Features
- [Compositional Actions](features/compositional-actions.md) - 5D action space for reasoning
- [Prompt Features](features/prompt-features.md) - Advanced prompt engineering
- [Solution Detection](features/solution-detection.md) - Automatic solution finalization
- [Meta-Reasoning](features/meta-reasoning.md) - LLM-guided action selection
- [Reflection & Critique](features/reflection.md) - Self-improvement loops
- [Learning System](features/learning.md) - Pattern extraction from experience
- [Context Management](features/context-management.md) - Intelligent summarization
- [Tree Diagnostics](features/tree-diagnostics.md) - Tree visualization and analysis

### Advanced Topics
- [MCP Integration](advanced/mcp-integration.md) - External tool integration
- [Benchmarking](advanced/benchmarking.md) - Quantitative evaluation
- [Automatic Summarization](advanced/automatic-summarization.md) - Context compression

### Development
- [Testing](development/testing.md) - Test suite and results
- [Recent Changes](development/changelog.md) - Version history
- [Bug Fixes](development/bug-fixes.md) - Recent bug fixes
- [New Features](development/new-features.md) - Recently added features

---

## Architecture Overview

```
mcts_reasoning/
├── core.py                    # Pure MCTS implementation
├── reasoning.py               # LLM-specific reasoning layer
├── sampling.py                # Sampling strategies
├── compositional/             # Compositional prompting system
│   ├── __init__.py           # Core enums + ComposingPrompt
│   ├── providers.py          # Unified LLM providers
│   ├── actions.py            # Compositional actions for MCTS
│   ├── mcp.py                # MCP client and tool integration
│   └── mcp_actions.py        # MCP-aware actions
└── tui/                       # Terminal User Interface
    ├── app.py                # Main TUI application
    ├── session.py            # Session state management
    └── commands.py           # Command handlers
```

---

## Use Cases

MCTS-Reasoning excels at:

- **Complex Problem Solving**: Multi-step mathematical, logical, or algorithmic problems
- **Exploratory Reasoning**: When multiple solution approaches might exist
- **Verification & Consistency**: Validating solutions across multiple reasoning paths
- **Tool-Augmented Reasoning**: Problems requiring code execution, web search, or file operations
- **Research & Analysis**: Exploring solution spaces systematically

---

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## License

MIT License - see LICENSE file for details.

---

## Citation

If you use this in your research, please cite:

```bibtex
@software{mcts_reasoning,
  title = {MCTS-Reasoning: Monte Carlo Tree Search for LLM Reasoning with Compositional Prompting},
  year = {2024},
  url = {https://github.com/yourusername/mcts-reasoning}
}
```

---

## Acknowledgments

- Inspired by AlphaGo's MCTS implementation
- Compositional prompting based on reasoning-llm-policy research
- TUI design inspired by Claude Code
- MCP integration following Model Context Protocol specification
