# MCTS-Reasoning

Monte Carlo Tree Search for LLM-based reasoning with advanced compositional prompting and MCP integration.

Combines systematic tree search with sophisticated prompt engineering to enable structured, exploratory reasoning across multiple solution paths.

## ✨ Features

- **🎯 Pure MCTS Implementation**: Clean, canonical MCTS with proper Selection, Expansion, Rollout, and Backpropagation phases
- **🧩 Advanced Compositional Prompting**: Rich action space with 5 dimensions (ω, φ, σ, κ, τ) enabling 30,000+ action combinations
  - Cognitive Operations (ω): decompose, analyze, synthesize, verify, etc.
  - Focus Aspects (φ): structure, patterns, assumptions, correctness, etc.
  - Reasoning Styles (σ): systematic, intuitive, critical, creative, etc.
  - Connection Types (κ): therefore, however, building_on, etc.
  - Output Formats (τ): steps, list, mathematical, code, etc.
- **🔌 MCP Tool Integration**: Automatic access to external tools via Model Context Protocol
  - Python code execution
  - Web search
  - File operations
  - Custom tools
- **🖥️ Interactive TUI**: Claude Code-style terminal interface with stateful sessions
- **🤖 Unified LLM Provider System**: Seamless support for OpenAI, Anthropic, Ollama, and mock LLMs
- **🎨 Fluent API**: Chainable method calls for intuitive configuration
- **📊 Multiple Sampling Strategies**: Value-based, visit-based, diverse, and top-K sampling
- **✅ Consistency Checking**: Validate solutions across multiple reasoning paths
- **🎓 Smart Termination**: Hybrid pattern + LLM-based detection of complete reasoning
- **💾 JSON Serialization**: Save/load search trees for analysis and resumption
- **🔧 Flexible Architecture**: Easy to extend and customize for different reasoning tasks

## Installation

```bash
# Install from source
git clone https://github.com/yourusername/mcts-reasoning.git
cd mcts-reasoning
pip install -e .

# With TUI (recommended)
pip install -e ".[tui]"

# With specific LLM support
pip install -e ".[openai]"
pip install -e ".[anthropic]"
pip install -e ".[ollama]"

# Everything (TUI + all LLM providers)
pip install -e ".[all]"
```

## Quick Start

### Interactive TUI (Recommended)

The easiest way to get started is with the TUI:

```bash
# Run the TUI
python mcts_tui.py

# Or if installed:
mcts-tui
```

Then use commands like:
```
> /model                                    # Show current model
> /model ollama llama2 base_url=http://192.168.0.225:11434  # Remote Ollama
> /models                                   # List available models
> /ask What is the sum of all prime numbers less than 100?
> /search 50
> /solution
> /sample 5
> /consistency 20
```

**Enhanced Prompt Features:**
- **Tab completion** for commands and arguments (try `/mo<Tab>`)
- **Persistent history** across sessions (use ↑/↓ arrows)
- **History search** with Ctrl+R
- **Syntax highlighting** for commands
- **Emacs-style editing** (Ctrl+A, Ctrl+E, etc.)

See the [TUI Guide](docs/TUI_GUIDE.md) and [Prompt Features](docs/PROMPT_FEATURES.md) for complete documentation.

### Basic Usage (Programmatic)

```python
from mcts_reasoning import ReasoningMCTS, get_llm

# Auto-detect LLM from environment (or use specific provider)
llm = get_llm()  # Auto-detect
# llm = get_llm("openai", model="gpt-4")
# llm = get_llm("anthropic", model="claude-3-5-sonnet-20241022")
# llm = get_llm("ollama", model="llama2")

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
initial_state = "Let's think about this step by step..."
mcts.search(initial_state, simulations=50)

# Get best solution
print(f"Solution: {mcts.solution}")
print(f"Confidence: {mcts.best_value:.2%}")

# Save tree
mcts.save("reasoning_tree.json")
```

### Compositional Prompting

```python
from mcts_reasoning import ComposingPrompt, CognitiveOperation, FocusAspect, ReasoningStyle

# Build a structured reasoning prompt
prompt = (
    ComposingPrompt()
    .cognitive_op(CognitiveOperation.DECOMPOSE)
    .focus(FocusAspect.STRUCTURE)
    .style(ReasoningStyle.SYSTEMATIC)
    .problem_context("Find all prime numbers less than 100")
    .build()
)

print(prompt)
# Output: "Problem: Find all prime numbers less than 100
# Let me break this problem down systematically. I'll focus on the structural
# relationships and organization. I'll approach this systematically and methodically."
```

### MCP Tool Integration

```python
from mcts_reasoning import create_mcp_client, create_mcp_provider

# Enable MCP tools
mcp_client = create_mcp_client({
    "python": {"type": "python"},
    "web": {"type": "web"}
})

# Wrap LLM with MCP awareness
mcp_llm = create_mcp_provider(base_llm, mcp_client=mcp_client)

# Now the LLM can automatically use tools!
mcts = ReasoningMCTS().with_llm(mcp_llm).with_question("Calculate fibonacci(50)")
```

See [MCP Integration Guide](docs/MCP_INTEGRATION.md) for details.

### Advanced Sampling

```python
# Sample diverse reasoning paths
paths = mcts.sample(n=5, strategy="diverse", temperature=1.5)

for i, path in enumerate(paths, 1):
    print(f"Path {i}: {path.final_state}")

# Check consistency across multiple samples
result = mcts.check_consistency(n_samples=20)
print(f"Most consistent solution (confidence={result['confidence']:.1%}):")
print(result['solution'])
```

## Documentation

- **[TUI Guide](docs/TUI_GUIDE.md)** - Complete TUI documentation with examples
- **[MCP Integration](docs/MCP_INTEGRATION.md)** - Using external tools with MCP
- **[API Reference](docs/API.md)** - Full API documentation (coming soon)
- **[Examples](examples/)** - Code examples and demos

## Examples

See the `examples/` directory for complete examples:

- `basic_usage.py` - Basic MCTS with simple actions
- `compositional_demo.py` - Full compositional prompting demonstration
- `mcp_demo.py` - MCP tool integration examples
- `sampling_demo.py` - Various sampling strategies
- `consistency_check.py` - Solution consistency validation

### Math Problem

```python
from mcts_reasoning import ReasoningMCTS, MockLLMProvider

llm = MockLLMProvider({
    "calculate": "37 * 43 = 1591",
    "verify": "Correct",
    "terminal": "YES"
})

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("What is 37 * 43?")
    .with_max_rollout_depth(3)
)

mcts.search("Let's calculate:", simulations=20)
print(f"Answer: {mcts.solution}")
print(f"Confidence: {mcts.best_value:.2f}")
```

### Logic Puzzle with Compositional Actions

```python
question = """
Three boxes contain fruits. Box A is labeled "Apples",
Box B is labeled "Oranges", Box C is labeled "Mixed".
All labels are wrong. You can pick one fruit from one box.
How do you determine the correct labels?
"""

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question(question)
    .with_exploration(1.5)  # More exploration
    .with_compositional_actions(enabled=True)
)

mcts.search("Let's reason systematically:", simulations=30)

# Get diverse solutions
paths = mcts.sample(n=5, strategy="diverse")
for i, path in enumerate(paths, 1):
    print(f"\nSolution {i}:")
    print(path.final_state)
```

## Compositional Actions

The system uses a compositional action space with 5 dimensions:

- **Operations**: analyze, decompose, solve, verify, synthesize
- **Focus**: problem, solution, assumptions, constraints, approach
- **Style**: systematic, intuitive, formal
- **Connection**: therefore, however, furthermore, alternatively, specifically
- **Format**: statement, question, list

This creates a rich space of 30,000+ possible actions that can be efficiently explored.

## Weighted Action Sampling

You can bias the compositional action space toward certain operations, styles, or focuses:

```python
# Define weights
weights = {
    'cognitive_op': {
        CognitiveOperation.DECOMPOSE: 3.0,  # Strongly prefer decomposition
        CognitiveOperation.VERIFY: 2.0,     # Prefer verification
    },
    'style': {
        ReasoningStyle.SYSTEMATIC: 3.0,     # Strongly prefer systematic
        ReasoningStyle.FORMAL: 1.5,
    }
}

# Sample with weights
prompt = ComposingPrompt.sample_action(weights)
print(prompt.get_action_vector())
```

## LLM Adapters

### Using Different LLMs

```python
from mcts_reasoning import get_llm

# Auto-detection from environment
llm = get_llm()

# Or specify provider
llm = get_llm("openai", model="gpt-4")
llm = get_llm("anthropic", model="claude-3-5-sonnet-20241022")
llm = get_llm("ollama", model="llama2")
llm = get_llm("mock")  # For testing
```

### Auto-detection from Environment

```bash
# Set environment variable
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key

# Or for Anthropic
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-key
```

```python
# In Python
from mcts_reasoning import get_llm
llm = get_llm()  # Auto-detects from environment
```

## Architecture

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

## Key Concepts

### MCTS Phases

1. **Selection**: Navigate tree using UCB1 formula
2. **Expansion**: Add one new child node
3. **Rollout**: Simulate to terminal state
4. **Backpropagation**: Update statistics up the tree

### UCB1 Formula

```python
ucb1 = value/visits + c * sqrt(ln(parent_visits)/visits)
```

Where `c` is the exploration constant (default: 1.414).

### Terminal State Detection

The system uses smart termination combining:
- Pattern matching (keywords like "therefore", "QED", "final answer")
- LLM-based assessment of reasoning completeness

### MCP Tool Integration

LLMs can automatically access external tools:
- Results are incorporated into reasoning context
- ~40% of MCTS actions encourage tool usage by default
- Custom tools can be easily registered

## Configuration

### MCTS Parameters

- `exploration_constant`: UCB1 exploration parameter (default: 1.414)
- `max_rollout_depth`: Maximum depth for rollouts (default: 5)
- `use_compositional`: Enable compositional actions (default: True)
- `discount_factor`: Reward discount for deeper nodes (default: 0.95)

### TUI Commands

See [TUI Guide](docs/TUI_GUIDE.md) for complete command reference.

Common commands:
- `/ask <question>` - Start reasoning
- `/search <n>` - Run simulations
- `/solution` - Show best solution
- `/tree` - Visualize search tree
- `/sample <n>` - Sample diverse paths
- `/consistency` - Check solution consistency

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=mcts_reasoning tests/

# Run examples
python examples/basic_usage.py
python examples/compositional_demo.py
python examples/mcp_demo.py
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this in your research, please cite:

```bibtex
@software{mcts_reasoning,
  title = {MCTS-Reasoning: Monte Carlo Tree Search for LLM Reasoning with Compositional Prompting},
  year = {2024},
  url = {https://github.com/yourusername/mcts-reasoning}
}
```

## Acknowledgments

- Inspired by AlphaGo's MCTS implementation
- Compositional prompting based on reasoning-llm-policy research
- TUI design inspired by Claude Code
- MCP integration following Model Context Protocol specification
