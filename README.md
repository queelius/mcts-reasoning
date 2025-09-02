# MCTS-Reasoning

Monte Carlo Tree Search for LLM-based reasoning with compositional actions.

## Features

- **Pure MCTS Implementation**: Clean, canonical MCTS with proper Selection, Expansion, Rollout, and Backpropagation phases
- **Compositional Action Space**: Rich action space combining cognitive operations, focus aspects, and reasoning styles
- **Multiple LLM Support**: Works with OpenAI, Anthropic, Ollama, or mock LLMs
- **Live Visualization**: Real-time tree visualization with IPC support
- **Flexible Architecture**: Easy to extend and customize for different reasoning tasks

## Installation

```bash
# Basic installation
pip install mcts-reasoning

# With specific LLM support
pip install mcts-reasoning[ollama]
pip install mcts-reasoning[openai]
pip install mcts-reasoning[anthropic]

# With visualization support
pip install mcts-reasoning[viewer]

# Everything
pip install mcts-reasoning[all]
```

## Quick Start

```python
from mcts_reasoning import ReasoningMCTS, get_llm

# Auto-detect LLM from environment
llm = get_llm()

# Create MCTS instance
mcts = ReasoningMCTS(
    llm_client=llm,
    original_question="What is the optimal algorithm for sorting 1 million integers?",
    exploration_constant=1.414,
    use_compositional=True
)

# Run search
initial_state = "Let's think about this step by step..."
root = mcts.search(initial_state, num_simulations=50)

# Get best path
best_path = mcts.get_best_path()
for action, state in best_path:
    print(f"Action: {action}")
    print(f"State: {state[:100]}...")
```

## Compositional Actions

The system uses a compositional action space with 5 dimensions:

- **Operations**: analyze, decompose, solve, verify, synthesize
- **Focus**: problem, solution, assumptions, constraints, approach
- **Style**: systematic, intuitive, formal
- **Connection**: therefore, however, furthermore, alternatively, specifically
- **Format**: statement, question, list

This creates a rich space of 1,200 possible actions that can be efficiently explored.

## LLM Adapters

### Using Different LLMs

```python
from mcts_reasoning import OllamaAdapter, OpenAIAdapter, AnthropicAdapter, MockLLMAdapter

# Ollama (local)
llm = OllamaAdapter(host="localhost", port=11434, model="llama2")

# OpenAI
llm = OpenAIAdapter(api_key="your-key", model="gpt-4")

# Anthropic
llm = AnthropicAdapter(api_key="your-key", model="claude-3-sonnet-20240229")

# Mock (for testing)
llm = MockLLMAdapter()
```

### Auto-detection from Environment

```python
# Set environment variable
export LLM_PROVIDER=openai
export OPENAI_API_KEY=your-key

# Or for Anthropic
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your-key

# In Python
from mcts_reasoning import get_llm
llm = get_llm()  # Auto-detects from environment
```

## Live Visualization

### Start the Viewer

```bash
# Terminal 1: Start the viewer server
cd viewer
python server.py

# Terminal 2: Run MCTS with IPC
python your_script.py
```

### Enable IPC in Your Code

```python
from mcts_reasoning import ReasoningMCTSWithIPC

mcts = ReasoningMCTSWithIPC(
    llm_client=llm,
    original_question="Your question here",
    enable_ipc=True,
    ipc_host="localhost",
    ipc_port=9999
)

# Run search - events will be sent to viewer
root = mcts.search(initial_state, num_simulations=50)
```

Then open http://localhost:8000 in your browser to see the tree grow in real-time.

## Examples

### Math Problem

```python
from mcts_reasoning import ReasoningMCTS, MockLLMAdapter

llm = MockLLMAdapter({
    "calculate": "37 * 43 = 1591",
    "verify": "Correct",
    "terminal": "YES"
})

mcts = ReasoningMCTS(
    llm_client=llm,
    original_question="What is 37 * 43?",
    max_rollout_depth=3
)

root = mcts.search("Let's calculate:", num_simulations=20)
print(f"Answer found with confidence: {root.value:.2f}")
```

### Logic Puzzle

```python
question = """
Three boxes contain fruits. Box A is labeled "Apples", 
Box B is labeled "Oranges", Box C is labeled "Mixed". 
All labels are wrong. You can pick one fruit from one box. 
How do you determine the correct labels?
"""

mcts = ReasoningMCTS(
    llm_client=llm,
    original_question=question,
    exploration_constant=1.5,  # More exploration
    use_compositional=True
)

root = mcts.search("Let's reason systematically:", num_simulations=30)
```

## Architecture

```
mcts_reasoning/
├── mcts_core.py           # Core MCTS implementation
├── reasoning_mcts.py       # LLM-specific reasoning layer
├── mcts_with_ipc.py       # IPC support for visualization
├── llm_adapters.py        # LLM provider adapters
├── actions/
│   └── compositional_actions.py  # Compositional action space
└── viewer/
    └── server.py          # Live visualization server
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

The system uses LLM-based terminal detection instead of pattern matching:

```python
def is_terminal(state, depth, question):
    # Ask LLM if this state represents a complete solution
    response = llm.generate(f"Is this a complete solution to: {question}?")
    return "YES" in response
```

## Configuration

### MCTS Parameters

- `exploration_constant`: UCB1 exploration parameter (default: 1.414)
- `max_rollout_depth`: Maximum depth for rollouts (default: 5)
- `use_compositional`: Enable compositional actions (default: True)
- `discount_factor`: Reward discount for deeper nodes (default: 0.95)

### IPC Configuration

- `enable_ipc`: Enable IPC for visualization (default: False)
- `ipc_host`: IPC server host (default: "localhost")
- `ipc_port`: IPC server port (default: 9999)

## Testing

```bash
# Run tests
pytest tests/

# With coverage
pytest --cov=mcts_reasoning tests/

# Test specific LLM adapter
pytest tests/test_adapters.py::test_ollama
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Citation

If you use this in your research, please cite:

```bibtex
@software{mcts_reasoning,
  title = {MCTS-Reasoning: Monte Carlo Tree Search for LLM Reasoning},
  year = {2024},
  url = {https://github.com/yourusername/mcts-reasoning}
}
```

## Acknowledgments

- Inspired by AlphaGo's MCTS implementation
- Compositional prompting based on reasoning-llm-policy research
- Tree visualization uses vis.js