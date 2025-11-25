# Examples

This guide showcases complete examples of using MCTS-Reasoning for various tasks.

All example code is available in the `examples/` directory of the repository.

## Basic Usage

### Simple Math Problem

**File:** `examples/basic_usage.py`

```python
from mcts_reasoning import ReasoningMCTS, get_llm

# Auto-detect LLM from environment
llm = get_llm()

# Create MCTS with fluent API
question = "What is 37 * 43?"

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question(question)
    .with_exploration(1.414)
    .with_max_rollout_depth(3)
    .with_metadata(domain="arithmetic", difficulty="easy")
)

# Run search
initial_state = f"Question: {question}\n\nLet's solve this step by step:"
mcts.search(initial_state, simulations=20)

# Get solution
solution, confidence = mcts.solution_with_confidence
print(f"Solution (confidence={confidence:.2%}):")
print(solution)

# Save tree
mcts.save("example_tree.json")
```

**Key Concepts:**
- Fluent API with method chaining
- Simple question-answering
- Tree persistence with JSON serialization

## Compositional Prompting

### Using Advanced Action Space

**File:** `examples/compositional_demo.py`

```python
from mcts_reasoning import (
    ReasoningMCTS,
    get_llm,
    CognitiveOperation,
    FocusAspect,
    ReasoningStyle,
    ComposingPrompt
)

# Build a structured reasoning prompt
prompt = (
    ComposingPrompt()
    .cognitive_op(CognitiveOperation.DECOMPOSE)
    .focus(FocusAspect.STRUCTURE)
    .style(ReasoningStyle.SYSTEMATIC)
    .problem_context("Find all prime numbers less than 20")
    .build()
)

print(prompt)
# Output: "Problem: Find all prime numbers less than 20
# Let me break this problem down systematically. I'll focus on the structural
# relationships and organization. I'll approach this systematically and methodically."
```

### MCTS with Compositional Actions

```python
llm = get_llm()

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Find all prime numbers less than 20")
    .with_exploration(1.414)
    .with_compositional_actions(enabled=True)  # Enable compositional!
    .with_max_rollout_depth(4)
)

mcts.search("Let's find all prime numbers less than 20.", simulations=30)

# Sample diverse solutions
paths = mcts.sample(n=3, strategy="diverse", temperature=1.5)

for i, path in enumerate(paths, 1):
    print(f"\nPath {i} (length={path.length}, value={path.total_value:.2f}):")
    print(f"  Final state: {path.final_state}")
```

**Key Concepts:**
- 5-dimensional compositional action space
- Diverse solution sampling
- Rich reasoning strategies

### Weighted Action Sampling

```python
# Define weights to bias toward certain operations
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
vector = prompt.get_action_vector()
print(f"Vector: ω={vector['omega']}, φ={vector['phi']}, σ={vector['sigma']}")
```

**Key Concepts:**
- Biased exploration toward specific reasoning styles
- Fine-grained control over action selection

## Sampling Strategies

### Value-Based Sampling

**File:** `examples/sampling_demo.py`

```python
from mcts_reasoning import ReasoningMCTS, get_llm

llm = get_llm()

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Solve x^2 - 5x + 6 = 0")
    .with_compositional_actions(enabled=True)
)

mcts.search("Let's solve this equation:", simulations=50)

# Sample best performing paths
best_paths = mcts.sample(n=5, strategy="value")

# Sample most explored paths
explored_paths = mcts.sample(n=5, strategy="visits")

# Sample diverse paths
diverse_paths = mcts.sample(n=5, strategy="diverse", temperature=1.5)

# Top-K paths
topk_paths = mcts.sample(n=5, strategy="topk")
```

**Key Concepts:**
- Multiple sampling strategies (value, visits, diverse, topk)
- Temperature control for diversity
- Exploring vs exploiting tradeoffs

## Consistency Checking

### Validating Solutions

**File:** `examples/consistency_check.py`

```python
from mcts_reasoning import ReasoningMCTS, get_llm

llm = get_llm()

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("What are the prime factors of 84?")
    .with_compositional_actions(enabled=True)
)

mcts.search("Let's find the prime factors:", simulations=50)

# Check consistency across 20 samples
result = mcts.check_consistency(n_samples=20, temperature=1.0)

print(f"Most consistent solution:")
print(f"  Confidence: {result['confidence']:.1%}")
print(f"  Support: {result['support']}/{result['total_samples']} samples")
print(f"  Number of clusters: {len(result['clusters'])}")
print(f"\nSolution: {result['solution']}")

# High confidence check
if result['confidence'] > 0.8:
    print("High confidence - solution is consistent!")
else:
    print("Low confidence - solutions disagree")
    for i, cluster in enumerate(result['clusters']):
        print(f"\nCluster {i+1} (support={cluster['support']}):")
        print(cluster['solution'])
```

**Key Concepts:**
- Solution validation across multiple reasoning paths
- Clustering similar solutions
- Confidence scoring

## MCP Tool Integration

### Using External Tools

**File:** `examples/mcp_demo.py`

```python
from mcts_reasoning import (
    ReasoningMCTS,
    get_llm,
    create_mcp_client,
    create_mcp_provider
)

# Create base LLM
base_llm = get_llm("openai", model="gpt-4")

# Enable MCP tools
mcp_client = create_mcp_client({
    "python": {"type": "python"},
    "web": {"type": "web"},
    "file": {"type": "file"}
})

# Wrap LLM with MCP awareness
mcp_llm = create_mcp_provider(base_llm, mcp_client=mcp_client)

# Now the LLM can use tools!
mcts = (
    ReasoningMCTS()
    .with_llm(mcp_llm)
    .with_question("Calculate the 50th Fibonacci number")
    .with_compositional_actions(enabled=True)
)

mcts.search("Let me solve this with code:", simulations=50)
print(mcts.solution)
```

**Key Concepts:**
- External tool integration via MCP
- Python execution, web search, file operations
- Tool-augmented reasoning

See [MCP Integration Guide](../advanced/mcp-integration.md) for complete documentation.

## Logic Puzzles

### Box Labeling Problem

```python
from mcts_reasoning import ReasoningMCTS, get_llm

question = """
Three boxes contain fruits. Box A is labeled "Apples",
Box B is labeled "Oranges", Box C is labeled "Mixed".
All labels are wrong. You can pick one fruit from one box.
How do you determine the correct labels?
"""

llm = get_llm()

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question(question)
    .with_exploration(1.5)  # More exploration for logic puzzles
    .with_compositional_actions(enabled=True)
)

mcts.search("Let's reason systematically:", simulations=30)

# Get diverse solutions
paths = mcts.sample(n=5, strategy="diverse")
for i, path in enumerate(paths, 1):
    print(f"\nSolution {i}:")
    print(path.final_state)
```

**Key Concepts:**
- Higher exploration constant for complex problems
- Diverse solution sampling
- Systematic reasoning approach

## Algorithm Analysis

### Analyzing Time Complexity

```python
from mcts_reasoning import ReasoningMCTS, get_llm

question = """
What is the time complexity of quicksort?
Explain best, average, and worst cases.
"""

llm = get_llm()

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question(question)
    .with_compositional_actions(enabled=True)
    .with_solution_detection(enabled=True)  # Auto-detect complete solutions
)

mcts.search("Let me analyze quicksort:", simulations=50)
print(mcts.solution)
```

**Key Concepts:**
- Solution detection for structured answers
- Algorithmic reasoning
- Multi-faceted analysis

## Advanced Features

### Complete Feature Showcase

```python
from mcts_reasoning import ReasoningMCTS, get_llm

llm = get_llm("openai", model="gpt-4")

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("What is the optimal algorithm for sorting 1 million integers?")
    .with_compositional_actions(enabled=True)
    .with_solution_detection(enabled=True)      # Auto-detect solutions
    .with_meta_reasoning(enabled=True)          # LLM-guided action selection
    .with_reflection(enabled=True)              # Self-critique and refinement
    .with_learning(enabled=True)                # Learn from successful paths
    .with_exploration(1.414)
    .with_max_rollout_depth(5)
)

mcts.search("Let's think about this systematically:", simulations=100)

# Check consistency
result = mcts.check_consistency(n_samples=20)
print(f"Most consistent solution (confidence={result['confidence']:.1%}):")
print(result['solution'])

# Save tree for later analysis
mcts.save("reasoning_tree.json")
```

**Key Concepts:**
- All advanced features enabled
- Solution detection and finalization
- Meta-reasoning for adaptive exploration
- Reflection for self-improvement
- Learning from experience
- High simulation count for complex problems

## Remote Ollama

### Using Remote Ollama Server

**File:** `examples/remote_ollama.py`

```python
from mcts_reasoning import ReasoningMCTS, get_llm

# Connect to remote Ollama server
llm = get_llm(
    "ollama",
    model="llama2",
    base_url="http://192.168.0.225:11434"
)

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Explain recursion in simple terms")
    .with_compositional_actions(enabled=True)
)

mcts.search("Let me explain:", simulations=30)
print(mcts.solution)
```

**Key Concepts:**
- Remote LLM server configuration
- Custom base URLs
- Network-based reasoning

See [Remote Ollama Setup](../guides/remote-ollama-setup.md) for detailed configuration.

## Running the Examples

All examples are in the `examples/` directory:

```bash
# Basic usage
python examples/basic_usage.py

# Compositional prompting
python examples/compositional_demo.py

# MCP integration
python examples/mcp_demo.py

# Sampling strategies
python examples/sampling_demo.py

# Consistency checking
python examples/consistency_check.py

# Remote Ollama
python examples/remote_ollama.py
```

## Tips for Your Own Examples

1. **Start simple**: Begin with basic MCTS, then add features incrementally
2. **Tune exploration**: Adjust based on problem complexity (0.5-2.0 range)
3. **Use enough simulations**: 50-100 for most problems, 100+ for complex ones
4. **Enable compositional actions**: Significantly improves reasoning quality
5. **Check consistency**: For critical problems, validate with multiple samples
6. **Save trees**: Use `mcts.save()` for later analysis
7. **Monitor performance**: Use tree statistics to understand behavior

## Next Steps

- [Compositional Actions](../features/compositional-actions.md) - Deep dive into action space
- [Solution Detection](../features/solution-detection.md) - Auto-detection of solutions
- [Meta-Reasoning](../features/meta-reasoning.md) - LLM-guided exploration
- [MCP Integration](../advanced/mcp-integration.md) - External tool integration
- [Benchmarking](../advanced/benchmarking.md) - Evaluate performance
