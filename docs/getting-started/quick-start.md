# Quick Start Guide

This guide will get you up and running with MCTS-Reasoning in minutes.

## Choose Your Interface

MCTS-Reasoning offers three ways to interact with the system:

1. **Interactive Shell** - Unix-style composable commands (recommended for power users)
2. **Interactive TUI** - Claude Code-style terminal interface (recommended for beginners)
3. **Programmatic API** - Python API for integration into your code

## Interactive Shell Quick Start

### Launch the Shell

```bash
mcts-shell
```

You'll see a prompt:

```
MCTS Reasoning Shell v0.2.0
Type 'help' for commands, 'exit' to quit
mcts>
```

### Basic Workflow

```bash
# 1. Ask a question
mcts> ask "What is the sum of all prime numbers less than 100?"

# 2. Run MCTS search
mcts> search 50

# 3. Get the best solution
mcts> solution

# 4. Sample diverse paths
mcts> sample 5

# 5. Check consistency
mcts> consistency 20
```

### Using Pipes

The shell supports Unix-style piping:

```bash
# Complete pipeline
mcts> ask "Solve x^2 - 5x + 6 = 0" | search 100 | sample 5 | best

# Save results
mcts> ask "Find primes < 100" | search 50 | solution > result.txt

# Complex pipeline
mcts> ask "problem" | search 100 | sample 10 | filter --min-value 0.8 | best
```

See [Shell Guide](../guides/shell-guide.md) for complete documentation.

## Interactive TUI Quick Start

### Launch the TUI

```bash
mcts-tui
```

You'll see a rich terminal interface with:
- Syntax highlighting
- Tab completion
- Command history
- Status information

### Basic Commands

```
> /ask What is the sum of all prime numbers less than 100?
> /search 50
> /solution
> /tree
> /sample 5
> /consistency 20
```

### TUI Features

- **Tab completion**: Type `/mo<Tab>` to complete `/model`
- **History search**: Press Ctrl+R to search command history
- **History navigation**: Use ↑/↓ arrows to navigate history
- **Emacs-style editing**: Ctrl+A (start), Ctrl+E (end), Ctrl+K (kill to end)

### Example Session

```
> /model openai gpt-4
Model set to: openai/gpt-4

> /ask What are the prime factors of 84?

> /search 50
Running 50 simulations...
Best value: 0.87

> /solution
Solution:
84 = 2² × 3 × 7

The prime factors of 84 are: 2, 3, and 7
- 84 ÷ 2 = 42
- 42 ÷ 2 = 21
- 21 ÷ 3 = 7
- 7 is prime

> /tree
Tree visualization:
[Shows ASCII tree structure]

> /save session.json
Session saved to session.json
```

See [TUI Guide](../guides/tui-guide.md) for complete documentation.

## Programmatic Quick Start

### Basic Usage

```python
from mcts_reasoning import ReasoningMCTS, get_llm

# Auto-detect LLM from environment
llm = get_llm()

# Create MCTS instance
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("What is 37 * 43?")
    .with_exploration(1.414)
    .with_max_rollout_depth(5)
)

# Run search
mcts.search("Let's calculate:", simulations=50)

# Get results
print(f"Solution: {mcts.solution}")
print(f"Confidence: {mcts.best_value:.2%}")
```

### With Compositional Actions

```python
from mcts_reasoning import ReasoningMCTS, get_llm

llm = get_llm("openai", model="gpt-4")

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Solve the equation x^2 - 5x + 6 = 0")
    .with_compositional_actions(enabled=True)
    .with_exploration(1.414)
)

mcts.search("Let's solve this equation:", simulations=50)

# Get diverse solutions
paths = mcts.sample(n=5, strategy="diverse")
for i, path in enumerate(paths, 1):
    print(f"\nPath {i}:")
    print(path.final_state)
```

### With Advanced Features

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
)

mcts.search("Let's think about this systematically:", simulations=100)

# Check consistency
result = mcts.check_consistency(n_samples=20)
print(f"Most consistent solution (confidence={result['confidence']:.1%}):")
print(result['solution'])

# Save tree for later analysis
mcts.save("reasoning_tree.json")
```

## Example Problems

### Math Problem

```python
from mcts_reasoning import ReasoningMCTS, get_llm

llm = get_llm()

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("What is the sum of all prime numbers less than 100?")
    .with_compositional_actions(enabled=True)
)

mcts.search("Let's solve this step by step:", simulations=50)
print(mcts.solution)
```

### Logic Puzzle

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

### Algorithm Problem

```python
question = "What is the time complexity of quicksort? Explain best, average, and worst cases."

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question(question)
    .with_compositional_actions(enabled=True)
    .with_solution_detection(enabled=True)
)

mcts.search("Let me analyze this:", simulations=50)
print(mcts.solution)
```

## Next Steps

Now that you've got the basics, explore more:

- [Examples](examples.md) - Complete code examples
- [Compositional Actions](../features/compositional-actions.md) - Rich action space
- [Solution Detection](../features/solution-detection.md) - Automatic solution finalization
- [Meta-Reasoning](../features/meta-reasoning.md) - LLM-guided exploration
- [MCP Integration](../advanced/mcp-integration.md) - External tool integration
- [Benchmarking](../advanced/benchmarking.md) - Evaluate performance

## Common Patterns

### Exploration vs Exploitation

```python
# More exploration (try diverse approaches)
mcts.with_exploration(2.0)

# More exploitation (focus on best paths)
mcts.with_exploration(0.5)

# Balanced (default)
mcts.with_exploration(1.414)
```

### Sampling Strategies

```python
# Best performing paths
paths = mcts.sample(n=5, strategy="value")

# Most explored paths
paths = mcts.sample(n=5, strategy="visits")

# Diverse paths (different approaches)
paths = mcts.sample(n=5, strategy="diverse", temperature=1.5)

# Top-K paths
paths = mcts.sample(n=5, strategy="topk")
```

### Consistency Checking

```python
# Check if solutions agree
result = mcts.check_consistency(n_samples=20)

if result['confidence'] > 0.8:
    print("High confidence solution:")
    print(result['solution'])
else:
    print("Low confidence, solutions disagree")
    print(f"Agreement rate: {result['confidence']:.1%}")
```

## Tips for Success

1. **Start simple**: Use basic MCTS first, then enable advanced features
2. **Tune exploration**: Adjust the exploration constant based on problem type
3. **Use enough simulations**: 50-100 is usually sufficient for most problems
4. **Enable compositional actions**: They significantly improve reasoning quality
5. **Check consistency**: For critical problems, validate with multiple samples
6. **Save your trees**: Use `mcts.save()` to save trees for later analysis
7. **Monitor performance**: Use tree diagnostics to understand search behavior

## Getting Help

- [Shell Quick Reference](../guides/shell-quick-reference.md) - Command cheat sheet
- [TUI Guide](../guides/tui-guide.md) - Complete TUI documentation
- [Features Documentation](../features/compositional-actions.md) - Feature guides
- [GitHub Issues](https://github.com/yourusername/mcts-reasoning/issues) - Report bugs or ask questions
