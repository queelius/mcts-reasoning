# Solution Detection and Finalization

## Overview

MCTS reasoning now includes **automatic solution detection and finalization** - a powerful system that:

1. **Detects complete solutions** using LLM-as-a-judge
2. **Creates polished final answers** from reasoning context
3. **Marks nodes as terminal** to stop expansion
4. **Allows other branches to continue** exploring alternative solutions

This implements your suggestion: "when we reach a summary node, we stop, but we also include the original problem and ask it to rewrite it in a nice way as the final answer. this becomes a leaf in the tree, marked as a solution."

## Key Features

### 1. LLM-as-a-Judge Solution Detection

More sophisticated than pattern matching - actually understands whether the reasoning provides a satisfactory answer.

**Evaluation Criteria:**
- Does it directly answer the question?
- Is the answer clear and unambiguous?
- Is the reasoning logically sound?
- Are all necessary steps present?
- Would a human consider this complete?

**Output:** `SolutionJudgment` with:
- `is_solution`: Boolean verdict
- `confidence`: 0-1 confidence score
- `reasoning`: Explanation of judgment
- `needs_refinement`: Whether answer could be improved

### 2. Solution Finalization

When a solution is detected, the system:

1. Takes current state (potentially summarized)
2. Includes the original problem
3. Asks LLM to create a polished final answer
4. Adds `[SOLUTION FINALIZED]` marker
5. Node becomes terminal (won't expand further)

**Finalized answer includes:**
- Clear, well-formatted final answer
- Key reasoning insights (bullet points)
- Professional presentation

### 3. Branch-Level Termination

**Critical design:** Only the specific branch with the finalized solution stops expanding. Other branches in the tree continue exploring alternative solutions!

This allows MCTS to:
- Find multiple different solutions
- Explore alternative approaches
- Compare solution quality across branches

## Usage

### Basic Usage

```python
from mcts_reasoning import ReasoningMCTS, get_llm

llm = get_llm("openai", model="gpt-4")

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("What are the prime numbers less than 20?")
    .with_compositional_actions(enabled=True)
    .with_solution_detection(enabled=True)  # Enable solution finalization
    .search("Let's solve this problem...", simulations=100)
)

# Finalization happens automatically!
# Branches with solutions stop, others continue
```

### Configuration Options

```python
mcts.with_solution_detection(
    enabled=True,           # Enable/disable solution detection
    threshold=0.7,          # Confidence threshold (0-1) for considering something a solution
    auto_finalize=True      # Automatically finalize detected solutions
)
```

**Parameters:**

- **threshold** (0-1): Higher values are more conservative
  - `0.5`: Aggressive - more solutions detected
  - `0.7`: Balanced - moderate detection (default)
  - `0.9`: Conservative - only very confident solutions

- **auto_finalize**: Whether to automatically create polished answers
  - `True`: Automatic finalization (recommended)
  - `False`: Detect but don't finalize

### With Context Summarization

Combine with automatic summarization for long reasoning sessions:

```python
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Solve x^3 - 3x^2 + x - 10 = 0")
    .with_compositional_actions(enabled=True)
    .with_context_config(auto_configure=True)  # Auto-summarize long context
    .with_solution_detection(enabled=True)      # Auto-finalize solutions
    .search("Let's solve this equation...", simulations=100)
)
```

**Interaction:**
- Context may be summarized during reasoning
- When solution detected, finalized answer includes original problem
- Finalization uses the current (potentially summarized) state
- Summary + original problem → polished final answer

## How It Works

### Workflow

```
1. Action executed → New state created
                    ↓
2. Context checked → Summarize if needed?
                    ↓
3. Solution checked → Is this a complete solution?
                    ↓
                   YES
                    ↓
4. Finalization → Create polished final answer
                    ↓
5. Node marked → [SOLUTION FINALIZED] marker added
                    ↓
6. Terminal check → Node becomes terminal (is_leaf = True)
                    ↓
7. Expansion stops → This branch won't expand further
                    ↓
8. Other branches → Continue exploring!
```

### Code Flow

**In `ReasoningMCTS._take_action()`:**

```python
def _take_action(self, state: str, action: Any) -> str:
    # 1. Execute action
    new_state = action.execute(...)

    # 2. Check if summarization needed
    if should_summarize(new_state):
        new_state = summarize_state(new_state)

    # 3. Check if solution should be finalized
    if should_attempt_finalization(new_state):
        # LLM-as-a-judge detects solution
        new_state = finalize_solution(new_state)  # Create polished answer
        # Adds [SOLUTION FINALIZED] marker

    return new_state
```

**In `ReasoningMCTS._is_terminal_state()`:**

```python
def _is_terminal_state(self, state: str) -> bool:
    # Check for finalized solution first (always terminal)
    if is_finalized_solution(state):
        return True

    # Otherwise use standard termination detection
    return smart_termination(state)
```

## Examples

### Example 1: Math Problem

```python
from mcts_reasoning import ReasoningMCTS, get_llm

llm = get_llm("anthropic", model="claude-3-5-sonnet-20241022")

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Find the sum of all prime numbers less than 30")
    .with_compositional_actions(enabled=True)
    .with_solution_detection(enabled=True, threshold=0.75)
    .search("Let's identify all primes...", simulations=50)
)

# Get finalized solutions
nodes = mcts.get_all_nodes()
for i, node in enumerate(nodes):
    if "[SOLUTION FINALIZED]" in node.state:
        print(f"Solution found at node {i}:")
        print(node.state)
        print(f"Confidence: {node.value / node.visits if node.visits > 0 else 0:.2f}")
        print()
```

**Output:**
```
Solution found at node 12:
[SOLUTION FINALIZED]

## Final Answer

The sum of all prime numbers less than 30 is **129**.

The prime numbers are: 2, 3, 5, 7, 11, 13, 17, 19, 23, 29

## Key Reasoning

- Identified all numbers from 2-29
- Applied primality test to each
- Summed all prime numbers found

Confidence: 0.85
```

### Example 2: Multiple Solutions

```python
# Run longer search to find multiple solution approaches
mcts.search("Initial state", simulations=200)

# Find all finalized solutions
from mcts_reasoning.solution_detection import is_finalized_solution

solutions = [
    (i, node) for i, node in enumerate(mcts.get_all_nodes())
    if is_finalized_solution(node.state)
]

print(f"Found {len(solutions)} finalized solutions")
for idx, node in solutions:
    print(f"\nSolution {idx} (visits={node.visits}, value={node.value:.3f}):")
    # Each solution may use different approach
```

**Result:**
- Multiple branches found solutions
- Each used different reasoning approach
- All properly finalized and marked terminal
- Other branches continued exploring until simulations exhausted

### Example 3: With Remote Ollama

```python
from mcts_reasoning.compositional.providers import OllamaProvider

llm = OllamaProvider(
    model="deepseek-r1:7b",
    base_url="http://192.168.0.225:11434"
)

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("What are the Fibonacci numbers up to 100?")
    .with_compositional_actions(enabled=True)
    .with_context_config(auto_configure=True)    # Auto-summarize for 4k context
    .with_solution_detection(enabled=True)        # Auto-finalize solutions
    .search("Let's generate Fibonacci...", simulations=60)
)
```

## Monitoring

### Check Detection Stats

```python
# After search
if mcts.solution_detector:
    print(f"Solution judgments made: {mcts.solution_detector._judgment_count}")

if mcts.solution_finalizer:
    stats = mcts.solution_finalizer.get_stats()
    print(f"Solutions finalized: {stats['finalization_count']}")
```

### Find Finalized Nodes

```python
from mcts_reasoning.solution_detection import is_finalized_solution

nodes = mcts.get_all_nodes()
finalized = [node for node in nodes if is_finalized_solution(node.state)]

print(f"Finalized solutions: {len(finalized)}/{len(nodes)} nodes")
```

### Inspect Finalized State

```python
# In TUI
nodes                       # List all nodes
inspect-full 15             # See full state (will show [SOLUTION FINALIZED])
show-prompt 15              # See what LLM saw when creating this node
```

## Integration with TUI

Solution finalization works automatically in the TUI:

```bash
# Configure model
model openai gpt-4o

# Ask question (solution detection enabled by default)
ask What are the first 10 Fibonacci numbers?

# Run search
search 50

# Check for finalized solutions
nodes
# Look for nodes with [SOLUTION FINALIZED] marker

# Inspect finalized node
inspect-full 12

# See final answer
solution
```

## Best Practices

### 1. Set Appropriate Threshold

**For formal problems (math, logic):**
```python
.with_solution_detection(enabled=True, threshold=0.8)  # Higher threshold
```
- Ensures solution is rigorous
- Reduces false positives

**For creative/exploratory problems:**
```python
.with_solution_detection(enabled=True, threshold=0.6)  # Lower threshold
```
- More permissive
- Captures partial solutions

### 2. Combine with Adequate Simulations

```python
# Too few simulations may not reach solutions
mcts.search(..., simulations=20)  # May not be enough

# More simulations = more chances to find solutions
mcts.search(..., simulations=100)  # Better for complex problems
```

### 3. Use with RAG for Domain-Specific Problems

```python
from mcts_reasoning.compositional.rag import get_math_compositional_rag

rag_store = get_math_compositional_rag()

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Solve quadratic equation x^2 - 5x + 6 = 0")
    .with_compositional_actions(enabled=True)
    .with_rag_store(rag_store)           # Guide reasoning for math
    .with_solution_detection(enabled=True)  # Finalize when complete
)
```

### 4. Monitor Finalization Frequency

After search, check:
```python
nodes = mcts.get_all_nodes()
finalized_count = sum(1 for n in nodes if is_finalized_solution(n.state))
finalization_rate = finalized_count / len(nodes)

if finalization_rate > 0.8:
    print("⚠️ Too aggressive - increase threshold")
elif finalization_rate < 0.05:
    print("⚠️ Too conservative - decrease threshold or increase simulations")
else:
    print("✅ Good balance")
```

## Troubleshooting

### Issue: No Solutions Detected

**Symptoms:**
- No `[SOLUTION FINALIZED]` markers
- Judgment count is high but finalization count is zero

**Solutions:**
1. Lower threshold: `threshold=0.5` instead of `0.7`
2. More simulations to reach complete solutions
3. Check if LLM responses actually contain answers
4. Verify `auto_finalize=True`

### Issue: Too Many False Positives

**Symptoms:**
- Most nodes marked as finalized
- Solutions aren't actually complete

**Solutions:**
1. Raise threshold: `threshold=0.85`
2. Improve LLM responses to be more exploratory early
3. Check solution detection prompt in `solution_detection.py`

### Issue: Finalized Nodes Still Expanding

**Symptoms:**
- Nodes have `[SOLUTION FINALIZED]` but have children

**Solutions:**
1. Verify `_is_terminal_state()` checks `is_finalized_solution()`
2. Check that marker is exactly `[SOLUTION FINALIZED]`
3. Ensure terminal check happens before expansion

## Advanced: Custom Solution Detection

You can create custom solution detection logic:

```python
from mcts_reasoning.solution_detection import SolutionDetector

class CustomDetector(SolutionDetector):
    def judge_solution(self, state, original_question, temperature=0.1):
        # Your custom logic here
        # Must return SolutionJudgment

        # Example: domain-specific patterns
        if "QED" in state and "proof complete" in state.lower():
            return SolutionJudgment(
                is_solution=True,
                confidence=0.95,
                reasoning="Formal proof completed with QED"
            )

        # Otherwise use default LLM-based judgment
        return super().judge_solution(state, original_question, temperature)

# Use custom detector
custom_detector = CustomDetector(llm, threshold=0.7)
mcts.solution_detector = custom_detector
```

## Summary

### Answering Your Questions

**Q1: "Do we have RAG retrieval being used?"**

✅ **YES** - RAG is implemented and actively used:
- `CompositionalRAGStore` maps problem patterns to compositional dimensions
- `ActionSelector.get_valid_actions()` uses RAG weights when problem provided
- Predefined stores for math, coding, and logic domains
- Location: `mcts_reasoning/compositional/rag.py`

**Q2: "Do we have LLM-as-a-judge for determining whether an output satisfies the idea of including the answer?"**

✅ **YES** - Now implemented:
- `SolutionDetector` uses LLM to judge if state contains complete solution
- More sophisticated than pattern matching
- Evaluates 5 criteria for completeness
- Returns structured `SolutionJudgment` with confidence
- Location: `mcts_reasoning/solution_detection.py`

**Q3: "Ask the LLM to rewrite it in a nice way as the final answer. This becomes a leaf in the tree, marked as a solution."**

✅ **YES** - Implemented exactly as requested:
- `SolutionFinalizer` creates polished final answers
- Includes original problem + current context
- Produces well-formatted response with key reasoning
- Adds `[SOLUTION FINALIZED]` marker
- Node becomes terminal (leaf) - won't expand further
- **Other branches continue exploring** (key feature!)

### Test Results

```bash
$ python test_solution_finalization.py

✅ Solution finalization worked!
   14 solution(s) detected and finalized
   All finalized nodes are properly marked as terminal
```

All features working as designed!

## See Also

- [Automatic Summarization](../advanced/automatic-summarization.md) - Context management
- [Context Management](context-management.md) - Understanding context flow
- [Tree Diagnostics](tree-diagnostics.md) - Inspecting trees
- [RAG Integration](/examples/rag_demo.py) - Using RAG stores
