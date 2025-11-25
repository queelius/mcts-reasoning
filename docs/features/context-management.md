# Context Management in MCTS Reasoning

## Overview

Understanding what context the LLM sees at each step is critical for debugging and improving reasoning quality. This document explains exactly how context flows through the MCTS tree.

## State Management

### What is "State"?

In MCTS reasoning, each node has a `state` - a string containing the accumulated reasoning up to that point. **However, not all operations accumulate context!**

### State Building: Cumulative vs Replacement

When an action is executed, `_build_new_state()` determines how to combine the parent state with the LLM's response:

```python
def _build_new_state(self, old_state: str, response: str) -> str:
    if self.operation == CognitiveOperation.DECOMPOSE:
        # REPLACES old state!
        return f"Problem decomposition:\n{response}"

    elif self.operation == CognitiveOperation.SYNTHESIZE:
        # REPLACES old state!
        return f"Synthesis of reasoning:\n{response}"

    elif self.operation == CognitiveOperation.REFINE:
        # REPLACES old state!
        return f"Refined approach:\n{response}"

    elif self.operation in [CognitiveOperation.ANALYZE, CognitiveOperation.EVALUATE]:
        # APPENDS to old state
        return f"{old_state}\n\nAnalysis:\n{response}"

    else:
        # APPENDS to old state
        return f"{old_state}\n\n{self.operation.value.title()}:\n{response}"
```

**Key Operations:**

| Operation | Behavior | Reason |
|-----------|----------|--------|
| DECOMPOSE | **Replaces** | Breaking down problem into parts (fresh start) |
| SYNTHESIZE | **Replaces** | Combining ideas into summary (compression) |
| REFINE | **Replaces** | Improving/simplifying previous work (distillation) |
| ANALYZE | **Appends** | Adding analysis to existing reasoning |
| EVALUATE | **Appends** | Adding evaluation to existing reasoning |
| Others | **Appends** | Default behavior |

### Why Replacement?

Replacement operations prevent context explosion:
- SYNTHESIZE compresses long reasoning into key points
- REFINE distills reasoning into clearer form
- DECOMPOSE breaks complex problem into simpler subproblems

This is **by design** but means you can lose earlier context!

---

## What the LLM Sees

### Prompt Construction

When creating a new node, the LLM sees:

1. **Original Question** (always included)
2. **Parent State** (last 4000 chars)
3. **Compositional Action Instructions** (operation, focus, style, etc.)

Here's the actual flow:

```python
# In CompositionalAction.to_prompt():

prompt_builder = (
    ComposingPrompt()
    .cognitive_op(self.operation)
    .focus(self.focus)
    .style(self.style)
    .connect(self.connection)
    .format(self.output_format)
    .problem_context(original_question)  # ← Always included
)

# Add parent state (last 4000 chars)
max_context = 4000  # Configurable
prompt_builder.base_prompt(f"Current state:\n{current_state[-max_context:]}")

return prompt_builder.build()
```

### Context Window

- **Maximum context: 4000 characters** (configurable via `max_context_length`)
- Takes the **last 4000 chars** of parent state
- If parent state is shorter, uses all of it

**Example:**
```
Parent state: 6000 chars
LLM sees: Last 4000 chars + question + action instructions
Lost: First 2000 chars
```

---

## Diagnostic Commands

### 1. `inspect <index>` - Quick Preview

Shows **last 500 chars** of state. Good for quick checks.

```bash
inspect 5
# Shows: action, metrics, last 500 chars of state
```

### 2. `inspect-full <index>` - Complete State

Shows **entire state** without truncation. Use this to see what was ACCUMULATED.

```bash
inspect-full 5
# Shows: complete state of node 5 (all chars)
```

**What you see:**
- The complete accumulated reasoning at this node
- Result of all `_build_new_state()` operations up to this point
- May include or exclude earlier context depending on operations used

### 3. `show-prompt <index>` - What LLM Saw

Shows the **exact prompt** the LLM received when creating this node.

```bash
show-prompt 5
# Shows: reconstructed prompt with parent state + action instructions
```

**What you see:**
- Problem context (original question)
- Parent state (last 4000 chars)
- Compositional action instructions
- **This is what the LLM actually saw!**

---

## Example Trace

Let's trace through a 3-node path:

### Node 0 (ROOT)
```
State (102 chars):
"Question: What is x^3 - 3x^2 + x - 10 = 0?

Let me think about this systematically."
```

### Node 1: Action(ANALYZE, details, systematic)

**What LLM saw:**
```
Problem: What is x^3 - 3x^2 + x - 10 = 0?

Current state:
Question: What is x^3 - 3x^2 + x - 10 = 0?

Let me think about this systematically.

[Compositional action: Analyze the details systematically...]
```

**LLM Response:**
```
"We can use the Rational Root Theorem to find possible rational roots..."
```

**New State (APPENDED because ANALYZE):**
```
Question: What is x^3 - 3x^2 + x - 10 = 0?

Let me think about this systematically.

Analysis:
We can use the Rational Root Theorem to find possible rational roots...
```

### Node 2: Action(SYNTHESIZE, solution, formal)

**What LLM saw:**
```
Problem: What is x^3 - 3x^2 + x - 10 = 0?

Current state:
Question: What is x^3 - 3x^2 + x - 10 = 0?

Let me think about this systematically.

Analysis:
We can use the Rational Root Theorem to find possible rational roots...

[Compositional action: Synthesize the solution formally...]
```

**LLM Response:**
```
"The solution is x ≈ 3.512 using numerical methods."
```

**New State (REPLACED because SYNTHESIZE):**
```
Synthesis of reasoning:
The solution is x ≈ 3.512 using numerical methods.
```

**Notice:** All previous reasoning is LOST! This is intentional compression.

---

## How to Debug Context Issues

### Problem: "LLM seems to forget earlier reasoning"

**Diagnosis:**
```bash
nodes                  # Find the node
inspect-full 5         # See complete state
show-prompt 5          # See what LLM actually saw
```

**Possible causes:**
1. **Replacement operation used** (DECOMPOSE/SYNTHESIZE/REFINE)
   - Check action type in `/inspect`
   - These intentionally reset context

2. **Context truncation** (state > 4000 chars)
   - Check state length in `/inspect-full`
   - LLM only sees last 4000 chars
   - Earlier reasoning was truncated

3. **Parent state too short**
   - Parent might have used replacement operation
   - Check parent with `inspect-full <parent_index>`

### Problem: "Tree is not branching properly"

**Diagnosis:**
```bash
nodes                  # Check tree structure
inspect 0              # Check root's tried_actions
```

**Expected:**
- Root should have multiple children
- Nodes should show `tried=X` where X increases

**If not branching:**
- Check if actions are being filtered too aggressively
- Verify action selector is returning diverse actions
- Check if tree hit terminal states early

---

## Best Practices

### 1. Monitor State Growth

Use `/inspect-full` periodically to check state size:
```bash
inspect-full 10
# Check: Length: XXX characters
```

If states are growing too large (> 4000 chars), consider:
- Using more SYNTHESIZE operations to compress
- Decreasing `max_context_length` to force compression
- Using REFINE to distill key points

### 2. Balance Accumulation vs Compression

**Too much accumulation:**
- States grow huge
- Context window exceeded
- LLM loses focus on recent steps

**Too much compression:**
- Lost context from earlier reasoning
- LLM doesn't see full problem decomposition
- May repeat work or forget constraints

**Good balance:**
- Accumulate with ANALYZE/EVALUATE during exploration
- Compress with SYNTHESIZE when combining results
- Refine with REFINE when quality is low

### 3. Use Diagnostic Commands

**Regular workflow:**
```bash
ask <question>
search 50
nodes                  # Overview
inspect-full 10        # Check accumulation
show-prompt 15         # Verify LLM saw enough context
path 15                # Trace reasoning flow
```

---

## Configuration

### Adjusting Context Window

You can modify the context window size:

```python
from mcts_reasoning.compositional.actions import CompositionalAction

# Create action
action = CompositionalAction(...)

# Set custom context length
action.max_context_length = 8000  # Use 8000 chars instead of 4000

# Or set globally by modifying defaults in actions.py:
# max_context = getattr(self, 'max_context_length', 8000)  # Change default
```

**Considerations:**
- Larger context = more information for LLM
- Larger context = slower generation
- Larger context = higher API costs
- Most LLMs handle 4000-8000 chars well
- Check your LLM's context window limit

---

## Common Patterns

### Pattern 1: Deep Analytical Path

```
ROOT
 └─ ANALYZE (append)
     └─ ANALYZE (append)
         └─ ANALYZE (append)
             └─ SYNTHESIZE (compress all analysis)
```

**Result:** Deep reasoning compressed into summary

### Pattern 2: Iterative Refinement

```
ROOT
 └─ DECOMPOSE (replace - break into parts)
     └─ ANALYZE (append - work on part)
         └─ REFINE (replace - improve solution)
```

**Result:** Each REFINE replaces previous attempts

### Pattern 3: Parallel Exploration

```
ROOT
 ├─ DECOMPOSE (part 1)
 ├─ DECOMPOSE (part 2)
 └─ DECOMPOSE (part 3)
      └─ SYNTHESIZE (combine parts)
```

**Result:** Each branch explores different decomposition

---

## See Also

- [Tree Diagnostics](tree-diagnostics.md) - Using diagnostic commands
- [TUI Guide](../guides/tui-guide.md) - Complete command reference
- [Bug Fixes](../development/bug-fixes.md) - Recent context improvements
