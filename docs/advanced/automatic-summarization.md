# Automatic Context Summarization

## Overview

MCTS reasoning can generate long context chains as the tree deepens. Without management, context can grow to exceed LLM limits or become inefficient. The automatic context summarization feature transparently compresses context when it gets too large, using the same LLM to create concise summaries.

## How It Works

### Automatic Detection

During tree expansion, after each action is executed:
1. The new state is checked against a token threshold (default: 80% of max context)
2. If threshold is exceeded, summarization is automatically triggered
3. The LLM generates a concise summary preserving key information
4. The summarized state replaces the long state
5. Reasoning continues normally with the compressed context

### Token Counting

Two modes are supported:

**Token-based counting (accurate):**
- Uses `tiktoken` library for precise token counting
- Automatically loads appropriate tokenizer for the model
- Supports GPT and Claude models

**Character-based estimation (fallback):**
- Estimates tokens using character count / chars_per_token
- Default: 4 characters per token
- Used when tiktoken unavailable

## Configuration

### Fluent API

```python
from mcts_reasoning import ReasoningMCTS, get_llm
from mcts_reasoning.context_manager import ContextConfig

llm = get_llm("openai", model="gpt-4")

# Approach 1: Auto-configure based on model
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Your question here")
    .with_compositional_actions(enabled=True)
    .with_context_config(auto_configure=True)  # Detects GPT-4 limits
)

# Approach 2: Custom configuration
config = ContextConfig(
    max_context_tokens=16000,      # Maximum context size
    summarize_threshold=0.8,       # Trigger at 80% (12800 tokens)
    summarize_target=0.5,          # Compress to 50% (8000 tokens)
    use_token_counting=True,       # Use accurate token counting
    chars_per_token=4.0            # Fallback estimation ratio
)

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Your question here")
    .with_compositional_actions(enabled=True)
    .with_context_config(config=config, auto_configure=False)
)

# Run search - summarization happens automatically!
mcts.search("Initial reasoning state", simulations=100)
```

### Model-Specific Defaults

When `auto_configure=True`, the system automatically detects model limits:

| Model | Max Context | Threshold | Target |
|-------|-------------|-----------|--------|
| GPT-4 Turbo | 128,000 tokens | 80% | 50% |
| GPT-4 | 8,192 tokens | 80% | 50% |
| GPT-3.5 Turbo | 4,096 tokens | 80% | 50% |
| GPT-3.5 16K | 16,384 tokens | 80% | 50% |
| Claude 3 (Opus/Sonnet) | 200,000 tokens | 80% | 50% |
| Claude 2 | 100,000 tokens | 80% | 50% |
| Claude 1 | 9,000 tokens | 80% | 50% |
| Llama/Mistral/Gemma | 8,000 tokens | 80% | 50% |
| Default | 8,000 tokens | 80% | 50% |

## Parameters Explained

### max_context_tokens
Maximum context size in tokens before compression becomes critical.
- **Too low:** Frequent summarization, may lose important details
- **Too high:** Risk exceeding LLM limits, higher costs
- **Recommendation:** Use model's actual context limit or slightly below

### summarize_threshold
Fraction of max_context_tokens that triggers summarization (0.0-1.0).
- **Default:** 0.8 (80%)
- **Lower values:** More aggressive compression, less detail retained
- **Higher values:** Less frequent compression, more detail retained
- **Recommendation:** 0.7-0.8 for most use cases

### summarize_target
Target size after summarization as fraction of max_context_tokens.
- **Default:** 0.5 (50%)
- **Lower values:** More aggressive compression
- **Higher values:** Less compression, more detail retained
- **Recommendation:** 0.4-0.6 depending on problem complexity

### use_token_counting
Whether to use accurate token counting (requires `tiktoken`).
- **True:** Precise token counting (recommended)
- **False:** Character-based estimation
- **Note:** Automatically falls back to estimation if tiktoken unavailable

### chars_per_token
Average characters per token for estimation (when token counting unavailable).
- **Default:** 4.0
- **GPT models:** ~4 chars/token
- **Claude models:** ~3.8 chars/token
- **Code-heavy content:** ~3 chars/token
- **Natural language:** ~4-5 chars/token

## Monitoring Summarization

### Get Statistics

```python
# After running search
stats = mcts.context_manager.get_stats()

print(f"Summarizations performed: {stats['summarization_count']}")
print(f"Last summary size: {stats['last_summarization_tokens']} tokens")
print(f"Max context: {stats['max_context_tokens']} tokens")
print(f"Using token counting: {stats['use_token_counting']}")
```

### Check Node States

Use diagnostic commands to inspect summarized states:

```python
# In Python
nodes = mcts.get_all_nodes()
for i, node in enumerate(nodes):
    if "[Context summarized" in node.state:
        print(f"Node {i} was summarized")
        print(f"  State length: {len(node.state)} chars")
```

Or in TUI:
```bash
nodes                  # See all nodes
inspect-full 5         # See full state (may show [Context summarized] marker)
show-prompt 10         # See what LLM saw (includes parent's potentially summarized state)
```

### Logging

Summarization events are logged:
```
INFO:mcts_reasoning.context_manager:Context size: 6420 tokens (threshold: 6400)
INFO:mcts_reasoning.context_manager:Triggering automatic summarization
INFO:mcts_reasoning.context_manager:Summarization #3: 6420 → ~4000 tokens
INFO:mcts_reasoning.context_manager:Summary generated: 3856 tokens
```

## Summarization Prompt

The LLM receives this prompt when summarizing:

```
You are helping with step-by-step reasoning. The reasoning context has grown too large and needs to be compressed.

Original Question:
{original_question}

Current Reasoning State (to be summarized):
{state}

Task: Create a concise summary that preserves:
1. The original question/problem
2. Key insights and findings so far
3. Important intermediate results
4. Current progress and next steps

Target length: Approximately {target_tokens} tokens (about {chars} characters).

Provide ONLY the summary, without meta-commentary:
```

**Key features:**
- Preserves original question for context
- Focuses on key insights and findings
- Maintains progress tracking
- Specifies target length for LLM
- Uses low temperature (0.3) for focused summarization

## Best Practices

### 1. Enable for Long Reasoning Sessions

For problems requiring deep exploration (>50 simulations, depth >10):
```python
mcts.with_context_config(auto_configure=True)
```

### 2. Adjust Thresholds Based on Problem Type

**Math/Logic (detailed steps needed):**
```python
config = ContextConfig(
    summarize_threshold=0.85,  # Less aggressive
    summarize_target=0.6       # Retain more detail
)
```

**Creative/Exploratory (can afford compression):**
```python
config = ContextConfig(
    summarize_threshold=0.75,  # More aggressive
    summarize_target=0.4       # More compression
)
```

### 3. Balance with Compositional Actions

Remember that some operations (SYNTHESIZE, REFINE, DECOMPOSE) already compress context:
- If using many SYNTHESIZE actions, can use higher thresholds
- If using mostly ANALYZE/EVALUATE, may need lower thresholds

### 4. Monitor Summarization Frequency

Check stats after search:
```python
stats = mcts.context_manager.get_stats()
summaries_per_node = stats['summarization_count'] / len(mcts.get_all_nodes())

if summaries_per_node > 2:
    print("⚠️ Too frequent! Consider higher threshold or target")
elif summaries_per_node < 0.1:
    print("⚠️ Too rare! Consider lower threshold")
else:
    print("✅ Good balance")
```

### 5. Install tiktoken for Accuracy

```bash
pip install tiktoken
```

Without tiktoken, character-based estimation may be less accurate, especially for:
- Code-heavy content
- Mathematical notation
- Special characters and Unicode

## Trade-offs

### Compression vs Detail

**More compression (lower thresholds/targets):**
- ✅ Faster generation (less context to process)
- ✅ Lower API costs
- ✅ Fits in smaller context windows
- ❌ May lose important details
- ❌ More frequent LLM calls for summarization

**Less compression (higher thresholds/targets):**
- ✅ Retains more reasoning history
- ✅ Better continuity between steps
- ✅ Fewer summarization calls
- ❌ Slower generation (more context)
- ❌ Higher API costs
- ❌ Risk exceeding context limits

### Automatic vs Manual

**Automatic (recommended):**
- ✅ Transparent - no code changes needed
- ✅ Prevents context overflow
- ✅ Consistent behavior
- ❌ May summarize when not desired
- ❌ Uses extra LLM calls

**Manual (via SYNTHESIZE actions):**
- ✅ Full control over when/how to compress
- ✅ Can preserve specific details
- ✅ Part of reasoning strategy
- ❌ Requires explicit action selection
- ❌ May forget to compress

**Recommendation:** Use both! Automatic prevents overflow, SYNTHESIZE provides strategic compression.

## Examples

### Example 1: Long Mathematical Proof

```python
from mcts_reasoning import ReasoningMCTS, get_llm
from mcts_reasoning.context_manager import ContextConfig

llm = get_llm("openai", model="gpt-4")

# Math proofs need detail, so use higher thresholds
config = ContextConfig(
    max_context_tokens=8000,
    summarize_threshold=0.85,
    summarize_target=0.6
)

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Prove that sqrt(2) is irrational")
    .with_compositional_actions(enabled=True)
    .with_context_config(config=config)
    .with_exploration(1.414)
)

mcts.search("Let's prove this by contradiction...", simulations=100)
print(f"Summarizations: {mcts.context_manager.get_stats()['summarization_count']}")
```

### Example 2: Creative Problem Solving

```python
# Creative exploration can afford more compression
config = ContextConfig(
    max_context_tokens=4000,
    summarize_threshold=0.75,
    summarize_target=0.4
)

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Design a unique puzzle game mechanic")
    .with_compositional_actions(enabled=True)
    .with_context_config(config=config)
)

mcts.search("Let's brainstorm creative mechanics...", simulations=80)
```

### Example 3: Remote Ollama with Limited Context

```python
from mcts_reasoning.compositional.providers import OllamaProvider

# Many Ollama models have 4k-8k context
llm = OllamaProvider(
    model="llama3.2",
    base_url="http://192.168.0.225:11434"
)

# Use conservative limits for local models
config = ContextConfig(
    max_context_tokens=4000,
    summarize_threshold=0.8,
    summarize_target=0.5
)

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("What are the prime numbers less than 50?")
    .with_compositional_actions(enabled=True)
    .with_context_config(config=config)
)

mcts.search("Let's find all primes...", simulations=50)
```

## Integration with TUI

Context management works automatically in the TUI:

```bash
# Configure model (auto-detects limits if possible)
model ollama llama3.2 base_url=http://192.168.0.225:11434

# Ask question and search
ask What are all the prime numbers less than 100?
search 100

# Check if summarization occurred
nodes
inspect-full 10  # May show [Context summarized] marker

# Export tree to analyze summarization patterns
export-tree primes_analysis.json
```

## Troubleshooting

### Issue: Too Many Summarizations

**Symptoms:**
- Most nodes show `[Context summarized]`
- Stats show `summarization_count` > total nodes

**Solutions:**
1. Increase `summarize_threshold` (0.8 → 0.9)
2. Increase `summarize_target` (0.5 → 0.7)
3. Increase `max_context_tokens` if model supports it

### Issue: Context Overflow

**Symptoms:**
- LLM errors about context length
- Generation fails mid-search

**Solutions:**
1. Decrease `summarize_threshold` (0.8 → 0.7)
2. Decrease `max_context_tokens` to match model limits
3. Ensure `with_context_config()` is called

### Issue: Lost Important Details

**Symptoms:**
- LLM seems to forget earlier reasoning
- Solutions lack context from earlier steps

**Solutions:**
1. Increase `summarize_target` (0.5 → 0.6 or 0.7)
2. Increase `summarize_threshold` to reduce frequency
3. Use more ANALYZE operations (append context) vs SYNTHESIZE (replace context)

### Issue: Slow Performance

**Symptoms:**
- Search takes much longer than expected
- Many summarization calls in logs

**Solutions:**
1. Increase `summarize_threshold` to reduce frequency
2. Check if `summarize_target` is too low (causing repeated summarizations)
3. Consider disabling for short searches (< 20 simulations)

## Implementation Details

### Code Location

- **ContextManager:** `mcts_reasoning/context_manager.py`
- **Integration:** `mcts_reasoning/reasoning.py` (ReasoningMCTS._take_action)
- **Configuration:** `mcts_reasoning/context_manager.py` (configure_context_for_model)

### Integration Point

Summarization happens in `ReasoningMCTS._take_action()`:

```python
def _take_action(self, state: str, action: Any) -> str:
    # Execute action (compositional or simple)
    new_state = ...

    # Automatic context management
    if self.context_manager and self.context_manager.should_summarize(new_state):
        new_state = self.context_manager.summarize_state(
            new_state,
            self.llm,
            self.original_question
        )

    return new_state
```

### Marker Format

Summarized states include a header:
```
[Context summarized - compression #N]

{summary}
```

Where N is the sequential summarization count.

## See Also

- [Context Management](../features/context-management.md) - Understanding context flow
- [Tree Diagnostics](../features/tree-diagnostics.md) - Inspecting nodes and states
- [Compositional Actions]() - Action types and state building
- [TUI Guide](../guides/tui-guide.md) - Using diagnostic commands
