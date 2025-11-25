# Meta-Reasoning: LLM-Guided Action Selection

Meta-reasoning enables the LLM to analyze the current reasoning state and suggest which cognitive operation would be most productive to try next, rather than relying solely on RAG-weighted or random action selection.

## Overview

**Traditional Action Selection**:
- Random sampling from compositional action space
- RAG-weighted sampling (patterns from past problems)
- No awareness of current reasoning progress

**Meta-Reasoning**:
- LLM analyzes current state and problem
- Suggests which cognitive operation would be most productive
- Biases action selection toward suggested operations
- Adaptive strategy based on reasoning progress

## How It Works

1. **Analysis Phase**: LLM examines current reasoning state
2. **Suggestion Phase**: LLM suggests next cognitive operation (decompose, analyze, verify, etc.)
3. **Biasing Phase**: Action selector boosts probability of suggested operation
4. **Exploration Maintained**: Other operations still possible (controlled exploration)

## Usage

### Basic Usage

```python
from mcts_reasoning import ReasoningMCTS, get_llm

llm = get_llm("openai", model="gpt-4")

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Solve x^2 - 5x + 6 = 0")
    .with_compositional_actions(enabled=True)
    .with_meta_reasoning(enabled=True)  # Enable meta-reasoning!
)

mcts.search("Let's solve this equation...", simulations=50)
```

### Configuration Parameters

```python
.with_meta_reasoning(
    enabled=True,          # Enable/disable meta-reasoning
    temperature=0.2,       # Temperature for meta-reasoning LLM calls (lower = more focused)
    bias_strength=3.0      # How much to boost suggested operations (multiplier)
)
```

**Parameters**:
- `enabled`: Whether to use meta-reasoning
- `temperature`: Lower values (0.1-0.3) make suggestions more consistent
- `bias_strength`: Higher values (3-5) strongly bias toward suggestions

### Combined with Other Features

```python
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Find all prime numbers between 30 and 50")
    .with_compositional_actions(enabled=True)
    .with_rag_store(rag_store)            # RAG patterns
    .with_meta_reasoning(enabled=True)     # LLM guidance
    .with_learning(enabled=True)           # Learn from experience
    .with_solution_detection(enabled=True) # Auto-detect solutions
)
```

All features work together:
- RAG provides general patterns
- Meta-reasoning adapts to current state
- Learning improves RAG over time
- Solution detection finalizes complete answers

## Meta-Reasoning Prompt

The meta-reasoner asks the LLM:

```
You are a meta-reasoning system analyzing a reasoning process.

Original Problem:
{question}

Current Reasoning State (recent):
{state}

Available Cognitive Operations:
decompose, analyze, synthesize, verify, abstract, concretize,
compare, evaluate, generate, refine

Task: Analyze the current reasoning state and suggest which cognitive
operation would be MOST PRODUCTIVE to try next.

Consider:
1. What has been tried so far?
2. What key aspects are missing or need more work?
3. What would advance the reasoning most effectively?
4. Is the reasoning stuck in a pattern?

Respond in this EXACT format:
OPERATION: [operation name]
FOCUS: [what aspect to focus on]
STYLE: [systematic/intuitive/critical/creative]
CONFIDENCE: [0.0 to 1.0]
REASONING: [One sentence explaining why]
```

## Example Meta-Reasoning Suggestions

### Example 1: Early in reasoning
```
OPERATION: decompose
FOCUS: problem structure
STYLE: systematic
CONFIDENCE: 0.85
REASONING: The problem needs systematic decomposition into prime checking steps.
```

### Example 2: After initial exploration
```
OPERATION: verify
FOCUS: correctness
STYLE: critical
CONFIDENCE: 0.90
REASONING: Multiple approaches attempted; time to verify which are correct.
```

### Example 3: Near solution
```
OPERATION: synthesize
FOCUS: complete solution
STYLE: systematic
CONFIDENCE: 0.88
REASONING: All components present; need to synthesize into final answer.
```

## Statistics and Monitoring

```python
# Get meta-reasoning statistics
stats = mcts.meta_reasoner.get_stats()

print(f"Suggestions made: {stats['suggestion_count']}")
print(f"Average confidence: {stats['average_confidence']}")
print(f"Most suggested operation: {stats['most_suggested']}")
print(f"Operation distribution: {stats['operation_distribution']}")

# Recent suggestions
for sugg in stats['recent_suggestions']:
    print(f"  {sugg['operation']} (confidence={sugg['confidence']})")
    print(f"    → {sugg['reasoning']}")
```

## Implementation Details

### MetaReasoner Class

```python
from mcts_reasoning.meta_reasoning import MetaReasoner

meta_reasoner = MetaReasoner(llm, temperature=0.2)

suggestion = meta_reasoner.suggest_next_action(
    current_state="Current reasoning...",
    original_question="Problem to solve",
    available_operations=["decompose", "analyze", ...]
)

# Returns ActionSuggestion:
# - operation: str (suggested operation)
# - focus: str (what to focus on)
# - style: str (reasoning style)
# - confidence: float (0-1)
# - reasoning: str (explanation)
```

### MetaReasoningActionSelector

Wraps base `ActionSelector` to bias action selection:

```python
from mcts_reasoning.meta_reasoning import MetaReasoningActionSelector

meta_selector = MetaReasoningActionSelector(
    base_selector=action_selector,
    meta_reasoner=meta_reasoner,
    bias_strength=3.0
)

# Get actions biased toward meta-reasoner's suggestion
actions = meta_selector.get_valid_actions(
    current_state=state,
    problem=question
)

# Suggested operations appear 3x more often (bias_strength=3.0)
```

## Benefits

1. **Adaptive Strategy**: Reasoning adapts to current state rather than following fixed patterns
2. **Stuck Detection**: LLM can identify when reasoning is stuck and suggest different approaches
3. **Guided Exploration**: Balances LLM guidance with exploration of diverse paths
4. **Confidence Tracking**: Low confidence suggestions allow more exploration

## Comparison: RAG vs Meta-Reasoning

### RAG (Retrieval-Augmented Generation)
- **When**: Problem matches known patterns
- **Strength**: Fast, learned from experience
- **Limitation**: Static patterns, may not adapt to novel problems

### Meta-Reasoning
- **When**: Every reasoning step
- **Strength**: Adapts to current state, detects stuck reasoning
- **Limitation**: Extra LLM calls, depends on LLM quality

### Combined (Recommended)
- RAG provides general patterns and biases
- Meta-reasoning adapts to specific state
- Learning improves RAG from successful meta-reasoning paths
- Best of both: pattern matching + adaptive intelligence

## Test Results

```bash
$ python test_meta_reasoning.py

TESTING META-REASONING WITH MOCK LLM
────────────────────────────────────────────────────────────────────────────────
Meta-reasoner enabled: True
Bias strength: 3.0

Running search with meta-reasoning...

Meta-Reasoning Statistics:
  Suggestions made: 47
  Average confidence: 0.500
  Most suggested operation: decompose (47 times)

✅ Meta-reasoning test PASSED - Suggestions being made
```

## Performance Considerations

### LLM Calls
- Meta-reasoning adds ~1 LLM call per action selection
- Use `temperature=0.1-0.3` for faster, more deterministic suggestions
- Consider disabling for very deep searches (100+ simulations)

### Bias Strength
- `bias_strength=2-3`: Gentle guidance, maintains exploration
- `bias_strength=4-5`: Strong guidance, focused search
- `bias_strength=6+`: Very focused, may reduce diversity

### When to Use
- ✅ Complex problems requiring adaptive strategies
- ✅ When reasoning gets stuck
- ✅ When you want LLM to guide search strategy
- ❌ Simple problems with known patterns (RAG alone is faster)
- ❌ Very deep searches where LLM call overhead matters

## Future Enhancements

Potential improvements to meta-reasoning:
1. **Multi-turn dialogue**: Let meta-reasoner ask questions about state
2. **Strategy planning**: Suggest sequences of operations, not just next one
3. **Confidence-based exploration**: Low confidence → more exploration
4. **Meta-learning**: Learn which suggestions worked best
5. **Collaborative filtering**: Combine suggestions from multiple LLMs

## Summary

Meta-reasoning enables **LLM-guided adaptive reasoning** where the system:
- Analyzes current reasoning progress
- Suggests most productive next steps
- Adapts strategy based on state
- Balances guidance with exploration

Combined with RAG and learning, it creates a powerful self-improving reasoning system.
