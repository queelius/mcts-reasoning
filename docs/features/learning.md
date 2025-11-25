# Learning System

The learning system enables MCTS-Reasoning to **automatically extract patterns** from successful reasoning paths and **improve performance over time** by learning from experience.

## Overview

Instead of treating each problem independently, the learning system:

1. **Tracks** successful reasoning paths during search
2. **Analyzes** what operations led to success
3. **Extracts** reusable patterns and strategies
4. **Stores** learned knowledge in a RAG (Retrieval-Augmented Generation) store
5. **Applies** learned patterns to similar future problems

This creates a **continuous improvement loop** where the system becomes better at reasoning over time.

## Key Components

### PathLearner

Analyzes reasoning paths and extracts patterns:

```python
from mcts_reasoning.learning import PathLearner

learner = PathLearner(llm)

# Learn from a successful path
path = mcts.sample(n=1, strategy="value")[0]
patterns = learner.learn_from_path(
    path=path,
    problem="Find all prime numbers less than 30",
    value=0.85
)

print(f"Learned {len(patterns)} patterns")
```

### CompositionalRAGStore

Stores and retrieves learned reasoning patterns:

```python
from mcts_reasoning.compositional.rag import CompositionalRAGStore

rag_store = CompositionalRAGStore()

# Patterns are automatically added during learning
print(f"RAG store contains {len(rag_store)} guidance entries")

# Retrieve relevant guidance for a new problem
guidance = rag_store.get_relevant_guidance("Find primes less than 50")

# Get weighted action biases from learned patterns
weights = rag_store.get_action_weights_for_problem("Find primes less than 50")
```

## Usage

### Basic Learning

```python
from mcts_reasoning import ReasoningMCTS, get_llm
from mcts_reasoning.compositional.rag import CompositionalRAGStore

llm = get_llm()
rag_store = CompositionalRAGStore()

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Find all prime numbers less than 30")
    .with_compositional_actions(enabled=True)
    .with_rag_store(rag_store)
    .with_learning(enabled=True, auto_learn=True)  # Enable automatic learning
)

mcts.search("Let's identify primes...", simulations=50)

# Learning happens automatically after search
print(f"RAG store now has {len(rag_store)} guidance entries")
```

When learning is enabled:
- After search completes, successful paths are analyzed
- Patterns are extracted and stored in RAG
- RAG store grows with experience
- Future searches benefit from learned patterns

### Persistent Learning Across Sessions

```python
from mcts_reasoning import ReasoningMCTS, get_llm
from mcts_reasoning.compositional.rag import CompositionalRAGStore

# Load existing RAG store (or create new one)
rag_store = CompositionalRAGStore.load("learned_patterns.json")

# First problem
mcts1 = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Find primes less than 30")
    .with_compositional_actions(enabled=True)
    .with_rag_store(rag_store)
    .with_learning(enabled=True, auto_learn=True)
)
mcts1.search("Let's solve...", simulations=50)

# Second problem (benefits from first)
mcts2 = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Find primes between 30 and 50")
    .with_compositional_actions(enabled=True)
    .with_rag_store(rag_store)  # Same RAG store!
    .with_learning(enabled=True, auto_learn=True)
)
mcts2.search("Let's solve...", simulations=50)

# Save learned patterns
rag_store.save("learned_patterns.json")
```

### Manual Learning

```python
# Learn explicitly from specific paths
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Problem")
    .with_compositional_actions(enabled=True)
    .with_rag_store(rag_store)
    .with_learning(enabled=True, auto_learn=False)  # Manual control
)

mcts.search("Solve...", simulations=50)

# Manually trigger learning from best paths
best_paths = mcts.sample(n=5, strategy="value")
for path in best_paths:
    mcts.learn_from_path(path)
```

## How It Works

### Pattern Extraction

The learning system analyzes successful paths to extract:

1. **Problem Characteristics**:
   - Problem type (math, logic, algorithm, etc.)
   - Keywords and concepts
   - Complexity indicators

2. **Successful Operations**:
   - Which cognitive operations were used
   - Which focus aspects were emphasized
   - Which reasoning styles were effective

3. **Success Metrics**:
   - Path value (quality score)
   - Whether it led to a solution
   - Reasoning depth and efficiency

4. **Reusable Patterns**:
   - Operation sequences that worked
   - Action weights that led to success
   - Strategies applicable to similar problems

### Pattern Storage

Patterns are stored as **CompositionalGuidance** entries:

```python
class CompositionalGuidance:
    problem_pattern: str           # Pattern for matching problems
    problem_keywords: List[str]    # Keywords for retrieval
    recommended_operations: List   # Operations that worked
    weights: Dict                  # Action weights for success
    success_rate: float            # Historical success rate
    example_path: ReasoningPath    # Example successful path
```

### Pattern Retrieval

When solving a new problem:

1. Extract keywords from problem statement
2. Find similar problems in RAG store
3. Retrieve relevant guidance entries
4. Aggregate action weights from matching patterns
5. Bias action selection toward successful operations

```python
# Automatic retrieval during search
weights = rag_store.get_action_weights_for_problem(
    "Find all prime numbers less than 50"
)

# weights = {
#     'cognitive_op': {
#         CognitiveOperation.DECOMPOSE: 2.3,
#         CognitiveOperation.VERIFY: 1.8,
#         ...
#     },
#     'focus': {
#         FocusAspect.STRUCTURE: 2.1,
#         ...
#     }
# }
```

## Learning Statistics

Track learning progress:

```python
if mcts.path_learner:
    stats = mcts.path_learner.get_stats()

    print(f"Learning Statistics:")
    print(f"  Patterns learned: {stats['patterns_learned']}")
    print(f"  Learning iterations: {stats['learning_count']}")

    if stats['recent_patterns']:
        print(f"  Recent patterns:")
        for pattern in stats['recent_patterns'][:3]:
            print(f"    - Problem: {pattern['problem']}")
            print(f"      Value: {pattern['value']:.3f}")
            print(f"      Operations: {pattern['operations']}")
```

## Benefits

### Improved Performance Over Time

```
First problem:
  Simulations needed: 50
  Best value: 0.72
  Solution found: Yes

Tenth similar problem (with learning):
  Simulations needed: 30
  Best value: 0.89
  Solution found: Yes
  Speedup: 40% fewer simulations
```

### Domain Adaptation

The system adapts to problem domains:

```python
# After solving 10 math problems
# RAG store has strong weights for:
# - VERIFY operation (check correctness)
# - SYSTEMATIC style (methodical approach)
# - CORRECTNESS focus (accuracy matters)

# After solving 10 creative problems
# RAG store has strong weights for:
# - EXPLORE operation (try new ideas)
# - CREATIVE style (unconventional thinking)
# - ALTERNATIVE connections (different approaches)
```

### Transfer Learning

Patterns learned on simple problems transfer to complex ones:

```python
# Learn from: "Find primes less than 20"
# Pattern: decompose → check divisibility → verify

# Transfer to: "Find twin primes less than 100"
# Same pattern applies, just more complex
```

## Example: Cumulative Learning

```python
from mcts_reasoning import ReasoningMCTS, get_llm
from mcts_reasoning.compositional.rag import CompositionalRAGStore

llm = get_llm()
rag_store = CompositionalRAGStore()

problems = [
    "Find all prime numbers less than 20",
    "Find primes between 20 and 40",
    "Find twin primes less than 50",
    "Find the largest prime less than 100",
]

for i, problem in enumerate(problems, 1):
    print(f"\nProblem {i}: {problem}")
    print(f"RAG store size: {len(rag_store)}")

    mcts = (
        ReasoningMCTS()
        .with_llm(llm)
        .with_question(problem)
        .with_compositional_actions(enabled=True)
        .with_rag_store(rag_store)
        .with_learning(enabled=True, auto_learn=True)
    )

    mcts.search("Let's solve...", simulations=30)

    print(f"Solution found: {mcts.solution is not None}")
    print(f"Best value: {mcts.best_value:.3f}")

print(f"\nFinal RAG store: {len(rag_store)} guidance entries")
rag_store.save("prime_patterns.json")
```

## Integration with Other Features

### Learning + Compositional Actions

Learning identifies which compositional actions work best:

```python
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Problem")
    .with_compositional_actions(enabled=True)  # Explore action space
    .with_learning(enabled=True)               # Learn what works
)
```

After learning:
- RAG store contains weights for successful operations
- Future searches bias toward proven strategies
- Exploration becomes more informed

### Learning + Meta-Reasoning

Combine learned patterns with real-time analysis:

```python
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Problem")
    .with_compositional_actions(enabled=True)
    .with_meta_reasoning(enabled=True)         # LLM suggests operations
    .with_learning(enabled=True)               # Learn successful suggestions
)
```

Meta-reasoning suggestions are influenced by learned patterns, creating adaptive intelligence.

See [Meta-Reasoning](meta-reasoning.md) for details.

### Learning + Reflection

Learn from high-quality refined reasoning:

```python
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Problem")
    .with_compositional_actions(enabled=True)
    .with_reflection(enabled=True)             # Refine reasoning quality
    .with_learning(enabled=True)               # Learn from refined reasoning
)
```

The learning system extracts patterns from the higher-quality reflected reasoning, leading to better learned patterns.

See [Reflection](reflection.md) for details.

## Configuration

```python
.with_learning(
    enabled=True,
    auto_learn=True,           # Automatically learn after search
    min_value_threshold=0.7,   # Only learn from high-quality paths
    max_patterns_per_search=10 # Limit patterns extracted per search
)
```

**Options:**

- `enabled`: Enable/disable learning system
- `auto_learn`: Automatically learn after search (vs manual)
- `min_value_threshold`: Minimum path value to learn from (0-1)
- `max_patterns_per_search`: Limit extracted patterns to prevent overfitting

## Testing

```bash
python test_learning.py
```

Tests cover:
- Pattern extraction from paths
- RAG store updates
- Weight accumulation
- Transfer to similar problems
- Persistent storage

## Best Practices

1. **Start with empty RAG store**: Let it grow naturally from experience
2. **Set quality threshold**: Only learn from successful paths (value > 0.7)
3. **Save RAG store**: Persist learned patterns between sessions
4. **Domain-specific stores**: Separate RAG stores for different domains
5. **Periodic cleanup**: Remove low-confidence patterns occasionally
6. **Monitor growth**: Track RAG store size and pattern quality
7. **Combine with reflection**: Learn from refined, high-quality reasoning

## Performance Considerations

### Memory

RAG store grows with learning:
- Each pattern: ~1-5 KB
- 100 patterns: ~100-500 KB
- 1000 patterns: ~1-5 MB

Manageable for most use cases.

### Speed

Pattern retrieval is fast:
- Keyword matching: <1ms
- Weight aggregation: <1ms
- Overall overhead: negligible

### Quality

Learning improves with data:
- 0-10 problems: Initial patterns forming
- 10-50 problems: Noticeable improvement
- 50+ problems: Strong domain adaptation

## Advanced Usage

### Custom Pattern Extraction

```python
from mcts_reasoning.learning import PathLearner

class CustomLearner(PathLearner):
    def extract_pattern(self, path, problem, value):
        # Custom pattern extraction logic
        pattern = super().extract_pattern(path, problem, value)
        # Add custom features
        pattern['custom_feature'] = self.analyze_custom(path)
        return pattern

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Problem")
    .with_path_learner(CustomLearner(llm))
)
```

### Guided Exploration

Use learned patterns to guide exploration:

```python
# Get learned weights
weights = rag_store.get_action_weights_for_problem(problem)

# Apply to MCTS
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question(problem)
    .with_compositional_actions(enabled=True, weights=weights)
)
```

### Pattern Analysis

Analyze learned patterns:

```python
for guidance in rag_store.guidance:
    print(f"Problem pattern: {guidance.problem_pattern}")
    print(f"Success rate: {guidance.success_rate:.3f}")
    print(f"Recommended operations: {guidance.recommended_operations}")
    print(f"Top weights: {dict(list(guidance.weights['cognitive_op'].items())[:3])}")
    print()
```

## Troubleshooting

### RAG Store Not Growing

If patterns aren't being learned:
- Check `auto_learn=True`
- Verify paths have sufficient value
- Lower `min_value_threshold`
- Check learning statistics

### Poor Transfer

If patterns don't transfer well:
- Problems may be too different
- Pattern keywords may be too specific
- Try broader problem categorization
- Increase number of learned examples

### Overfitting

If learning reduces diversity:
- Lower pattern weights (use smaller multipliers)
- Increase exploration constant
- Limit max_patterns_per_search
- Periodically reset RAG store

## Next Steps

- [Compositional Actions](compositional-actions.md) - What the system learns about
- [Meta-Reasoning](meta-reasoning.md) - Real-time adaptive learning
- [Reflection](reflection.md) - Improving pattern quality
- [Benchmarking](../advanced/benchmarking.md) - Measuring learning impact
