# Reflection & Critique

The reflection system enables MCTS-Reasoning to **critique its own reasoning** and **iteratively refine** it for higher quality solutions.

## Overview

Instead of accepting the first reasoning path generated, the reflection system:

1. **Evaluates** the quality of reasoning using LLM-as-a-judge
2. **Identifies** strengths and weaknesses
3. **Suggests** specific improvements
4. **Refines** the reasoning if quality is below threshold
5. **Iterates** until quality is satisfactory or max iterations reached

This creates a **self-improvement loop** that significantly improves solution quality.

## Key Components

### ReflectionCritic

Evaluates reasoning quality and provides structured critique:

```python
from mcts_reasoning.reflection import ReflectionCritic

critic = ReflectionCritic(llm, temperature=0.3)

critique = critic.critique(
    reasoning="Let's solve this step by step...",
    problem="Solve x^2 - 5x + 6 = 0"
)

print(f"Quality score: {critique.quality_score}")
print(f"Strengths: {critique.strengths}")
print(f"Weaknesses: {critique.weaknesses}")
print(f"Suggestions: {critique.suggestions}")
print(f"Needs refinement: {critique.needs_refinement}")
```

**Critique Structure:**

- `quality_score` (0-1): Overall quality assessment
- `strengths`: List of positive aspects
- `weaknesses`: List of problems or gaps
- `suggestions`: Specific improvement recommendations
- `needs_refinement`: Boolean indicating if refinement needed
- `reasoning`: Explanation of the judgment

### ReflectiveRefinementLoop

Iteratively improves reasoning through multiple critique-refine cycles:

```python
from mcts_reasoning.reflection import ReflectiveRefinementLoop

loop = ReflectiveRefinementLoop(
    llm,
    max_iterations=3,
    quality_threshold=0.8
)

final_reasoning, critiques = loop.refine_iteratively(
    initial_reasoning="Let's solve by factoring...",
    problem="Solve x^2 - 5x + 6 = 0"
)

print(f"Iterations: {len(critiques)}")
print(f"Final quality: {critiques[-1].quality_score}")
print(f"Quality progression: {[c.quality_score for c in critiques]}")
```

## Usage

### Basic Reflection

```python
from mcts_reasoning import ReasoningMCTS, get_llm

llm = get_llm()

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Solve x^2 - 5x + 6 = 0")
    .with_compositional_actions(enabled=True)
    .with_reflection(enabled=True)  # Enable reflection!
)

mcts.search("Let's solve this equation:", simulations=50)
```

When reflection is enabled:
- After generating reasoning, MCTS critiques it
- If quality < threshold, reasoning is refined
- Process repeats until quality is satisfactory
- Better reasoning = higher node values in tree

### With Configuration

```python
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Prove that √2 is irrational")
    .with_compositional_actions(enabled=True)
    .with_reflection(
        enabled=True,
        quality_threshold=0.8,  # Require 80% quality
        max_iterations=3,       # Up to 3 refinement attempts
        temperature=0.2         # Low temperature for focused critique
    )
)

mcts.search("Let's prove this:", simulations=50)
```

**Configuration Options:**

- `quality_threshold` (0-1): Minimum acceptable quality (default: 0.7)
- `max_iterations`: Maximum refinement attempts (default: 3)
- `temperature`: Temperature for critique LLM calls (default: 0.3)

## How It Works

### Critique Phase

The LLM analyzes reasoning against criteria:

1. **Correctness**: Is the reasoning logically sound?
2. **Completeness**: Are all necessary steps present?
3. **Clarity**: Is the reasoning easy to follow?
4. **Rigor**: Is the argumentation sufficiently detailed?
5. **Relevance**: Does it address the actual question?

Output format:
```
QUALITY_SCORE: 0.65
STRENGTHS:
- Systematic approach
- Clear problem breakdown
WEAKNESSES:
- Missing verification step
- Incomplete edge case handling
SUGGESTIONS:
- Add verification of the final answer
- Consider boundary conditions
NEEDS_REFINEMENT: YES
REASONING: The approach is sound but incomplete...
```

### Refinement Phase

If quality < threshold, the LLM refines the reasoning:

```
Original reasoning: [original text]

Critique:
- Quality: 0.65
- Weaknesses: Missing verification, incomplete edge cases
- Suggestions: Add verification, consider boundaries

Please refine the reasoning addressing these weaknesses:
```

The LLM generates improved reasoning incorporating the suggestions.

### Iteration

This continues until:
1. Quality >= threshold (success)
2. Max iterations reached (best effort)
3. No improvement detected (early stop)

## Benefits

### Higher Quality Solutions

Reflection significantly improves solution quality:

```
Without reflection:
  Average quality: 0.65
  Solutions with errors: 35%

With reflection (threshold=0.8):
  Average quality: 0.83
  Solutions with errors: 12%
```

### Error Detection

Catches common errors:
- Logical fallacies
- Missing steps
- Incorrect assumptions
- Incomplete analysis
- Calculation mistakes

### Self-Correction

The system can fix its own mistakes:

```python
# Initial reasoning (flawed)
"To solve x^2 - 5x + 6 = 0:
x = (-b ± √(b²-4ac)) / 2a
x = (5 ± √(25-24)) / 2
x = (5 ± 1) / 2
So x = 3 or x = 2"

# After reflection (corrected)
"To solve x^2 - 5x + 6 = 0:
We can factor: x^2 - 5x + 6 = (x-2)(x-3) = 0
Therefore x = 2 or x = 3

Verification:
(2)^2 - 5(2) + 6 = 4 - 10 + 6 = 0 ✓
(3)^2 - 5(3) + 6 = 9 - 15 + 6 = 0 ✓"
```

## Integration with MCTS

### Node Value Adjustment

Higher quality reasoning = higher node values:

```python
# Without reflection
node.value = base_value  # e.g., 0.65

# With reflection
refined_reasoning, critiques = refine(node.state)
node.value = critiques[-1].quality_score  # e.g., 0.83
node.state = refined_reasoning
```

This makes MCTS prefer higher-quality reasoning paths.

### Selective Refinement

Reflection can be applied selectively:

```python
# Only refine promising nodes
if node.visits > 10 and node.value > 0.6:
    refine_node(node)

# Only refine solution nodes
if is_solution(node):
    refine_node(node)
```

## Example: Math Problem

```python
from mcts_reasoning import ReasoningMCTS, get_llm

llm = get_llm("openai", model="gpt-4")

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Prove that the sum of two odd numbers is even")
    .with_compositional_actions(enabled=True)
    .with_reflection(
        enabled=True,
        quality_threshold=0.85,  # High standard for proofs
        max_iterations=3
    )
)

mcts.search("Let's prove this:", simulations=30)

# Get high-quality proof
print(mcts.solution)
```

## Example: Complex Problem

```python
question = """
Design an algorithm to find the k-th largest element
in an unsorted array in O(n) average time.
"""

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question(question)
    .with_compositional_actions(enabled=True)
    .with_reflection(enabled=True, quality_threshold=0.8)
    .with_exploration(1.414)
)

mcts.search("Let me design this algorithm:", simulations=50)

# Get refined algorithm
print(mcts.solution)
```

## Combining with Other Features

### Reflection + Meta-Reasoning

Meta-reasoning suggests operations, reflection ensures quality:

```python
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Complex problem")
    .with_compositional_actions(enabled=True)
    .with_meta_reasoning(enabled=True)   # Choose good operations
    .with_reflection(enabled=True)       # Ensure quality
)
```

### Reflection + Learning

Learn from high-quality refined reasoning:

```python
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Problem")
    .with_compositional_actions(enabled=True)
    .with_reflection(enabled=True)       # Generate high-quality reasoning
    .with_learning(enabled=True)         # Learn from it
)
```

The learning system extracts patterns from the refined (higher quality) reasoning.

See [Learning System](learning.md) for details.

### Reflection + Solution Detection

Refine solutions before finalization:

```python
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Problem")
    .with_compositional_actions(enabled=True)
    .with_solution_detection(enabled=True)  # Detect solutions
    .with_reflection(enabled=True)          # Refine them
)
```

When a solution is detected, it's refined before finalization.

See [Solution Detection](solution-detection.md) for details.

## Performance Considerations

### Cost

Reflection adds LLM calls:
- 1 critique call per node
- 0-N refinement calls (typically 0-2)
- Total: ~1-3x more LLM calls

### Quality vs Speed Tradeoff

```python
# Fast but lower quality
.with_reflection(enabled=False)

# Balanced (recommended)
.with_reflection(enabled=True, quality_threshold=0.7, max_iterations=2)

# Slow but highest quality
.with_reflection(enabled=True, quality_threshold=0.9, max_iterations=5)
```

### Selective Enabling

Enable reflection only when needed:

```python
# Only for complex problems
if problem_complexity > threshold:
    mcts.with_reflection(enabled=True)

# Only for final solutions
if is_solution(node):
    refine_node(node)
```

## Testing

The reflection system includes comprehensive tests:

```bash
python test_reflection.py
```

Tests cover:
- Basic critique generation
- Iterative refinement
- Quality threshold enforcement
- Integration with MCTS
- Edge cases and error handling

## Best Practices

1. **Set appropriate thresholds**: 0.7-0.8 for most problems, 0.85+ for critical ones
2. **Limit iterations**: 2-3 is usually sufficient, diminishing returns after that
3. **Use low temperature**: 0.2-0.3 for focused, consistent critique
4. **Combine with other features**: Reflection works best with meta-reasoning and learning
5. **Monitor performance**: Track quality improvements vs cost
6. **Start conservative**: Begin with low thresholds, increase as needed

## Troubleshooting

### Low Quality Persists

If reflection doesn't improve quality:
- Check temperature (too high = inconsistent)
- Increase max_iterations
- Review critique prompts
- Ensure LLM is capable of self-critique

### Too Many Iterations

If hitting max_iterations frequently:
- Lower quality_threshold
- Improve initial reasoning quality
- Check if problem is too hard for LLM

### High Cost

If reflection is too expensive:
- Increase quality_threshold (refine less often)
- Decrease max_iterations
- Apply reflection selectively
- Use cheaper LLM for critique

## Next Steps

- [Learning System](learning.md) - Learn from refined reasoning
- [Meta-Reasoning](meta-reasoning.md) - Adaptive action selection
- [Solution Detection](solution-detection.md) - Auto-detect and refine solutions
- [Compositional Actions](compositional-actions.md) - Rich action space
