# Compositional Actions

MCTS-Reasoning features an advanced compositional prompting system that enables rich, structured reasoning through a 5-dimensional action space.

## Overview

Traditional LLM reasoning often uses simple, predefined prompts. The compositional action system creates a **rich action space** with over 30,000 possible combinations by combining five dimensions:

- **ω (omega)**: Cognitive Operation - What to do
- **φ (phi)**: Focus Aspect - What to focus on
- **σ (sigma)**: Reasoning Style - How to approach it
- **κ (kappa)**: Connection Type - How to connect ideas
- **τ (tau)**: Output Format - How to present it

This enables MCTS to explore diverse reasoning strategies systematically.

## The Five Dimensions

### Cognitive Operations (ω)

What cognitive operation to perform:

- `DECOMPOSE` - Break problem into smaller parts
- `ANALYZE` - Examine components and relationships
- `SYNTHESIZE` - Combine insights into coherent whole
- `VERIFY` - Check correctness and validity
- `GENERATE` - Create new ideas or solutions
- `COMPARE` - Evaluate alternatives
- `ABSTRACT` - Identify patterns and generalizations
- `INSTANTIATE` - Apply general principles to specific cases
- `CRITIQUE` - Evaluate weaknesses and limitations
- `REFINE` - Improve existing solution
- `EXPLORE` - Investigate new possibilities

### Focus Aspects (φ)

What aspect to focus attention on:

- `STRUCTURE` - Organization and relationships
- `PATTERNS` - Recurring themes and regularities
- `ASSUMPTIONS` - Underlying premises
- `CONSTRAINTS` - Limitations and requirements
- `CORRECTNESS` - Accuracy and validity
- `EDGE_CASES` - Boundary conditions
- `EFFICIENCY` - Performance and optimization
- `GENERALITY` - Applicability to broader contexts
- `CLARITY` - Understandability and precision
- `COMPLETENESS` - Coverage and thoroughness

### Reasoning Styles (σ)

How to approach the reasoning:

- `SYSTEMATIC` - Methodical and organized
- `INTUITIVE` - Insight-driven and creative
- `CRITICAL` - Questioning and analytical
- `CREATIVE` - Novel and unconventional
- `FORMAL` - Rigorous and precise
- `PRAGMATIC` - Practical and goal-oriented
- `HOLISTIC` - Big-picture and integrative
- `REDUCTIVE` - Bottom-up and detailed

### Connection Types (κ)

How to connect ideas:

- `THEREFORE` - Logical consequence
- `HOWEVER` - Contrast or limitation
- `BUILDING_ON` - Extension of previous idea
- `ALTERNATIVELY` - Different approach
- `SPECIFICALLY` - Concrete instantiation
- `GENERALLY` - Broader principle
- `FOR_EXAMPLE` - Illustration
- `IN_CONTRAST` - Opposite perspective

### Output Formats (τ)

How to structure the output:

- `STEPS` - Numbered procedure
- `LIST` - Bullet points
- `NARRATIVE` - Flowing explanation
- `MATHEMATICAL` - Formal notation
- `CODE` - Programming representation
- `DIAGRAM` - Visual structure (described)
- `TABLE` - Tabular organization
- `PROOF` - Logical derivation

## Usage

### Basic Usage

```python
from mcts_reasoning import (
    ComposingPrompt,
    CognitiveOperation,
    FocusAspect,
    ReasoningStyle
)

# Build a structured prompt
prompt = (
    ComposingPrompt()
    .cognitive_op(CognitiveOperation.DECOMPOSE)
    .focus(FocusAspect.STRUCTURE)
    .style(ReasoningStyle.SYSTEMATIC)
    .problem_context("Find all prime numbers less than 100")
    .build()
)

print(prompt)
```

Output:
```
Problem: Find all prime numbers less than 100

Let me break this problem down systematically. I'll focus on the structural
relationships and organization. I'll approach this systematically and methodically.
```

### With MCTS

```python
from mcts_reasoning import ReasoningMCTS, get_llm

llm = get_llm()

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Solve x^2 - 5x + 6 = 0")
    .with_compositional_actions(enabled=True)  # Enable compositional actions
    .with_exploration(1.414)
)

mcts.search("Let's solve this equation:", simulations=50)
```

When compositional actions are enabled, MCTS samples from the 5D action space during expansion, creating diverse reasoning strategies automatically.

### Action Vectors

Get the compositional structure of an action:

```python
prompt = (
    ComposingPrompt()
    .cognitive_op(CognitiveOperation.VERIFY)
    .focus(FocusAspect.CORRECTNESS)
)

vector = prompt.get_action_vector()
print(vector)
# {'omega': 'verify', 'phi': 'correctness', 'sigma': None, 'kappa': None, 'tau': None}
```

## Weighted Sampling

Bias the action space toward specific operations or styles:

```python
from mcts_reasoning import ComposingPrompt, CognitiveOperation, ReasoningStyle

# Define weights
weights = {
    'cognitive_op': {
        CognitiveOperation.DECOMPOSE: 3.0,  # 3x more likely
        CognitiveOperation.VERIFY: 2.0,     # 2x more likely
        CognitiveOperation.ANALYZE: 1.0,    # Normal
    },
    'style': {
        ReasoningStyle.SYSTEMATIC: 3.0,     # Strongly prefer
        ReasoningStyle.FORMAL: 1.5,
    }
}

# Sample with weights
for i in range(5):
    prompt = ComposingPrompt.sample_action(weights)
    vector = prompt.get_action_vector()
    print(f"Sample {i+1}: ω={vector['omega']}, σ={vector['sigma']}")
```

This is useful for:
- Domain-specific reasoning (e.g., prefer formal style for math)
- Problem-specific strategies (e.g., emphasize verification for correctness)
- Learning from experience (increase weights for successful operations)

## Action Selection in MCTS

When compositional actions are enabled, MCTS uses the `ActionSelector` class:

```python
from mcts_reasoning.compositional import ActionSelector

selector = ActionSelector()

# Get random action
action = selector.get_action(current_state="Working on problem...")

# Get weighted action
action_weighted = selector.get_action(
    current_state="Working on problem...",
    weights=weights
)
```

During MCTS expansion:
1. Sample an action from the compositional space (possibly weighted)
2. Build the prompt using `ComposingPrompt`
3. Send to LLM with the current reasoning context
4. LLM generates next reasoning step
5. Add new node to tree

## Benefits

### Diversity

The compositional space enables **systematic exploration** of reasoning strategies:

- Try decomposition with systematic style
- Try analysis with critical style
- Try synthesis with creative style
- And 30,000+ other combinations

### Structure

Each action has a **clear semantic meaning**:

- Not just "prompt variant #1453"
- But "decompose problem structure systematically"
- Enables interpretability and analysis

### Adaptability

Weights can be **learned from experience**:

- Track which operations succeed
- Increase weights for successful strategies
- Decrease weights for unsuccessful ones
- Adapt to problem domain over time

See [Learning System](learning.md) for details.

### Compositionality

Dimensions can be **mixed and matched**:

- Systematic decomposition
- Creative synthesis
- Critical verification
- Formal analysis
- And any other combination

## Example: Math Problem

```python
from mcts_reasoning import ReasoningMCTS, get_llm

llm = get_llm()

# Math problems benefit from:
# - Decomposition and analysis
# - Systematic and formal styles
# - Verification of correctness

weights = {
    'cognitive_op': {
        CognitiveOperation.DECOMPOSE: 2.0,
        CognitiveOperation.ANALYZE: 2.0,
        CognitiveOperation.VERIFY: 3.0,
    },
    'style': {
        ReasoningStyle.SYSTEMATIC: 2.5,
        ReasoningStyle.FORMAL: 2.0,
    },
    'focus': {
        FocusAspect.CORRECTNESS: 3.0,
        FocusAspect.STRUCTURE: 2.0,
    }
}

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Prove that √2 is irrational")
    .with_compositional_actions(enabled=True, weights=weights)
    .with_exploration(1.414)
)

mcts.search("Let's prove this:", simulations=50)
```

## Example: Creative Problem

```python
# Creative problems benefit from:
# - Exploration and generation
# - Creative and intuitive styles
# - Alternative approaches

weights = {
    'cognitive_op': {
        CognitiveOperation.EXPLORE: 3.0,
        CognitiveOperation.GENERATE: 3.0,
        CognitiveOperation.COMPARE: 2.0,
    },
    'style': {
        ReasoningStyle.CREATIVE: 3.0,
        ReasoningStyle.INTUITIVE: 2.0,
    },
    'connection': {
        ConnectionType.ALTERNATIVELY: 2.0,
        ConnectionType.BUILDING_ON: 2.0,
    }
}

mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Design a novel data structure for social network analysis")
    .with_compositional_actions(enabled=True, weights=weights)
    .with_exploration(2.0)  # Higher exploration for creativity
)

mcts.search("Let's brainstorm:", simulations=50)
```

## Integration with Other Features

### With Meta-Reasoning

Meta-reasoning can suggest which cognitive operation to try next:

```python
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Complex problem")
    .with_compositional_actions(enabled=True)
    .with_meta_reasoning(enabled=True)  # LLM suggests next operation
)
```

The meta-reasoner analyzes the current state and biases action selection toward productive operations.

See [Meta-Reasoning](meta-reasoning.md) for details.

### With Learning

The learning system extracts patterns from successful reasoning:

```python
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Problem")
    .with_compositional_actions(enabled=True)
    .with_learning(enabled=True)  # Learn successful action patterns
)
```

After solving problems, the system identifies which cognitive operations led to success and increases their weights for similar problems.

See [Learning System](learning.md) for details.

### With Reflection

Reflection can critique the reasoning style:

```python
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("Problem")
    .with_compositional_actions(enabled=True)
    .with_reflection(enabled=True)  # Critique reasoning approach
)
```

The reflection system evaluates whether the chosen reasoning style and operations were effective.

See [Reflection & Critique](reflection.md) for details.

## Technical Details

### Action Space Size

With all dimensions:
- 11 cognitive operations
- 10 focus aspects
- 8 reasoning styles
- 8 connection types
- 8 output formats

Total combinations: **11 × 10 × 8 × 8 × 8 = 56,320 possible actions**

In practice, not all dimensions are always used, giving effective space of ~30,000 actions.

### Sampling Strategy

Actions are sampled using weighted random selection:

1. For each dimension, compute probabilities from weights
2. Sample dimension values independently
3. Combine into complete action vector
4. Build prompt using template

### Performance

Compositional actions add minimal overhead:
- Action sampling: <1ms
- Prompt building: <1ms
- Main cost is still LLM inference

## Best Practices

1. **Start with defaults**: Enable compositional actions without weights first
2. **Add weights gradually**: After observing behavior, add light biases
3. **Don't over-weight**: Keep weights in 1.0-3.0 range for diversity
4. **Use domain knowledge**: Weight operations that make sense for your domain
5. **Enable learning**: Let the system learn weights from experience
6. **Monitor diversity**: Check that different actions are being tried
7. **Combine with meta-reasoning**: For adaptive action selection

## Next Steps

- [Meta-Reasoning](meta-reasoning.md) - LLM-guided action selection
- [Learning System](learning.md) - Learning from successful patterns
- [Reflection](reflection.md) - Self-critique of reasoning approach
- [Prompt Features](prompt-features.md) - Advanced prompt engineering
