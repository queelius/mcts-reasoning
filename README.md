# MCTS-Reasoning

Monte Carlo Tree Search for LLM-based step-by-step reasoning.

A clean, canonical implementation of MCTS that explores multiple reasoning paths to find high-quality solutions.

## Features

- **Canonical MCTS**: Selection (UCB1), Expansion, Tree-building Rollouts, Backpropagation
- **Multiple Evaluators**: LLM-as-judge, ground truth, numeric (with tolerance), process quality
- **Sampling Strategies**: Value-based, visit-based, diverse, top-k
- **Pluggable Providers**: OpenAI, Anthropic, Ollama, Mock (for testing)
- **CLI Tool**: `mcts-reason` for command-line reasoning
- **Full Test Coverage**: 255+ tests

## Installation

```bash
git clone https://github.com/queelius/mcts-reasoning.git
cd mcts-reasoning
pip install -e .

# With OpenAI support
pip install -e ".[openai]"

# With Anthropic support
pip install -e ".[anthropic]"

# Everything
pip install -e ".[all]"
```

## Quick Start

### Command Line

```bash
# Simple question
mcts-reason "What is 15*7+23?"

# With ground truth for evaluation
mcts-reason "What is 15*7+23?" --answer 128 --simulations 10

# With specific provider
mcts-reason "Explain photosynthesis" --provider ollama --model llama3.2

# JSON output for scripting
mcts-reason "What is 2+2?" --json

# Verbose with tree visualization
mcts-reason "Solve: 5*6+10" --answer 40 -v --consistency
```

### Python API

```python
from mcts_reasoning import (
    MCTS,
    LLMGenerator,
    NumericEvaluator,
    ProcessEvaluator,
    PathSampler,
    get_llm,
)

# Get LLM provider (auto-detect or specify)
llm = get_llm()  # Auto-detect from environment
# llm = get_llm("ollama", model="llama3.2")
# llm = get_llm("openai", model="gpt-4")

# Create generator and evaluator
generator = LLMGenerator(llm=llm, temperature=0.7)
evaluator = ProcessEvaluator(
    answer_evaluator=NumericEvaluator(ground_truth=128),
    answer_weight=0.7,
    process_weight=0.3,
)

# Create MCTS and search
mcts = MCTS(
    generator=generator,
    evaluator=evaluator,
    exploration_constant=1.414,
    max_rollout_depth=5,
)

result = mcts.search("What is 15*7+23?", simulations=20)

print(f"Answer: {result.best_answer}")
print(f"Confidence: {result.confidence:.1%}")
print(f"Stats: {result.stats}")

# Sample diverse paths
sampler = PathSampler(result.root)
paths = sampler.sample(n=5, strategy="diverse")
for path in paths:
    print(f"Answer: {path.answer}, Value: {path.value:.2f}")
```

## Core Components

### MCTS Search

```python
from mcts_reasoning import MCTS, MockGenerator, MockEvaluator

mcts = MCTS(
    generator=MockGenerator(),
    evaluator=MockEvaluator(),
    exploration_constant=1.414,  # UCB1 exploration
    max_children_per_node=3,     # Branching factor
    max_rollout_depth=5,         # Max reasoning steps
)

result = mcts.search(question="What is 2+2?", simulations=50)
```

### Evaluators

```python
from mcts_reasoning import (
    LLMEvaluator,        # LLM-as-judge scoring
    GroundTruthEvaluator,# Compare to known answer
    NumericEvaluator,    # Math with tolerance
    ProcessEvaluator,    # Reasoning quality
    CompositeEvaluator,  # Combine multiple
)

# For math problems
evaluator = NumericEvaluator(
    ground_truth=42.0,
    rel_tol=0.01,  # 1% tolerance
)

# For evaluating reasoning process
evaluator = ProcessEvaluator(
    answer_evaluator=NumericEvaluator(ground_truth=42.0),
    answer_weight=0.7,
    process_weight=0.3,
)

# For open-ended problems
evaluator = LLMEvaluator(llm=my_llm, temperature=0.1)
```

### Sampling

```python
from mcts_reasoning import PathSampler

sampler = PathSampler(result.root)

# By value (quality)
best_paths = sampler.sample(n=5, strategy="value")

# By visits (confidence)
confident_paths = sampler.sample(n=5, strategy="visits")

# Maximize diversity
diverse_paths = sampler.sample(n=5, strategy="diverse")

# Answer distribution
dist = sampler.get_answer_distribution()
print(dist)  # {'42': {'count': 3, 'avg_value': 0.9}, ...}

# Consistency score
score = sampler.consistency_score()
print(f"Consistency: {score:.1%}")
```

### Terminal Detection

```python
from mcts_reasoning import (
    MarkerTerminalDetector,    # Look for "ANSWER:"
    BoxedTerminalDetector,     # Look for \boxed{}
    MultiMarkerTerminalDetector,  # Multiple markers
)

# Custom marker
detector = MarkerTerminalDetector(marker="FINAL:")

# Math benchmark style
detector = BoxedTerminalDetector()

# Multiple formats
detector = MultiMarkerTerminalDetector(
    markers=["ANSWER:", "\\boxed{", "Therefore, the answer is"]
)
```

## Environment Variables

```bash
# LLM Provider selection
export LLM_PROVIDER=ollama  # or openai, anthropic
export OLLAMA_BASE_URL=http://localhost:11434
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key
```

## Architecture

```
mcts_reasoning/
├── node.py          # Tree node with UCB1
├── mcts.py          # MCTS search algorithm
├── generator.py     # LLM continuation generation
├── evaluator.py     # Terminal state evaluation
├── terminal.py      # Terminal detection
├── actions.py       # Action space abstraction
├── sampling.py      # Path sampling strategies
├── cli.py           # Command-line interface
└── compositional/
    └── providers.py # LLM provider adapters
```

## MCTS Algorithm

1. **Selection**: Navigate tree using UCB1 to find promising leaf
2. **Expansion**: Generate new reasoning step via LLM
3. **Rollout**: Continue reasoning until terminal or max depth
4. **Backpropagation**: Update values up the tree

```
UCB1 = average_value + c * sqrt(ln(parent_visits) / visits)
```

## Testing

```bash
# Run all tests
pytest tests/

# With coverage
pytest --cov=mcts_reasoning tests/

# Specific module
pytest tests/test_sampling.py -v
```

## License

MIT License

## Citation

```bibtex
@software{mcts_reasoning,
  title = {MCTS-Reasoning: Monte Carlo Tree Search for LLM Reasoning},
  author = {Towell, Alex},
  year = {2024},
  url = {https://github.com/queelius/mcts-reasoning}
}
```
