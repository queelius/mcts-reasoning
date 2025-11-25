# Benchmarking Suite

Comprehensive benchmarking framework for quantitatively evaluating MCTS reasoning system performance on standard datasets.

## Overview

The benchmarking suite provides:
- **Dataset Loading**: Load problems from JSON/CSV files
- **Automated Testing**: Run MCTS on multiple problems with metrics tracking
- **Configuration Comparison**: Compare different MCTS configurations
- **Detailed Metrics**: Accuracy, time, LLM calls, tree statistics
- **Result Export**: Save results to JSON for analysis

## Quick Start

```python
from mcts_reasoning.benchmarking import BenchmarkRunner
from mcts_reasoning.dataset_loader import DatasetLoader
from mcts_reasoning import get_llm

# Load dataset
problems = DatasetLoader.load_json("benchmarks/gsm8k_sample.json")

# Create runner with LLM
llm = get_llm("openai", model="gpt-4")
runner = BenchmarkRunner(llm)

# Configure MCTS
config = {
    'llm': llm,
    'compositional': True,
    'solution_detection': True,
    'meta_reasoning': False,
    'reflection': False,
    'exploration': 1.414,
    'max_depth': 4
}

# Run benchmark
results, summary = runner.run_benchmark(
    problems,
    config,
    simulations=50,
    max_problems=10,
    verbose=True
)

# Print results
print(f"Accuracy: {summary.accuracy:.1%}")
print(f"Avg Time: {summary.avg_time:.2f}s")
print(f"Avg LLM Calls: {summary.avg_llm_calls:.0f}")

# Save results
runner.save_results(results, summary, "results.json")
```

## Dataset Format

### JSON Format

```json
{
  "name": "Dataset Name",
  "description": "Description of the dataset",
  "source": "https://source.com",
  "problems": [
    {
      "id": "prob_001",
      "category": "algebra",
      "question": "Solve for x: 2x + 5 = 17",
      "answer": "6",
      "difficulty": "easy",
      "metadata": {
        "answer_type": "number",
        "topic": "linear_equations"
      }
    }
  ]
}
```

### CSV Format

```csv
id,category,question,answer,difficulty,answer_type
prob_001,algebra,"Solve for x: 2x + 5 = 17",6,easy,number
prob_002,arithmetic,"What is 15% of 80?",12,easy,number
```

## Included Datasets

### GSM8K Sample (`benchmarks/gsm8k_sample.json`)
- **Source**: Grade School Math 8K dataset
- **Problems**: 10 sample problems
- **Categories**: arithmetic, word_problem, multi_step, percentage
- **Difficulty**: easy to medium

Sample problems:
- "Janet has 16 apples. She gives 7 apples to her friend. How many apples does Janet have left?"
- "A baker makes 12 cupcakes in the morning and 15 cupcakes in the afternoon. If each cupcake costs $2, how much money will the baker make if all cupcakes are sold?"

### MATH Sample (`benchmarks/math_sample.json`)
- **Source**: MATH dataset (algebra, number theory)
- **Problems**: 10 sample problems
- **Categories**: algebra, number_theory, arithmetic
- **Difficulty**: easy to medium

Sample problems:
- "Solve for x: x^2 - 5x + 6 = 0. Give the smallest solution."
- "What is the greatest common divisor of 48 and 18?"

### Logic Sample (`benchmarks/logic_sample.json`)
- **Source**: Custom logic problems
- **Problems**: 10 sample problems
- **Categories**: deduction, set_theory, sequence, counting, probability
- **Difficulty**: easy to medium

Sample problems:
- "If all cats are mammals, and all mammals are animals, are all cats animals?"
- "In a group of 50 people, 30 like coffee, 25 like tea, and 10 like both. How many like neither?"

## Loading Datasets

### Load Single Dataset

```python
from mcts_reasoning.dataset_loader import DatasetLoader

# Load from JSON
problems = DatasetLoader.load_json("benchmarks/gsm8k_sample.json")

# Load from CSV
problems = DatasetLoader.load_csv(
    "benchmarks/my_dataset.csv",
    id_col="id",
    question_col="question",
    answer_col="answer"
)
```

### Load All Datasets from Directory

```python
# Load all JSON files in directory
datasets = DatasetLoader.load_directory("benchmarks/", pattern="*.json")

for name, problems in datasets.items():
    print(f"{name}: {len(problems)} problems")
```

### Filter Problems

```python
# Filter by category
algebra_problems = DatasetLoader.filter_problems(
    problems,
    category="algebra"
)

# Filter by difficulty
easy_problems = DatasetLoader.filter_problems(
    problems,
    difficulty="easy"
)

# Limit count
first_5 = DatasetLoader.filter_problems(
    problems,
    max_count=5
)

# Combine filters
filtered = DatasetLoader.filter_problems(
    problems,
    category="algebra",
    difficulty="medium",
    max_count=10
)
```

## Running Benchmarks

### Basic Benchmark

```python
from mcts_reasoning.benchmarking import BenchmarkRunner
from mcts_reasoning import get_llm

llm = get_llm("anthropic")
runner = BenchmarkRunner(llm)

config = {
    'llm': llm,
    'compositional': True,
    'solution_detection': True
}

results, summary = runner.run_benchmark(
    problems,
    config,
    simulations=50,
    verbose=True
)
```

### Configuration Comparison (Ablation Study)

```python
configurations = {
    "baseline": {
        'llm': llm,
        'compositional': True,
        'solution_detection': False,
        'meta_reasoning': False,
        'reflection': False
    },
    "with_solution_detection": {
        'llm': llm,
        'compositional': True,
        'solution_detection': True,
        'meta_reasoning': False,
        'reflection': False
    },
    "with_meta_reasoning": {
        'llm': llm,
        'compositional': True,
        'solution_detection': True,
        'meta_reasoning': True,
        'reflection': False
    },
    "full_features": {
        'llm': llm,
        'compositional': True,
        'solution_detection': True,
        'meta_reasoning': True,
        'reflection': True,
        'learning': True
    }
}

comparison = runner.compare_configurations(
    problems,
    configurations,
    simulations=50,
    max_problems=10
)

# Compare results
for config_name, (results, summary) in comparison.items():
    print(f"{config_name}:")
    print(f"  Accuracy: {summary.accuracy:.1%}")
    print(f"  Avg LLM Calls: {summary.avg_llm_calls:.0f}")
```

## Metrics and Results

### BenchmarkResult

Results for each individual problem:

```python
@dataclass
class BenchmarkResult:
    problem_id: str              # Problem identifier
    category: str                # Problem category
    question: str                # Question text
    ground_truth: str            # Correct answer
    predicted_answer: str        # MCTS predicted answer
    is_correct: bool             # Whether prediction matches ground truth
    confidence: float            # Best node value (0-1)
    time_seconds: float          # Time taken
    simulations: int             # Number of MCTS simulations
    tree_stats: Dict            # Tree statistics (nodes, depth, etc.)
    llm_calls: int              # Estimated LLM calls
    error: Optional[str]        # Error message if failed
```

### BenchmarkSummary

Aggregate statistics:

```python
@dataclass
class BenchmarkSummary:
    total_problems: int          # Total problems attempted
    correct: int                 # Number correct
    incorrect: int               # Number incorrect
    errors: int                  # Number with errors
    accuracy: float              # Accuracy (0-1)
    avg_time: float              # Average time per problem
    avg_simulations: float       # Average simulations
    avg_llm_calls: float         # Average LLM calls
    category_breakdown: Dict     # Per-category statistics
```

### Accessing Results

```python
# Overall metrics
print(f"Accuracy: {summary.accuracy:.1%}")
print(f"Correct: {summary.correct}/{summary.total_problems}")

# Category breakdown
for category, stats in summary.category_breakdown.items():
    print(f"{category}: {stats['accuracy']:.1%} "
          f"({stats['correct']}/{stats['total']})")

# Individual results
for result in results:
    if result.is_correct:
        print(f"✓ {result.question}")
    else:
        print(f"✗ {result.question}")
        print(f"  Expected: {result.ground_truth}")
        print(f"  Got: {result.predicted_answer}")
```

## Answer Extraction and Checking

The benchmark suite includes sophisticated answer extraction:

### Answer Extraction

```python
from mcts_reasoning.benchmarking import AnswerExtractor

extractor = AnswerExtractor()

# Extract answer from reasoning output
reasoning = """
Let's solve step by step:
x + 2 = 7
x = 7 - 2
x = 5

## Final Answer
The answer is 5.
"""

answer = extractor.extract_answer(reasoning)
# Returns: "The answer is 5"

# Normalize for comparison
normalized = extractor.normalize_answer(answer, answer_type="number")
# Returns: "5"
```

### Answer Checking

```python
from mcts_reasoning.benchmarking import AnswerChecker

checker = AnswerChecker()

# Exact match
checker.check_answer("5", "5")  # True

# Numerical tolerance
checker.check_answer("3.14159", "3.14", answer_type="number")  # False (outside tolerance)
checker.check_answer("3.14159", "3.14159265", answer_type="number")  # True (within tolerance)

# Contains match
checker.check_answer("The answer is 42", "42")  # True
```

## Saving and Loading Results

### Save Results

```python
# Save to JSON
runner.save_results(results, summary, "benchmark_results.json")
```

### Result JSON Format

```json
{
  "summary": {
    "total_problems": 10,
    "correct": 8,
    "incorrect": 2,
    "errors": 0,
    "accuracy": 0.8,
    "avg_time": 2.5,
    "avg_simulations": 50,
    "avg_llm_calls": 65,
    "category_breakdown": {
      "algebra": {
        "total": 5,
        "correct": 4,
        "accuracy": 0.8
      }
    }
  },
  "results": [
    {
      "problem_id": "math_001",
      "category": "algebra",
      "question": "Solve for x: 2x + 5 = 17",
      "ground_truth": "6",
      "predicted_answer": "6",
      "is_correct": true,
      "confidence": 0.85,
      "time_seconds": 2.3,
      "simulations": 50,
      "tree_stats": {
        "total_nodes": 50,
        "max_depth": 5,
        "best_value": 0.85
      },
      "llm_calls": 65
    }
  ]
}
```

## Example Benchmarks

### Example 1: Quick Validation

```python
"""Quick validation on easy problems."""
from mcts_reasoning.benchmarking import BenchmarkRunner
from mcts_reasoning.dataset_loader import DatasetLoader
from mcts_reasoning import get_llm

# Load easy problems only
all_problems = DatasetLoader.load_json("benchmarks/math_sample.json")
easy_problems = DatasetLoader.filter_problems(all_problems, difficulty="easy")

llm = get_llm("openai")
runner = BenchmarkRunner(llm)

config = {
    'llm': llm,
    'compositional': True,
    'solution_detection': True,
    'max_depth': 3
}

results, summary = runner.run_benchmark(
    easy_problems,
    config,
    simulations=30,
    verbose=True
)

print(f"Easy problems accuracy: {summary.accuracy:.1%}")
```

### Example 2: Ablation Study

```python
"""Study impact of different features."""
from mcts_reasoning.benchmarking import BenchmarkRunner
from mcts_reasoning.dataset_loader import DatasetLoader
from mcts_reasoning import get_llm
from mcts_reasoning.compositional.rag import get_math_compositional_rag

problems = DatasetLoader.load_json("benchmarks/math_sample.json")
llm = get_llm("anthropic")
rag = get_math_compositional_rag()

configs = {
    "no_features": {
        'llm': llm,
        'compositional': False
    },
    "compositional_only": {
        'llm': llm,
        'compositional': True
    },
    "compositional_rag": {
        'llm': llm,
        'compositional': True,
        'rag_store': rag
    },
    "compositional_rag_detection": {
        'llm': llm,
        'compositional': True,
        'rag_store': rag,
        'solution_detection': True
    }
}

runner = BenchmarkRunner(llm)
comparison = runner.compare_configurations(
    problems,
    configs,
    simulations=50
)

# Analyze impact of features
for name, (_, summary) in comparison.items():
    print(f"{name:30s} - Accuracy: {summary.accuracy:.1%}, "
          f"LLM Calls: {summary.avg_llm_calls:.0f}")
```

### Example 3: Performance vs Accuracy Tradeoff

```python
"""Compare different simulation counts."""
from mcts_reasoning.benchmarking import BenchmarkRunner
from mcts_reasoning.dataset_loader import DatasetLoader
from mcts_reasoning import get_llm

problems = DatasetLoader.load_json("benchmarks/gsm8k_sample.json")[:5]
llm = get_llm("openai")

config = {
    'llm': llm,
    'compositional': True,
    'solution_detection': True
}

runner = BenchmarkRunner(llm)

for sim_count in [10, 30, 50, 100]:
    results, summary = runner.run_benchmark(
        problems,
        config,
        simulations=sim_count,
        verbose=False
    )

    print(f"Simulations: {sim_count:3d} - "
          f"Accuracy: {summary.accuracy:.1%}, "
          f"Avg Time: {summary.avg_time:.1f}s, "
          f"LLM Calls: {summary.avg_llm_calls:.0f}")
```

## Best Practices

### 1. Start Small
Test on a few problems first to validate configuration:
```python
results, summary = runner.run_benchmark(
    problems[:5],  # First 5 problems
    config,
    simulations=30
)
```

### 2. Use Appropriate Simulations
- **Quick validation**: 20-30 simulations
- **Standard benchmarking**: 50 simulations
- **High accuracy**: 100+ simulations

### 3. Track LLM Costs
Monitor `avg_llm_calls` to estimate API costs:
```python
estimated_cost_per_call = 0.001  # $0.001 per call (example)
total_cost = summary.avg_llm_calls * summary.total_problems * estimated_cost_per_call
print(f"Estimated cost: ${total_cost:.2f}")
```

### 4. Compare Fairly
Use same problems and simulations when comparing configurations:
```python
comparison = runner.compare_configurations(
    problems=same_problems,  # Same for all
    configurations=configs,
    simulations=50  # Same for all
)
```

### 5. Save Results
Always save benchmark results for later analysis:
```python
runner.save_results(results, summary, f"results_{dataset_name}_{timestamp}.json")
```

## Testing

Run the benchmark test suite:

```bash
python test_benchmarking.py
```

This tests:
- Dataset loading from JSON/CSV
- Problem filtering
- Benchmark running with mock LLM
- Configuration comparison
- Result saving
- Real LLM benchmarking (if API keys available)

## Summary

The benchmarking suite provides:

✅ **Dataset Management**
- Load from JSON/CSV
- Filter by category/difficulty
- Multiple dataset support

✅ **Automated Testing**
- Run on multiple problems
- Track detailed metrics
- Handle errors gracefully

✅ **Sophisticated Answer Checking**
- Extract answers from reasoning
- Normalize for comparison
- Support multiple answer types

✅ **Configuration Comparison**
- Ablation studies
- Feature impact analysis
- Performance tradeoffs

✅ **Result Export**
- Save to JSON
- Detailed per-problem results
- Aggregate statistics

This enables rigorous, quantitative evaluation of the MCTS reasoning system.
