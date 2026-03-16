# mcts-reasoning v0.6: Architecture Redesign

**Date**: 2026-03-16
**Status**: Draft
**Version**: 0.5.2 -> 0.6.0 (breaking)

## Overview

Strip ~4,000 lines of speculative code from the library, restructure around composable ABCs where every strategic decision point is pluggable, add a benchmarking and evaluation framework, and expose the engine as both an MCP server (primary) and CLI (secondary, feature parity).

## Identity

A composable MCTS framework for LLM reasoning. Every decision point is an ABC with swappable implementations. The benchmark framework makes every design decision empirically testable.

## Module Structure

```
mcts_reasoning/
    __init__.py              # Public API re-exports
    types.py                 # Message, State, SearchState, Continuation, Evaluation,
                             #   SampledPath, ConsensusResult, Problem, SolverResult

    # --- Core engine (zero external dependencies) ---
    node.py                  # Node dataclass (UCB1, serialization, proper fields)
    mcts.py                  # MCTS (stateless: search() returns SearchState)
    generator.py             # Generator ABC + LLMGenerator
    evaluator.py             # Evaluator ABC + GroundTruth, Numeric, LLM, Process, Composite
    terminal.py              # TerminalDetector protocol + Marker, Boxed, MultiMarker
    prompt.py                # PromptStrategy ABC + StepByStepPrompt, FewShotPrompt
    sampling.py              # SamplingStrategy ABC + Value, Visits, Diverse, TopK
    consensus.py             # ConsensusStrategy ABC + MajorityVote, WeightedVote

    # --- Providers (one file per, optional dependencies) ---
    providers/
        __init__.py          # get_provider(), detect_provider(), LLMProvider re-export
        base.py              # LLMProvider ABC with detect() classmethod
        openai.py            # OpenAIProvider (requires openai extra)
        anthropic.py         # AnthropicProvider (requires anthropic extra)
        ollama.py            # OllamaProvider (requires requests extra)

    # --- Benchmarking framework ---
    bench/
        __init__.py          # BenchRunner, BenchReport re-exports
        benchmark.py         # Benchmark ABC, Problem dataclass
        solver.py            # Solver ABC, BaselineSolver, MCTSSolver
        runner.py            # BenchRunner
        report.py            # BenchReport (table, json, csv output)
        optimizer.py         # PromptOptimizer ABC + GridSearchOptimizer
        benchmarks/
            __init__.py      # Registry of built-in benchmarks
            knights.py       # KnightsAndKnaves (logic)
            arithmetic.py    # ArithmeticChains (math)

    # --- User-facing surfaces (thin wiring) ---
    server/
        __init__.py          # FastMCP server
        tools.py             # mcts_search, mcts_explore, mcts_bench

    cli.py                   # CLI with feature parity to MCP server

    # --- Testing utilities (not auto-selected, import explicitly) ---
    testing/
        __init__.py          # MockLLMProvider, MockGenerator, MockEvaluator
```

## Deleted from v0.5

| File/Directory | Lines | Reason |
|----------------|-------|--------|
| `config.py` | 287 | Unused. Libraries accept config, they don't manage it. |
| `llm_provider.py` | 75 | Vestigial adapter. LLMProvider ABC replaces it. |
| `actions.py` | 362 | ActionSpace/ContinueAction inlined into MCTS rollout. |
| `compositional/__init__.py` enums | 90 | 5 enums (53 members) that never change behavior. |
| `compositional/rag.py` | ~300 | Speculative. CompositionalRAGStore consumes enums that flow nowhere. |
| `compositional/examples.py` | 346 | Only used by speculative RAG. |
| `tools/` (entire directory) | 2,376 | Demoted to future `mcts-reasoning-tools` package. |
| `LLMJudgeTerminalDetector` | 68 | Dead code under "EXTENSIONS (for future consideration)". |
| `CompressAction`/`ExtendedActionSpace` | 133 | Speculative extension, never used. |

Total removed: ~4,037 source lines, ~2,800 test lines.

## ABCs

Every strategic decision point in the MCTS loop is an ABC. Each has a minimal interface.

### LLMProvider

```python
# providers/base.py
class LLMProvider(ABC):
    @abstractmethod
    def generate(self, messages: list[Message], max_tokens: int = 1000,
                 temperature: float = 0.7) -> str:
        """Send messages to the LLM and return the response text."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this provider is configured and reachable."""

    @classmethod
    def detect(cls, base_url: str | None = None) -> bool:
        """Return True if this provider can serve at the given URL or from env vars.
        Best-effort, may make network calls."""
        return False

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable provider name (e.g., 'Ollama-llama3.2')."""
```

Note: `generate()` takes `list[Message]` not a raw string. Providers that need raw strings flatten internally.

### PromptStrategy

```python
# prompt.py
class PromptStrategy(ABC):
    @abstractmethod
    def format(self, question: str, state: State, n: int = 1) -> list[Message]:
        """Build messages to send to the LLM.
        n=1: ask for one continuation.
        n>1: ask for N diverse continuations."""

    @abstractmethod
    def parse(self, response: str, n: int = 1) -> list[str]:
        """Extract continuations from the LLM response."""
```

Implementations:
- `StepByStepPrompt`: current default template ("continue reasoning step by step"). Accepts a `TerminalDetector` to include the right completion instruction.
- `FewShotPrompt`: includes examples selected from a pool. Used by the learnable prompt optimizer.

### Generator

```python
# generator.py
class Generator(ABC):
    @abstractmethod
    def generate(self, question: str, state: State, n: int = 1) -> list[Continuation]:
        """Generate n continuations from the current state."""

class LLMGenerator(Generator):
    def __init__(self, provider: LLMProvider, prompt_strategy: PromptStrategy,
                 terminal_detector: TerminalDetector): ...
```

LLMGenerator becomes thin: calls `prompt_strategy.format()`, sends to `provider.generate()`, calls `prompt_strategy.parse()`, checks `terminal_detector`. No hardcoded prompt templates.

### Evaluator

```python
# evaluator.py
class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, question: str, state: State, answer: str) -> Evaluation:
        """Score a terminal state. Returns Evaluation(score: float, explanation: str)."""
```

Implementations (same 5 from v0.5, minus dead LLMJudge):
- `GroundTruthEvaluator`: exact string comparison with normalization
- `NumericEvaluator`: math with tolerance (fix: `_parse_score` uses 0-1 range matching)
- `LLMEvaluator`: LLM-as-judge scoring
- `ProcessEvaluator`: reasoning quality heuristics
- `CompositeEvaluator`: weighted combination of evaluators

### TerminalDetector

```python
# terminal.py
class TerminalDetector(Protocol):
    def is_terminal(self, state: State) -> TerminalCheck: ...
    def format_instruction(self) -> str: ...
```

Implementations: `MarkerTerminalDetector`, `BoxedTerminalDetector`, `MultiMarkerTerminalDetector`.

### SamplingStrategy

```python
# sampling.py
class SamplingStrategy(ABC):
    @abstractmethod
    def sample(self, root: Node, n: int) -> list[SampledPath]:
        """Select n paths from the completed search tree."""
```

Implementations: `ValueSampling`, `VisitSampling`, `DiverseSampling`, `TopKSampling`.

### ConsensusStrategy

```python
# consensus.py
class ConsensusStrategy(ABC):
    @abstractmethod
    def vote(self, paths: list[SampledPath]) -> ConsensusResult:
        """Aggregate sampled paths into a final answer with confidence."""
```

Implementations: `MajorityVote`, `WeightedVote`.

```python
@dataclass
class ConsensusResult:
    answer: str
    confidence: float          # 0-1
    distribution: dict[str, int]  # answer -> count
    paths_used: int
```

## Stateless MCTS

```python
@dataclass
class SearchState:
    root: Node
    question: str
    terminal_states: list[dict]
    simulations_run: int

class MCTS:
    def __init__(self, generator: Generator, evaluator: Evaluator,
                 terminal_detector: TerminalDetector,
                 exploration_constant: float = 1.414,
                 max_children_per_node: int = 3,
                 max_rollout_depth: int = 5,
                 on_simulation: Callable | None = None):
        ...

    def search(self, question: str, simulations: int = 10) -> SearchState:
        """Run MCTS and return the search state. Does not mutate self."""

    def continue_search(self, state: SearchState, simulations: int = 10) -> SearchState:
        """Continue searching from a previous state. Returns new state."""
```

`on_simulation(sim_number: int, phase: str, node: Node, state: SearchState)` fires after each phase of each simulation. This is the observability hook for logging, visualization, progress reporting, and the explorable's data generation.

## State Type

```python
State = NewType("State", str)

def extend_state(state: State, continuation: str) -> State:
    """The single place where states grow by concatenation."""
    return State(f"{state}\n\n{continuation}")
```

All state concatenation goes through `extend_state`. One place to change the convention.

## Provider Auto-Detection

```python
# providers/__init__.py
_PROVIDER_CLASSES = [OllamaProvider, OpenAIProvider, AnthropicProvider]

def detect_provider(base_url: str | None = None, **kwargs) -> LLMProvider:
    """Probe base_url or env vars to find a working provider.
    Raises RuntimeError if none found. Never returns a mock."""
    if base_url:
        for cls in _PROVIDER_CLASSES:
            if cls.detect(base_url):
                return cls(base_url=base_url, **kwargs)

    for cls in _PROVIDER_CLASSES:
        if cls.detect():
            return cls(**kwargs)

    raise RuntimeError(
        "No LLM provider available. "
        "Set OPENAI_API_KEY, ANTHROPIC_API_KEY, or start Ollama."
    )

def get_provider(name: str, **kwargs) -> LLMProvider:
    """Create a provider by explicit name. No auto-detection."""
    registry = {"openai": OpenAIProvider, "anthropic": AnthropicProvider, "ollama": OllamaProvider}
    if name not in registry:
        raise ValueError(f"Unknown provider: {name}. Choose from: {list(registry.keys())}")
    return registry[name](**kwargs)
```

Each provider's `detect()` classmethod:
- `OllamaProvider.detect(base_url)`: GET base_url, check for "Ollama is running". Without base_url: check `OLLAMA_BASE_URL` env or probe localhost:11434.
- `OpenAIProvider.detect()`: check `OPENAI_API_KEY` env var.
- `AnthropicProvider.detect()`: check `ANTHROPIC_API_KEY` env var.

## Benchmarking Framework

### Problem and Benchmark

```python
@dataclass
class Problem:
    question: str
    ground_truth: str
    domain: str              # "math", "logic", "coding"
    difficulty: str          # "easy", "medium", "hard"
    metadata: dict           # extra info

class Benchmark(ABC):
    @abstractmethod
    def problems(self) -> list[Problem]: ...

    @property
    @abstractmethod
    def name(self) -> str: ...
```

### Solver

```python
class Solver(ABC):
    @abstractmethod
    def solve(self, problem: Problem) -> SolverResult: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

@dataclass
class SolverResult:
    answer: str | None
    correct: bool
    score: float
    time_seconds: float
    metadata: dict           # solver-specific (simulations, tree_size, etc.)

class BaselineSolver(Solver):
    """Single-pass LLM call. The control group."""
    def __init__(self, provider: LLMProvider, prompt_strategy: PromptStrategy): ...

class MCTSSolver(Solver):
    """Full MCTS search. The experimental group."""
    def __init__(self, provider: LLMProvider, prompt_strategy: PromptStrategy,
                 evaluator: Evaluator, terminal_detector: TerminalDetector,
                 simulations: int = 10, exploration_constant: float = 1.414): ...
```

### BenchRunner and Report

```python
class BenchRunner:
    def run(self, benchmark: Benchmark, solvers: list[Solver]) -> BenchReport: ...

@dataclass
class BenchReport:
    benchmark_name: str
    results: dict[str, list[SolverResult]]  # solver_name -> results

    def accuracy(self, solver: str) -> float: ...
    def accuracy_by_domain(self, solver: str) -> dict[str, float]: ...
    def accuracy_by_difficulty(self, solver: str) -> dict[str, float]: ...
    def lift(self, baseline: str, experimental: str) -> float: ...
    def to_table(self) -> str: ...
    def to_json(self) -> dict: ...
    def to_csv(self, path: str): ...
```

### synthdata Integration

The `Benchmark` ABC accepts any problem source. The [synthdata](https://github.com/spinoza/synthdata) package generates unlimited problems with guaranteed-correct solutions via symbolic term rewriting (algebra, calculus, number theory, propositional logic). A `SynthdataBenchmark` adapter wraps synthdata output:

```python
class SynthdataBenchmark(Benchmark):
    """Wrap synthdata JSONL output as a Benchmark.
    Each synthdata example has a problem, solution, reasoning trace, and difficulty score."""
    def __init__(self, path: str, domain: str = "algebra", max_problems: int = 100): ...
```

This gives us arbitrarily large, difficulty-controlled benchmark sets with verified ground truth. No manually curated problem sets needed beyond the small built-in ones (knights, arithmetic).

### Prompt Optimizer

```python
class PromptOptimizer(ABC):
    @abstractmethod
    def optimize(self, base_strategy: PromptStrategy, benchmark: Benchmark,
                 solver_factory: Callable[[PromptStrategy], Solver],
                 budget: int) -> PromptStrategy:
        """Search for a better prompt strategy within the given budget."""

class GridSearchOptimizer(PromptOptimizer):
    """Exhaustively try combinations from a parameter grid."""
    def __init__(self, grid: dict[str, list]): ...
```

Fancier optimizers (BanditOptimizer, EvolutionaryOptimizer, LLMGuidedOptimizer) are future work enabled by the ABC.

## MCP Server (Primary Interface)

```python
# server/tools.py (exposed as MCP tools)

@tool
def mcts_search(question: str, provider: str = "auto", model: str = None,
                simulations: int = 10, exploration_constant: float = 1.414) -> dict:
    """Run MCTS search on a question. Returns best answer, confidence, and tree stats."""

@tool
def mcts_explore(question: str, provider: str = "auto", model: str = None,
                 simulations: int = 10) -> dict:
    """Run MCTS and return the full tree structure for inspection."""

@tool
def mcts_bench(benchmark: str = "knights", provider: str = "auto",
               model: str = None, simulations: list[int] = [10]) -> dict:
    """Run benchmark comparison: baseline vs MCTS. Returns accuracy table."""
```

The server is a thin layer over the engine. All logic lives in the engine ABCs.

## CLI (Secondary, Feature Parity)

```bash
# Search
mcts-reason search "question" --provider ollama --model llama3.2 --simulations 20

# Explore (full tree output)
mcts-reason explore "question" --provider ollama --json

# Benchmark
mcts-reason bench --benchmark knights --provider ollama --simulations 10,20,50
mcts-reason bench --benchmark arithmetic --format csv --output results.csv

# Grid search optimization
mcts-reason optimize --benchmark knights --provider ollama \
    --simulations 10,20 --exploration 0.5,1.414,2.0
```

Same capabilities as MCP, different surface.

## Bug Fixes Included in v0.6

1. **`_continuation_info`**: index-based parallel list, not text-keyed dict
2. **Rollout `max_children_per_node`**: enforced during rollout, not just expansion
3. **`LLMEvaluator._parse_score`**: prefer numbers in 0-1 range, not first number found
4. **Test fixtures**: use `value` field, not nonexistent `_total_value`
5. **`_continuation_info`**: proper `Optional[dict]` field on Node, not monkey-patched
6. **Tree-walking**: iterative (stack-based DFS) throughout, no recursion limit risk
7. **Prompt template**: format at generate-time, not pre-format at construction

## Dependencies

```toml
[project]
dependencies = []  # Zero-dependency core

[project.optional-dependencies]
ollama = ["requests>=2.28.0"]
openai = ["openai>=1.0.0"]
anthropic = ["anthropic>=0.18.0"]
server = ["mcp>=1.0.0"]
all = ["requests>=2.28.0", "openai>=1.0.0", "anthropic>=0.18.0", "mcp>=1.0.0"]
dev = ["pytest", "pytest-cov", "pytest-asyncio", "black", "ruff", "mypy"]
```

Migrate from `setup.py` to `pyproject.toml`.

## Testing Strategy

- `testing/` module provides MockLLMProvider, MockGenerator, MockEvaluator for consumers
- Core engine tests use mocks only (zero network calls)
- Provider tests are integration tests gated by `@pytest.mark.integration`
- Benchmark framework tests use mock benchmarks and solvers
- CLI tests added (currently 0% coverage)
- MCP server tests use mock engine

## What This Is NOT (unchanged from v0.5)

- Not a general-purpose MCTS library (text states only)
- Not a prompt engineering framework (prompt optimization is a feature, not the identity)
- Not an LLM SDK wrapper (providers are minimal, just enough for MCTS)

## Future Directions (out of scope for v0.6)

- **mcts-reasoning-tools**: MCP tool integration as a separate package (the current tools/ directory)
- **BanditOptimizer**: use UCB1 (the library's own algorithm) to allocate benchmark budget across prompt strategies
- **EvolutionaryOptimizer**: mutate/crossover prompt configurations
- **LLMGuidedOptimizer**: show the LLM its failures and ask it to suggest better prompts (OPRO-style)
- **LearnableFewShotPrompt**: DSPy-style few-shot example selection with weight learning
- **GSM8K benchmark**: standard grade school math benchmark
- **HumanEval benchmark**: coding problems
- **Parallel simulations**: multi-threaded MCTS

## Success Criteria

1. `pytest --cov` shows >85% coverage on core engine
2. `mcts-reason bench --benchmark knights` produces a comparison table
3. MCP server exposes all three tools and works with Claude
4. Zero-dependency core: `pip install mcts-reasoning` pulls nothing
5. Library size: <4,000 source lines (down from 7,585)
