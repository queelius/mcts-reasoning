# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCTS-Reasoning is a canonical Monte Carlo Tree Search implementation for LLM-based step-by-step reasoning. It explores multiple reasoning paths to find high-quality solutions.

**Version**: 0.5.0

**Core Value Proposition:**
- LLMs often make reasoning errors in a single pass
- MCTS explores multiple paths, backtracking from poor ones
- Tree structure preserves reasoning traces for inspection
- UCB1 balances exploration (new paths) vs exploitation (promising paths)

## Current Scope

| Capability | Status |
|------------|--------|
| Single-LLM reasoning | Full |
| Multi-step math/logic | Full |
| Tree serialization/resume | Full |
| Self-consistency voting | Full |
| Path sampling | Full (4 strategies) |
| Custom evaluators | Full (6 types) |
| Multi-provider support | Full (OpenAI, Anthropic, Ollama) |
| Custom action spaces | Partial (CONTINUE only by default) |
| Tool use (MCP) | Full (ToolAwareGenerator + MCP servers) |
| RAG-guided reasoning | Full (SolutionRAGStore, CompositionalRAGStore, MCP server) |

## Known Limitations

- **Single-threaded**: No parallel simulations
- **Text-only states**: Designed for LLM reasoning, not general MCTS
- **Unbounded state growth**: States accumulate text (use ExtendedActionSpace with CompressAction for long reasoning)
- **Terminal detection**: Relies on "ANSWER:" marker (configurable via TerminalDetector)
- **Action selection in rollout**: Uses random selection, not UCB1
- **No policy learning**: UCB1 is fixed, doesn't learn from experience
- **No multi-model ensemble**: Single generator throughout search

## Core Development Commands

### Installation & Setup
```bash
pip install -e .                    # Basic installation
pip install -e ".[openai]"         # With OpenAI support
pip install -e ".[anthropic]"      # With Anthropic support
pip install -e ".[tools]"          # With MCP tool support
pip install -e ".[all]"            # All providers + tools
pip install -e ".[dev]"            # Development dependencies
```

### Testing
```bash
pytest tests/                                        # Run all tests
pytest tests/test_v2.py -v                          # Core MCTS tests (verbose)
pytest tests/test_sampling.py::TestPathSampler -v   # Specific test class
pytest tests/test_v2.py::test_mcts_search -v        # Single test function
pytest -k "serialization" -v                        # Tests matching pattern
pytest --cov=mcts_reasoning tests/                  # With coverage
pytest --cov=mcts_reasoning --cov-report=html tests/  # HTML coverage report
```

### CLI Usage
```bash
mcts-reason "What is 2+2?"                          # Simple question
mcts-reason "What is 15*7+23?" --answer 128         # With ground truth
mcts-reason "Question" --provider ollama --model llama3.2  # Specific provider
mcts-reason "Question" --simulations 20 -v          # Verbose output
mcts-reason "Question" --json                       # JSON output
mcts-reason "Question" --consistency --sample 5    # With analysis
mcts-reason "Hard problem" --simulations 50 --save tree.json  # Save tree
mcts-reason "Hard problem" --load tree.json --simulations 50  # Continue search
mcts-reason "What is 5*6?" --simulations 20 --vote majority   # Self-consistency
mcts-reason "Question" --sample 5 --sample-strategy diverse   # Specific sampling
```

### Code Quality
```bash
black mcts_reasoning/    # Format
flake8 mcts_reasoning/   # Lint
mypy mcts_reasoning/     # Type check
```

## Architecture

### Module Structure

```
mcts_reasoning/
├── node.py              # Node: UCB1, tree structure, path traversal
├── mcts.py              # MCTS: Selection, Expansion, Rollout, Backpropagation
├── generator.py         # Generator: LLM continuation generation
├── evaluator.py         # Evaluator: Terminal state scoring (6 implementations)
├── terminal.py          # TerminalDetector: Answer detection (marker, boxed, multi)
├── actions.py           # Action/ActionSpace: ContinueAction (default), CompressAction (extension)
├── sampling.py          # PathSampler: Extract paths from tree (value/visits/diverse/topk)
├── cli.py               # CLI: mcts-reason command
├── llm_provider.py      # Adapter: Create Generator/Evaluator from LLMProvider
├── config.py            # Config: Persistent settings storage
│
├── compositional/       # LLM Provider system + RAG
│   ├── providers.py     # LLMProvider: OpenAI, Anthropic, Ollama, Mock
│   ├── __init__.py      # Compositional action enums + re-exports
│   ├── examples.py      # Few-shot example management
│   └── rag.py           # RAG stores (SolutionRAGStore, CompositionalRAGStore)
│
├── tools/               # MCP Tool Integration (NEW)
│   ├── __init__.py      # Public API: ToolContext, ToolAwareGenerator
│   ├── context.py       # ToolContext: high-level interface
│   ├── generator.py     # ToolAwareGenerator: wraps Generator with tools
│   ├── client.py        # MCPClientManager: MCP server connections
│   ├── registry.py      # MCPToolRegistry: tool discovery
│   ├── execution.py     # ToolCallParser, ToolCallHandler
│   ├── formats.py       # ToolFormat, ToolDefinition, ToolCall, ToolResult
│   └── rag_server.py    # RAG as MCP server (FastMCP)
│
tests/
├── test_v2.py           # Core MCTS, Node, Generator, Evaluator tests
├── test_actions.py      # Action and ActionSpace tests
├── test_terminal.py     # Terminal detection tests
├── test_sampling.py     # Sampling strategy tests
├── test_serialization.py # Tree save/load and continue_search tests
├── test_examples.py     # Few-shot examples tests
├── test_rag.py          # RAG store tests
├── test_tools.py        # MCP tools integration tests (NEW)
├── test_rag_server.py   # RAG MCP server tests (NEW)
├── test_probe.py        # Provider probing tests
├── test_remote_ollama.py # Remote Ollama integration tests
│
examples/
└── run_mcts.py          # Example: Run MCTS with real LLM
```

## Core Components

### MCTS Search
```python
from mcts_reasoning import MCTS, LLMGenerator, NumericEvaluator, get_llm

llm = get_llm("ollama", model="llama3.2")
generator = LLMGenerator(llm=llm, temperature=0.7)
evaluator = NumericEvaluator(ground_truth=128)

mcts = MCTS(
    generator=generator,
    evaluator=evaluator,
    exploration_constant=1.414,  # UCB1 exploration
    max_children_per_node=3,     # Branching factor
    max_rollout_depth=5,         # Max reasoning steps
)

result = mcts.search(question="What is 15*7+23?", simulations=20)
print(f"Answer: {result.best_answer}")
print(f"Confidence: {result.confidence:.1%}")
```

### Evaluators
- **LLMEvaluator**: LLM-as-judge scoring (0-1)
- **GroundTruthEvaluator**: Compare to known answer with normalization
- **NumericEvaluator**: Math with tolerance (fractions, sci notation, partial credit)
- **ProcessEvaluator**: Reasoning quality heuristics (steps, logic, verification)
- **CompositeEvaluator**: Weighted combination of evaluators
- **MockEvaluator**: For testing

### Terminal Detection
- **MarkerTerminalDetector**: Look for "ANSWER:" marker
- **BoxedTerminalDetector**: Look for \boxed{} (math benchmarks)
- **MultiMarkerTerminalDetector**: Multiple completion markers

### Sampling Strategies
```python
from mcts_reasoning import PathSampler

sampler = PathSampler(result.root)
paths = sampler.sample(n=5, strategy="diverse")  # value, visits, diverse, topk
distribution = sampler.get_answer_distribution()
consistency = sampler.consistency_score()
```

### Self-Consistency Voting
```python
sampler = PathSampler(result.root)
answer, confidence = sampler.majority_vote()     # Simple majority
answer, confidence = sampler.weighted_vote()     # Value-weighted
vote_result = sampler.self_consistency_vote(weighted=True)  # Full details
```

### Tree Serialization
```python
# Save tree after search
result = mcts.search("What is 2+2?", simulations=50)
mcts.save("tree.json")

# Load and continue searching
mcts = MCTS.load("tree.json", generator, evaluator)
result = mcts.continue_search(simulations=50)
```

### Configuration
```python
from mcts_reasoning import Config, get_config

# Get global config (creates if needed)
config = get_config()

# Access settings
print(config.get("default_provider"))
config.set("default_provider", "ollama")
```

## MCTS Algorithm

1. **Selection**: Navigate tree using UCB1 to find promising leaf
2. **Expansion**: Generate new child via Generator
3. **Rollout**: Continue reasoning until terminal or max depth (tree-building)
4. **Backpropagation**: Update values up the tree

```
UCB1 = average_value + c * sqrt(ln(parent_visits) / visits)
```

Key design: **Tree-building rollouts** - rollout nodes are added to tree, not discarded.

## State Space

States are represented as **text strings** (`node.state: str`).

```python
# Initial state
initial_state = f"Question: {question}\n\nLet me solve this step by step."

# State transition (accumulation)
new_state = f"{old_state}\n\n{continuation}"
```

| Aspect | Status | Notes |
|--------|--------|-------|
| Well-defined | Yes | Strings with clear accumulation semantics |
| Markov property | Yes | Generator sees only current state |
| Provenance | Yes | Parent pointers reconstruct path |
| Bounded | No | States grow unboundedly |

## Action Space

The canonical implementation uses only `ContinueAction` (generate next reasoning step).

```python
from mcts_reasoning import DefaultActionSpace, ExtendedActionSpace

# Default: only CONTINUE action
action_space = DefaultActionSpace(generator=generator)

# Extended: CONTINUE + COMPRESS (for long reasoning chains)
action_space = ExtendedActionSpace(generator=generator, llm=llm, compress_threshold=2000)
```

## Environment Variables

```bash
export LLM_PROVIDER=ollama          # openai, anthropic, ollama, mock
export OLLAMA_BASE_URL=http://localhost:11434
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key
```

## Key Classes

### SearchResult
```python
@dataclass
class SearchResult:
    best_answer: Optional[str]      # Best answer found
    confidence: float               # Confidence score (0-1)
    root: Node                      # Tree root for further analysis
    simulations: int                # Number of simulations run
    terminal_states: List[Dict]     # All terminal states found

    @property
    def stats(self) -> Dict         # Tree statistics
```

### Node
```python
class Node:
    state: str                      # Reasoning state
    visits: int                     # Visit count
    children: List[Node]            # Child nodes
    is_terminal: bool               # Terminal state?
    answer: Optional[str]           # Extracted answer

    def ucb1(self, c: float) -> float           # UCB1 score
    def average_value(self) -> float            # Average value
    def path_from_root(self) -> List[Node]      # Path to this node
    def best_child(self, c: float) -> Node      # Best child by UCB1
```

## Testing Strategy

- Use `MockGenerator` and `MockEvaluator` for deterministic tests
- Tests organized by component: `test_v2.py`, `test_sampling.py`, `test_serialization.py`, etc.
- Run `pytest --cov=mcts_reasoning tests/` for coverage report

## Development Patterns

### Adding a New Evaluator
1. Inherit from `Evaluator` ABC in `evaluator.py`
2. Implement `evaluate(question, state, answer) -> Evaluation`
3. Add to exports in `__init__.py`
4. Add tests in `test_v2.py`

### Adding a New LLM Provider
1. Implement `LLMProvider` interface in `compositional/providers.py`
2. Add to `get_llm()` factory function
3. Test with `MockLLMProvider` pattern

### Extending MCTS
- Custom actions: Implement `Action` protocol, use custom `ActionSpace`
- Custom terminal detection: Implement `TerminalDetector` protocol
- Custom evaluation: Implement `Evaluator` ABC

## Debugging Tips

### No Terminal States Found
- Increase `max_rollout_depth`
- Check terminal detector configuration
- Verify LLM generates "ANSWER:" markers

### Low Confidence
- Increase simulations
- Use `ProcessEvaluator` for reasoning quality
- Check answer consistency with `PathSampler.consistency_score()`

### LLM Provider Issues
- Ollama: `curl http://localhost:11434/api/tags`
- Check API keys in environment
- Use `MockLLMProvider` for testing

## MCP Tool Integration

The `tools/` module provides MCP (Model Context Protocol) integration for tool-augmented reasoning.

### Quick Start
```python
from mcts_reasoning import MCTS, LLMGenerator, MockEvaluator, get_llm
from mcts_reasoning.tools import ToolContext, ToolAwareGenerator

# Create base generator
llm = get_llm("ollama", model="llama3.2")
base_gen = LLMGenerator(llm=llm)

# Add tool support
tool_context = ToolContext.from_servers({
    "rag": {"command": ["python", "-m", "mcts_reasoning.tools.rag_server"]},
})
tool_gen = ToolAwareGenerator(base_generator=base_gen, tool_context=tool_context)

# Use with MCTS
mcts = MCTS(generator=tool_gen, evaluator=MockEvaluator())
result = mcts.search("Solve x^2 + 5x + 6 = 0", simulations=20)
```

### Mock Tools for Testing
```python
from mcts_reasoning.tools import ToolContext, ToolAwareGenerator

# Create mock tools
context = ToolContext.mock({
    "calculator": {"description": "Calculate", "response": "42"},
    "search": {"description": "Search", "response": lambda args: f"Results for {args['query']}"},
})

tool_gen = ToolAwareGenerator(base_generator=base_gen, tool_context=context)
```

### RAG MCP Server
The built-in RAG server exposes retrieval tools:
```bash
# Start the server
python -m mcts_reasoning.tools.rag_server

# Tools provided:
# - retrieve_examples: Get similar solution examples
# - get_guidance: Get compositional guidance for problem type
# - get_recommended_weights: Get sampling weights for actions
# - add_example: Add new example to store
# - list_domains: List available domains
```

### Tool Format Support
- XML format: `<tool_call name="..."><param>value</param></tool_call>`
- JSON format: `{"tool": "...", "arguments": {...}}`
- Anthropic format: `{"name": "...", "input": {...}}`

### Native Function Calling
For OpenAI and Anthropic providers, native function calling APIs are supported for better reliability:
```python
from mcts_reasoning import LLMGenerator, get_llm
from mcts_reasoning.tools import ToolContext, create_native_tool_generator

# Create provider and generator
llm = get_llm("openai", model="gpt-4")
base_gen = LLMGenerator(llm=llm)
context = ToolContext.mock({"calc": {"response": "42"}})

# Create generator with native function calling
tool_gen = create_native_tool_generator(base_gen, context, llm)

# Or manually wrap with native provider
from mcts_reasoning.tools import OpenAINativeWrapper, ToolAwareGenerator
native = OpenAINativeWrapper(llm)
tool_gen = ToolAwareGenerator(base_gen, context, native_provider=native)
```

Native mode uses the provider's built-in tool APIs instead of parsing XML/JSON from text.
Falls back to parsing mode for unsupported providers (Ollama, Mock).

## RAG-Guided Reasoning

### SolutionRAGStore
Few-shot example retrieval for similar problems:
```python
from mcts_reasoning.compositional.rag import SolutionRAGStore

store = SolutionRAGStore()
store.add(problem="What is 2+2?", solution="4", reasoning_steps=["Add numbers"])

examples = store.retrieve("What is 3+3?", k=3)
prompt = store.to_few_shot_prompt("What is 5+5?", n_examples=2)
```

### CompositionalRAGStore
Maps problem types to recommended reasoning approaches:
```python
from mcts_reasoning.compositional.rag import CompositionalRAGStore, CompositionalGuidance
from mcts_reasoning.compositional import CognitiveOperation, ReasoningStyle

store = CompositionalRAGStore()
store.add_guidance(CompositionalGuidance(
    problem_pattern="quadratic equations",
    problem_keywords=["solve", "x^2", "equation"],
    recommended_operations=[CognitiveOperation.DECOMPOSE],
    recommended_styles=[ReasoningStyle.SYSTEMATIC],
    domain="math",
))

weights = store.get_recommended_weights("solve x^2 + 5x + 6 = 0")
```

## Compositional Action Dimensions

The compositional framework defines five dimensions for reasoning:
- **ω (Cognitive Operation)**: DECOMPOSE, ANALYZE, VERIFY, SYNTHESIZE, etc.
- **φ (Focus Aspect)**: STRUCTURE, DETAILS, CONSTRAINTS, PATTERNS, etc.
- **σ (Reasoning Style)**: SYSTEMATIC, FORMAL, CREATIVE, CRITICAL, etc.
- **κ (Connection Type)**: THEREFORE, HOWEVER, BUILDING_ON, etc.
- **τ (Output Format)**: STEPS, MATHEMATICAL, CODE, EXPLANATION, etc.
