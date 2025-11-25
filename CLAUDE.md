# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

MCTS-Reasoning is a Monte Carlo Tree Search implementation for LLM-based reasoning with advanced compositional prompting and MCP (Model Context Protocol) integration. It combines systematic tree search with sophisticated prompt engineering to enable structured, exploratory reasoning across multiple solution paths.

## Core Development Commands

### Installation & Setup
```bash
pip install -e .                    # Basic installation
pip install -e ".[tui]"            # With TUI support (recommended)
pip install -e ".[all]"            # TUI + all LLM providers
pip install -e ".[dev]"            # Development dependencies
```

### Testing
```bash
pytest tests/                                        # Run all tests
pytest tests/test_<module>.py                       # Run specific test file
pytest tests/test_core.py::test_mcts_search        # Run specific test
pytest --cov=mcts_reasoning tests/                  # With coverage
pytest --cov=mcts_reasoning --cov-report=html tests/  # HTML coverage report
```

### Running the Interfaces
```bash
# Interactive shell (TUI)
mcts-shell              # Primary command
python mcts_tui.py      # Direct execution

# Non-interactive CLI
mcts ask "What is 2+2?" --search 50
mcts solution --session my_session.json
mcts export json output.json --session my_session.json
```

### Code Quality
```bash
black mcts_reasoning/    # Format
flake8 mcts_reasoning/   # Lint
mypy mcts_reasoning/     # Type check
```

## Architecture

### Core Module Structure

```
mcts_reasoning/
├── core.py               # Pure MCTS: MCTSNode, MCTS with UCB1, serialization
├── reasoning.py          # ReasoningMCTS: LLM integration, all features unified
├── sampling.py           # MCTSSampler, SampledPath, sampling strategies
├── config.py             # Persistent configuration (~/.mcts-reasoning/config.json)
├── cli.py                # Non-interactive CLI (mcts command)
│
├── compositional/        # 5D compositional prompting system
│   ├── __init__.py      # Enums (ω,φ,σ,κ,τ), ComposingPrompt, smart_termination
│   ├── providers.py     # LLMProvider interface + OpenAI/Anthropic/Ollama/Mock
│   ├── actions.py       # CompositionalAction, ActionSelector
│   ├── rag.py           # CompositionalRAGStore, SolutionRAGStore
│   ├── examples.py      # Example, ExampleSet for few-shot learning
│   ├── mcp.py           # MCPClient, MCPLLMProvider, tool integration
│   └── mcp_actions.py   # MCPActionSelector, tool-encouraging actions
│
├── solution_detection.py # SolutionDetector, SolutionFinalizer (LLM-as-judge)
├── context_manager.py    # ContextManager, auto-summarization for long contexts
├── learning.py           # PathLearner: learn from successful paths
├── meta_reasoning.py     # MetaReasoner: LLM suggests next action
├── reflection.py         # ReflectionCritic: self-critique and refinement
├── benchmarking.py       # BenchmarkRunner for standard dataset evaluation
│
├── tui/                  # Interactive terminal interface
│   ├── app.py           # Main TUI application
│   ├── session.py       # SessionState management
│   ├── commands.py      # Command handlers (40+ commands)
│   └── prompt.py        # History, completion, syntax highlighting
│
└── shell/                # Deprecated Unix-style shell (internal only)
```

### Key Design Patterns

**Fluent API**: All major components support method chaining:
```python
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question("...")
    .with_exploration(1.414)
    .with_compositional_actions(enabled=True)
    .with_max_rollout_depth(5)
    .with_solution_detection(enabled=True, threshold=0.7)
    .with_context_management(max_tokens=8000)
    .with_meta_reasoning(enabled=True)
    .with_reflection(enabled=True, threshold=0.6)
    .with_learning(enabled=True)
)
```

**Provider Abstraction**: LLM providers implement `LLMProvider` interface with `generate()` method. Use `get_llm()` for auto-detection or explicit provider selection.

**5D Compositional Actions**: 30,000+ action combinations from:
- ω: CognitiveOperation (decompose, analyze, synthesize, verify, etc.)
- φ: FocusAspect (structure, patterns, assumptions, correctness, etc.)
- σ: ReasoningStyle (systematic, intuitive, critical, creative, etc.)
- κ: ConnectionType (therefore, however, building_on, etc.)
- τ: OutputFormat (steps, list, mathematical, code, etc.)

**Sampling Strategies**: `MCTSSampler` provides `value`, `visits`, `diverse`, `topk` strategies.

## Environment Variables

```bash
export LLM_PROVIDER=openai|anthropic|ollama|mock
export OPENAI_API_KEY=your-key
export ANTHROPIC_API_KEY=your-key
export OLLAMA_BASE_URL=http://localhost:11434
export OLLAMA_MODEL=llama2
```

## TUI Commands Reference

Commands work with or without leading `/`:

**Model Management:**
- `model <provider> [model] [key=value...]` - Switch provider/model
- `models` - List available models
- `model-info` - Show current model info
- `probe <url>` - Probe endpoint for models (Ollama)
- `temperature <value>` / `temp <value>` - Set temperature
- `exploration <value>` - Set UCB1 exploration constant

**Core Reasoning:**
- `ask <question>` - Start reasoning session
- `search <n>` / `continue <n>` - Run N simulations
- `solution` - Show best solution
- `tree` - Visualize search tree

**Sampling & Validation:**
- `sample <n>` - Sample N diverse paths
- `consistency` - Check solution consistency
- `verify` - Verify solution correctness

**Tree Diagnostics:**
- `nodes` - List all nodes with indices
- `inspect <index>` - Show node details (truncated)
- `inspect-full <index>` - Show full node state
- `show-prompt <index>` - Show exact LLM prompt
- `path <index>` - Show reasoning path to node
- `compare <i1> <i2> ...` - Compare multiple nodes
- `export-tree <file>` - Export tree to JSON

**Advanced Features:**
- `stats` - Show context/solution detection stats
- `solutions` - List all finalized solutions
- `config [feature] [on|off]` - Configure features

**Session Management:**
- `save [filename]` - Save session
- `load <filename>` - Load session
- `status` - Show current status
- `export <format> <file>` - Export (json/markdown/dot)

**MCP Tools:**
- `mcp-enable` - Enable MCP integration
- `mcp-connect <tool>` - Connect to MCP tool
- `mcp-list` - List connected tools
- `mcp-tools` - Show available tools

## Key Implementation Details

### MCTS Algorithm Flow
1. **Selection** (`_select()`): Traverse tree using UCB1 to find promising leaf
2. **Expansion** (`_expand()`): Add new child node with sampled action
3. **Rollout** (`_rollout()`): Simulate to terminal or max depth
4. **Backpropagation** (`_backpropagate()`): Update visits/values up the tree

### ReasoningMCTS Extensions
`ReasoningMCTS` extends `MCTS` and overrides:
- `_get_actions()`: Returns compositional or simple actions
- `_take_action()`: Uses LLM to apply action, generate new state
- `_is_terminal_state()`: Smart termination (pattern + LLM-based)
- `_evaluate_state()`: Uses LLM to score state quality (0-1)

### State Management
- States are accumulated reasoning strings
- Terminal detection: completion keywords OR LLM assessment
- Context truncation (last 1500 chars) prevents overflow
- ContextManager triggers summarization when context exceeds threshold

### JSON Serialization
```python
# Save/load trees
mcts.save("tree.json")
mcts = ReasoningMCTS.load("tree.json", llm=llm)

# Or via to_json/from_json for more control
tree_data = mcts.to_json()
mcts = MCTS.from_json(tree_data)
```

## Common Development Patterns

### Adding a New LLM Provider
1. Implement `LLMProvider` interface in `compositional/providers.py`
2. Add to `get_llm()` factory function
3. Update `Config.DEFAULT_CONFIG` with provider defaults
4. Optionally implement `probe_endpoint()`, `list_models()`, `get_model_info()`

### Adding New Compositional Dimensions
1. Add enum values to appropriate class in `compositional/__init__.py`
2. Update prompt templates in `ComposingPrompt._build_template()`
3. Consider updating action sampling weights in `ActionSelector`

### Extending MCTS Behavior
- Override `_get_actions()` for custom action spaces
- Override `_is_terminal()` for custom termination logic
- Override `_apply_action()` for custom state transitions

### Using RAG for Guided Action Selection
```python
from mcts_reasoning.compositional.rag import get_math_compositional_rag

rag_store = get_math_compositional_rag()
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_compositional_actions(enabled=True)
    .with_rag_store(rag_store)
)
```

## Testing Strategy

- Use `MockLLMProvider` for deterministic LLM testing
- Test both isolated units and integration flows
- Tests in `tests/` organized by feature: `test_core.py`, `test_sampling.py`, `test_rag.py`, etc.
- Run coverage to identify gaps: `pytest --cov=mcts_reasoning tests/`

## Debugging Tips

### Tree Structure
```python
print(mcts.stats)  # Dict with nodes, depth, values
print(mcts.explain_reasoning())  # Human-readable explanation
```

### LLM Provider Issues
- Missing LLM: Use `.with_llm()` before `.search()`
- Auto-detection fails: Explicitly specify `get_llm("openai", model="gpt-4")`
- Ollama: Verify with `curl http://localhost:11434/api/tags`

### Empty Sampling Results
- Ensure `.search()` was called first
- Check `mcts.root.children` has nodes
- Verify simulations > 0

### Terminal States Never Reached
- Increase `max_rollout_depth` (default: 5)
- Disable LLM-based termination: `.with_terminal_detection(use_llm=False)`

### Poor Solution Quality
- Increase simulations (100-500 for complex problems)
- Enable compositional actions: `.with_compositional_actions(enabled=True)`
- Use consistency checking: `mcts.check_consistency(n_samples=20)`
