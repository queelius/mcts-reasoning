# Changelog

All notable changes to MCTS-Reasoning will be documented in this file.

## [0.2.0] - 2024-10-08

### Major Features Added

#### 🧩 Compositional Prompting System
- **Merged compositional-prompting package** into mcts-reasoning
- Full 5-dimensional action space (ω, φ, σ, κ, τ) with 30,000+ combinations
- Cognitive Operations: decompose, analyze, synthesize, verify, abstract, etc.
- Focus Aspects: structure, patterns, assumptions, correctness, etc.
- Reasoning Styles: systematic, intuitive, critical, creative, formal, etc.
- Connection Types: therefore, however, building_on, etc.
- Output Formats: steps, list, mathematical, code, narrative, etc.
- Weighted action sampling for biased exploration
- Smart termination detection (pattern + LLM hybrid)

#### 🔌 MCP (Model Context Protocol) Integration
- **Transparent tool access**: LLMs automatically use external tools
- **Tool-encouraging actions**: MCTS actions that guide toward tool usage
- Built-in tool types: Python execution, web search, file ops, bash, etc.
- Custom tool handlers
- ~40% of actions have MCP intent by default
- Auto tool result incorporation into reasoning context

#### 🖥️ Interactive TUI (Text User Interface)
- **Claude Code-style interface** with rich formatting
- **Enhanced prompt system** powered by `prompt_toolkit`:
  - Tab completion for commands and arguments
  - Persistent command history across sessions
  - History search with Ctrl+R (reverse-i-search)
  - Syntax highlighting for commands
  - Emacs-style editing (Ctrl+A/E/K/U/W, etc.)
  - Navigate history with ↑/↓ arrows
  - Auto-completion while typing
- Stateful sessions with save/load
- Slash commands for all operations
- Commands:
  - Session: `/ask`, `/search`, `/solution`, `/save`, `/load`, `/status`
  - Config: `/model`, `/models`, `/model-info`, `/temperature`, `/exploration`
  - Analysis: `/tree`, `/sample`, `/consistency`
  - MCP: `/mcp-enable`, `/mcp-connect`, `/mcp-list`, `/mcp-tools`
- Tree visualization with statistics
- Session persistence to `~/.mcts-reasoning/sessions/`
- Command history in `~/.mcts-reasoning/history`
- Global configuration in `~/.mcts-reasoning/config.json`
- Recent models tracking

#### 🤖 Unified LLM Provider System
- Clean, unified LLMProvider interface for all LLMs
- Single consistent API across all providers
- Auto-detection from environment variables
- Support for: OpenAI, Anthropic, Ollama (local/remote), Mock
- Configuration system with `~/.mcts-reasoning/config.json`
- Remote Ollama support with `base_url` parameter
- Model discovery: `list_models()` and `get_model_info()`

### Breaking Changes
- Removed old `actions/` directory (replaced by compositional system)
- Removed IPC and HTML viewer (replaced by TUI)
- Version bump from 0.1.0 → 0.2.0

### Improvements
- Updated README with comprehensive examples
- Added detailed documentation:
  - `docs/TUI_GUIDE.md` - Complete TUI user guide
  - `docs/MCP_INTEGRATION.md` - MCP integration guide
- New examples:
  - `examples/compositional_demo.py` - Compositional prompting demo
  - `examples/mcp_demo.py` - MCP tool integration demo
- Updated `setup.py` with TUI entry points: `mcts-tui`, `mcts-reasoning-tui`
- Added extras: `[tui]`, `[mcp]` for optional dependencies

### Architecture Changes
```
mcts_reasoning/
├── compositional/         # NEW - Merged compositional prompting
│   ├── __init__.py       # Core enums + ComposingPrompt
│   ├── providers.py      # Unified LLM providers
│   ├── actions.py        # Compositional actions for MCTS
│   ├── mcp.py            # MCP client and tools
│   └── mcp_actions.py    # MCP-aware actions
└── tui/                   # NEW - Interactive interface
    ├── app.py            # Main TUI application
    ├── session.py        # Session state management
    └── commands.py       # Command parser and handlers
```

### API Changes

#### New Exports
```python
from mcts_reasoning import (
    # Compositional prompting
    CognitiveOperation, FocusAspect, ReasoningStyle,
    ConnectionType, OutputFormat,
    ComposingPrompt, smart_termination,
    CompositionalAction, ActionSelector,

    # MCP integration
    MCPToolType, MCPTool, MCPClient,
    MCPLLMProvider, create_mcp_client, create_mcp_provider,
    MCPActionIntent, MCPCompositionalAction, MCPActionSelector,
    create_mcp_action, create_code_execution_action,

    # TUI
    ReasoningTUI, run_tui,
)
```

#### Enhanced ReasoningMCTS
```python
mcts = (
    ReasoningMCTS()
    .with_llm(llm)
    .with_question(question)
    .with_compositional_actions(enabled=True)  # NEW
    .with_exploration(1.414)
)
```

### Removed
- `mcts_reasoning/actions/` directory
- `mcts_reasoning/mcts_with_ipc.py`
- `mcts_reasoning/ollama_integration.py`
- `viewer/` directory and HTML visualization

### Bug Fixes
- Fixed action selection for compositional actions
- Improved terminal detection with smart hybrid approach
- Better context management in state transitions

### Documentation
- Complete rewrite of README
- Added TUI user guide with examples
- Added MCP integration documentation
- Updated all code examples to use new API
- Added architecture diagrams

### Testing
- All examples updated and tested
- TUI tested with mock LLM
- MCP integration tested with demo
- Compositional prompting tested extensively

## [0.1.0] - 2024-09-01

### Initial Release
- Pure MCTS implementation
- Basic compositional actions
- LLM adapters (OpenAI, Anthropic, Ollama)
- IPC support for visualization
- HTML tree viewer
- JSON serialization
- Sampling strategies
