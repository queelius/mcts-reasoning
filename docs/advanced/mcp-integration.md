# MCP Integration

MCTS-Reasoning now supports **Model Context Protocol (MCP)** integration, allowing LLMs to automatically use external tools during reasoning.

## Overview

The MCP integration has two levels:

1. **Transparent Tool Access**: LLMs automatically have access to MCP tools and can call them during any generation
2. **Tool-Encouraging Actions**: MCTS actions that explicitly guide the LLM toward using specific tools

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      ReasoningMCTS                          │
│                                                             │
│  ┌───────────────────────────────────────────────────────┐ │
│  │              MCPLLMProvider                           │ │
│  │                                                       │ │
│  │  ┌────────────────┐        ┌──────────────────────┐ │ │
│  │  │ Base LLM       │◄───────┤  MCP Client          │ │ │
│  │  │ (GPT/Claude)   │        │                      │ │ │
│  │  └────────────────┘        │  ┌────────────────┐  │ │ │
│  │         ▲                  │  │ Python Server  │  │ │ │
│  │         │                  │  ├────────────────┤  │ │ │
│  │         │ Tool Results     │  │ Web Search     │  │ │ │
│  │         │                  │  ├────────────────┤  │ │ │
│  │         └──────────────────┤  │ Filesystem     │  │ │ │
│  │                            │  └────────────────┘  │ │ │
│  └────────────────────────────┴──────────────────────┘ │ │
│                                                         │ │
│  ┌──────────────────────────────────────────────────┐  │ │
│  │         MCPActionSelector                        │  │ │
│  │                                                  │  │ │
│  │  • Generates MCP-aware actions (~40% default)   │  │ │
│  │  • Matches tools to reasoning needs             │  │ │
│  │  • Creates tool-encouraging prompts             │  │ │
│  └──────────────────────────────────────────────────┘  │ │
└─────────────────────────────────────────────────────────┘
```

## Quick Start

### Basic Usage

```python
from mcts_reasoning import ReasoningMCTS, get_llm
from mcts_reasoning import create_mcp_client, create_mcp_provider

# Create MCP client with Python execution
mcp_client = create_mcp_client({
    "python": {"type": "python"},
    "web": {"type": "web"}
})

# Wrap base LLM with MCP awareness
base_llm = get_llm("openai", model="gpt-4")
mcp_llm = create_mcp_provider(base_llm, mcp_client=mcp_client)

# Use in MCTS - tools are automatically available
mcts = (
    ReasoningMCTS()
    .with_llm(mcp_llm)
    .with_question("Find all primes less than 100")
    .with_compositional_actions(enabled=True)
)

mcts.search("Let's solve this:", simulations=50)
```

The LLM can now automatically call `execute_python` or `search_web` during reasoning!

### Tool-Encouraging Actions

For explicit tool guidance, use MCP-aware actions:

```python
from mcts_reasoning import (
    MCPActionSelector,
    create_code_execution_action,
    create_research_action
)

# Create action that encourages Python execution
code_action = create_code_execution_action()

# Or use the MCP action selector in MCTS
# (Will automatically generate ~40% MCP-aware actions)
selector = MCPActionSelector(mcp_client=mcp_client)
```

## Available Tool Types

Built-in tool types:

- `PYTHON_EVAL` - Execute Python code
- `WEB_SEARCH` - Search the web
- `READ_FILE` - Read files
- `WRITE_FILE` - Write files
- `FETCH_URL` - Fetch URLs
- `BASH_COMMAND` - Run bash commands
- `DATABASE_QUERY` - Query databases
- `CUSTOM` - Custom tools

## Custom Tool Handlers

Register custom implementations:

```python
def custom_python_executor(arguments):
    code = arguments["code"]
    # Your sandboxed execution
    result = safe_eval(code)
    return result

# Register handler
mcp_client.register_tool_handler("execute_python", custom_python_executor)
```

## MCP Action Intents

Actions can explicitly encourage tool usage:

- `EXECUTE_CODE` - Run Python code
- `TEST_SOLUTION` - Test with code
- `CALCULATE` - Numerical calculations
- `RESEARCH` - Web research
- `VERIFY_FACTS` - Fact checking
- `READ_DATA` - Load data files
- `WRITE_RESULTS` - Save results
- `NONE` - No specific intent

## Tool Call Format

LLMs can call tools using XML format:

```xml
<tool_call>
{
  "tool": "execute_python",
  "arguments": {
    "code": "print(sum(range(1, 101)))"
  }
}
</tool_call>
```

Or function call format:

```python
execute_python(code="print(sum(range(1, 101)))")
```

Results are automatically appended to the reasoning context.

## TUI Commands (Planned)

The TUI will support MCP management:

```bash
/mcp-connect <server> <config>  # Connect to MCP server
/mcp-list                       # List connected servers
/mcp-tools                      # Show available tools
/mcp-disconnect <server>        # Disconnect from server
/mcp-test <tool> <args>        # Test a tool
```

## Integration with Real MCP Servers

To use real MCP servers (not mocks):

1. **Install MCP SDK** (when available)
2. **Configure servers** in your environment
3. **Connect programmatically**:

```python
from mcp import MCPServer  # Future official SDK

# Connect to real server
server = MCPServer.connect("localhost:9000")

# Register with client
mcp_client = MCPClient()
mcp_client.connect_server("my_server", server)
```

## Best Practices

1. **Sandbox Execution**: Always sandbox Python/bash execution
2. **Timeouts**: Set reasonable timeouts for tool calls
3. **Error Handling**: Tools can fail - handle gracefully
4. **Context Management**: Tool results can be large - truncate if needed
5. **Security**: Validate tool arguments before execution

## Examples

See `examples/mcp_demo.py` for complete examples:

- Transparent tool access
- Explicit MCP actions
- MCP action selector
- Custom tool handlers
- Full MCTS with MCP

## Future Enhancements

- [ ] Real MCP SDK integration
- [ ] Async tool execution
- [ ] Tool result caching
- [ ] Cost tracking for tool calls
- [ ] Tool chain composition
- [ ] Multi-turn tool conversations
- [ ] TUI command interface
