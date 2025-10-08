#!/usr/bin/env python3
"""
MCP (Model Context Protocol) Integration Demo

Demonstrates how MCTS reasoning can use MCP tools automatically:
- Python code execution
- Web search
- File operations
- Custom tools

The LLM automatically gets access to tools and can use them during reasoning.
MCTS actions can also explicitly encourage tool usage.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mcts_reasoning import ReasoningMCTS, MockLLMProvider
from mcts_reasoning.compositional import (
    create_mcp_client,
    create_mcp_provider,
    MCPActionSelector,
    MCPActionIntent,
    create_code_execution_action,
    create_research_action,
)


def demo_mcp_transparent():
    """Demo 1: Transparent MCP tool access (LLM uses tools automatically)"""
    print("=" * 60)
    print("Demo 1: Transparent MCP Tool Access")
    print("=" * 60)

    # Create MCP client with Python execution
    mcp_client = create_mcp_client({
        "python": {"type": "python"},
        "web": {"type": "web"}
    })

    # Show available tools
    print("\nAvailable MCP tools:")
    for tool_name, tool in mcp_client.tools.items():
        print(f"  - {tool_name}: {tool.description}")

    # Create base LLM
    base_llm = MockLLMProvider({
        "prime": "Let me test this with code. <tool_call>{\"tool\": \"execute_python\", \"arguments\": {\"code\": \"def is_prime(n):\\n    if n < 2: return False\\n    for i in range(2, int(n**0.5)+1):\\n        if n % i == 0: return False\\n    return True\\n\\nprimes = [n for n in range(2, 20) if is_prime(n)]\\nprint(primes)\"}}</tool_call>",
        "terminal": "YES",
        "quality": "0.90"
    })

    # Wrap with MCP awareness
    mcp_llm = create_mcp_provider(base_llm, mcp_client=mcp_client)

    print(f"\nLLM Provider: {mcp_llm.get_provider_name()}")
    print("The LLM will automatically have access to MCP tools during generation.")

    # Use in MCTS
    mcts = (
        ReasoningMCTS()
        .with_llm(mcp_llm)
        .with_question("Find all prime numbers less than 20")
        .with_exploration(1.414)
        .with_compositional_actions(enabled=False)  # Simple actions for now
    )

    print("\nRunning MCTS search...")
    mcts.search("Let's find the primes:", simulations=10)

    print("\n" + "=" * 40)
    print("RESULT")
    print("=" * 40)
    solution = mcts.solution
    print(solution[-500:] if len(solution) > 500 else solution)


def demo_mcp_explicit_actions():
    """Demo 2: Explicit MCP-encouraging actions"""
    print("\n\n" + "=" * 60)
    print("Demo 2: Explicit MCP-Encouraging Actions")
    print("=" * 60)

    # Create MCP client
    mcp_client = create_mcp_client({
        "python": {"type": "python"}
    })

    # Mock LLM
    base_llm = MockLLMProvider({
        "calculate": "I'll use Python to calculate this accurately.",
        "execute": "<tool_call>{\"tool\": \"execute_python\", \"arguments\": {\"code\": \"result = 37 * 43\\nprint(f'37 * 43 = {result}')\"}}</tool_call>",
        "verify": "The calculation is correct: 1591",
        "terminal": "YES",
        "quality": "0.95"
    })

    mcp_llm = create_mcp_provider(base_llm, mcp_client=mcp_client)

    # Create action that explicitly encourages code execution
    code_action = create_code_execution_action()

    print("\nCreated MCP action:")
    print(f"  Operation: {code_action.operation.value}")
    print(f"  Focus: {code_action.focus.value}")
    print(f"  MCP Intent: {code_action.mcp_intent.value}")
    print(f"  Suggested Tools: {code_action.suggested_tools}")

    # Test the action's prompt
    prompt = code_action.to_prompt(
        current_state="Question: What is 37 * 43?",
        original_question="What is 37 * 43?",
        available_tools=["execute_python"]
    )

    print("\nGenerated prompt excerpt:")
    print("-" * 40)
    print(prompt[-300:])


def demo_mcp_action_selector():
    """Demo 3: MCP-aware action selector"""
    print("\n\n" + "=" * 60)
    print("Demo 3: MCP-Aware Action Selector")
    print("=" * 60)

    # Create MCP client
    mcp_client = create_mcp_client({
        "python": {"type": "python"},
        "web": {"type": "web"}
    })

    # Create MCP-aware action selector
    selector = MCPActionSelector(
        exploration_constant=1.414,
        mcp_client=mcp_client
    )

    print("\nGenerating MCP-aware actions...")
    actions = selector.get_mcp_actions(
        current_state="Let's solve this problem",
        n_samples=10
    )

    print(f"\nGenerated {len(actions)} actions:")
    print("-" * 40)

    # Count MCP intents
    intent_counts = {}
    for action in actions:
        intent = action.mcp_intent.value
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    for i, action in enumerate(actions[:5], 1):  # Show first 5
        print(f"\n{i}. {action.operation.value} + {action.focus.value}")
        print(f"   MCP Intent: {action.mcp_intent.value}")
        if action.suggested_tools:
            print(f"   Suggested Tools: {', '.join(action.suggested_tools)}")

    print("\n" + "=" * 40)
    print("Intent Distribution:")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}/{len(actions)} ({count/len(actions):.0%})")


def demo_mcts_with_mcp_actions():
    """Demo 4: Full MCTS with automatic MCP actions"""
    print("\n\n" + "=" * 60)
    print("Demo 4: MCTS with Automatic MCP Actions")
    print("=" * 60)

    # Setup
    mcp_client = create_mcp_client({
        "python": {"type": "python"}
    })

    base_llm = MockLLMProvider({
        "solve": "Let me write code to solve this",
        "execute": "<tool_call>{\"tool\": \"execute_python\", \"arguments\": {\"code\": \"sum(p for p in range(2, 20) if all(p % i != 0 for i in range(2, int(p**0.5)+1)))\"}}</tool_call>",
        "result": "The sum is 77",
        "terminal": "YES",
        "quality": "0.92"
    })

    mcp_llm = create_mcp_provider(base_llm, mcp_client=mcp_client)

    # Create custom ReasoningMCTS that uses MCP action selector
    # (This would be integrated into ReasoningMCTS with a flag)

    print("\nQuestion: Sum of all primes less than 20")
    print("Tools available: execute_python")
    print("\nMCTS will automatically generate actions that encourage tool usage")
    print("(~40% of actions will have MCP intent)")

    # For this demo, we'll just show what would happen
    selector = MCPActionSelector(mcp_client=mcp_client)
    sample_actions = selector.get_mcp_actions("Sum of primes", n_samples=5)

    print("\nSample actions that MCTS might explore:")
    for i, action in enumerate(sample_actions, 1):
        print(f"\n{i}. {action}")
        print(f"   Intent: {action.mcp_intent.value}")
        if action.mcp_intent != MCPActionIntent.NONE:
            print(f"   → Will encourage using: {', '.join(action.suggested_tools)}")


def demo_custom_tool_handler():
    """Demo 5: Register custom tool handler"""
    print("\n\n" + "=" * 60)
    print("Demo 5: Custom Tool Handler")
    print("=" * 60)

    # Create MCP client
    mcp_client = create_mcp_client({})

    # Register custom Python execution handler
    def python_executor(arguments: dict) -> str:
        """Custom Python execution with actual eval."""
        code = arguments.get("code", "")
        print(f"\n[Executing Python code:]")
        print(code)

        try:
            # In production, use proper sandboxing!
            result = eval(code)
            return f"Result: {result}"
        except Exception as e:
            return f"Error: {e}"

    # Register the handler
    from mcts_reasoning.compositional import MCPTool, MCPToolType

    tool = MCPTool(
        name="execute_python_real",
        description="Execute Python code (real execution)",
        tool_type=MCPToolType.PYTHON_EVAL,
        parameters_schema={"code": {"type": "string"}}
    )

    mcp_client.register_tool(tool)
    mcp_client.register_tool_handler("execute_python_real", python_executor)

    print("\nRegistered custom tool: execute_python_real")

    # Test it
    from mcts_reasoning.compositional import MCPToolCall

    call = MCPToolCall(
        tool_name="execute_python_real",
        arguments={"code": "sum(range(1, 11))"}
    )

    result = mcp_client.execute_tool(call)

    print(f"\nTool execution result:")
    print(f"  Success: {result.success}")
    print(f"  Result: {result.result}")


def main():
    """Run all demos."""
    demo_mcp_transparent()
    demo_mcp_explicit_actions()
    demo_mcp_action_selector()
    demo_mcts_with_mcp_actions()
    demo_custom_tool_handler()

    print("\n\n" + "=" * 60)
    print("✅ MCP Integration Demo Complete!")
    print("=" * 60)

    print("\nKey Takeaways:")
    print("1. LLMs automatically get access to MCP tools")
    print("2. Tools are used transparently during generation")
    print("3. MCTS actions can explicitly encourage tool usage")
    print("4. ~40% of actions have MCP intent by default")
    print("5. Custom tools can be easily registered")

    print("\nIn the TUI, you'll be able to:")
    print("  /mcp-connect <server> - Connect to MCP server")
    print("  /mcp-list            - List connected servers")
    print("  /mcp-tools           - Show available tools")
    print("  /mcp-disconnect      - Disconnect from server")


if __name__ == "__main__":
    main()
