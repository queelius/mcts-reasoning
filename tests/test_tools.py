"""
Tests for the MCP tools integration module.
"""

import pytest
import json

from mcts_reasoning.tools import (
    # Formats
    ToolFormat,
    ToolDefinition,
    ToolCall,
    ToolResult,
    # Execution
    ToolExecutionResult,
    ToolCallParser,
    ToolCallHandler,
    # Registry
    MCPToolRegistry,
    create_tool_from_mcp,
    # Client
    ServerConfig,
    MockMCPClientManager,
    # Context
    ToolContext,
    # Generator
    ToolAwareGenerator,
    wrap_generator_with_tools,
    create_native_tool_generator,
    # Native function calling
    OpenAINativeWrapper,
    AnthropicNativeWrapper,
    wrap_provider_for_native_tools,
    supports_native_function_calling,
)
from mcts_reasoning.generator import MockGenerator, Continuation


# ========== Test ToolDefinition ==========

class TestToolDefinition:
    """Tests for ToolDefinition."""

    def test_create_basic_definition(self):
        """Test creating a basic tool definition."""
        tool = ToolDefinition(
            name="calculator",
            description="Perform calculations",
        )
        assert tool.name == "calculator"
        assert tool.description == "Perform calculations"
        assert tool.parameters == {}
        assert tool.required == []

    def test_create_definition_with_parameters(self):
        """Test creating a tool with parameters."""
        tool = ToolDefinition(
            name="search",
            description="Search the web",
            parameters={
                "query": {"type": "string", "description": "Search query"},
                "limit": {"type": "integer", "description": "Max results"},
            },
            required=["query"],
        )
        assert "query" in tool.parameters
        assert "limit" in tool.parameters
        assert tool.required == ["query"]

    def test_to_xml_schema(self):
        """Test XML schema generation."""
        tool = ToolDefinition(
            name="calculator",
            description="Perform math",
            parameters={
                "expression": {"type": "string", "description": "Math expression"},
            },
            required=["expression"],
        )
        xml = tool.to_xml_schema()
        assert '<tool name="calculator">' in xml
        assert "<description>Perform math</description>" in xml
        assert "<expression>" in xml
        assert "(required)" in xml

    def test_to_json_schema(self):
        """Test JSON schema generation."""
        tool = ToolDefinition(
            name="search",
            description="Search",
            parameters={"query": {"type": "string"}},
            required=["query"],
        )
        schema = tool.to_json_schema()
        assert schema["name"] == "search"
        assert schema["description"] == "Search"
        assert schema["parameters"]["type"] == "object"
        assert "query" in schema["parameters"]["properties"]

    def test_to_function_schema(self):
        """Test function calling schema generation."""
        tool = ToolDefinition(
            name="get_weather",
            description="Get weather",
            parameters={"city": {"type": "string"}},
        )
        schema = tool.to_function_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"


# ========== Test ToolResult ==========

class TestToolResult:
    """Tests for ToolResult."""

    def test_success_result(self):
        """Test successful result."""
        result = ToolResult(
            call_id="123",
            name="calculator",
            result="42",
        )
        assert result.success is True
        assert result.result == "42"
        assert result.error is None

    def test_error_result(self):
        """Test error result."""
        result = ToolResult(
            call_id="123",
            name="calculator",
            result=None,
            error="Division by zero",
        )
        assert result.success is False
        assert result.error == "Division by zero"

    def test_to_xml_success(self):
        """Test XML formatting for success."""
        result = ToolResult(call_id=None, name="calc", result="42")
        xml = result.to_xml()
        assert '<tool_result name="calc">42</tool_result>' == xml

    def test_to_xml_error(self):
        """Test XML formatting for error."""
        result = ToolResult(call_id=None, name="calc", result=None, error="fail")
        xml = result.to_xml()
        assert 'error="true"' in xml
        assert "fail" in xml

    def test_to_json(self):
        """Test JSON formatting."""
        result = ToolResult(call_id="abc", name="calc", result="42")
        data = result.to_json()
        assert data["name"] == "calc"
        assert data["result"] == "42"
        assert data["call_id"] == "abc"


# ========== Test ToolCallParser ==========

class TestToolCallParser:
    """Tests for ToolCallParser."""

    def test_parse_xml_single_call(self):
        """Test parsing single XML tool call."""
        parser = ToolCallParser()
        text = '''Let me calculate that.
<tool_call name="calculator">
  <expression>2 + 2</expression>
</tool_call>
'''
        calls = parser.parse(text, ToolFormat.XML)
        assert len(calls) == 1
        assert calls[0].name == "calculator"
        assert calls[0].arguments["expression"] == "2 + 2"

    def test_parse_xml_multiple_calls(self):
        """Test parsing multiple XML tool calls."""
        parser = ToolCallParser()
        text = '''<tool_call name="search">
  <query>python mcts</query>
</tool_call>
<tool_call name="calculator">
  <expression>5 * 3</expression>
</tool_call>'''
        calls = parser.parse(text, ToolFormat.XML)
        assert len(calls) == 2
        assert calls[0].name == "search"
        assert calls[1].name == "calculator"

    def test_parse_xml_no_calls(self):
        """Test parsing text without tool calls."""
        parser = ToolCallParser()
        text = "This is just regular text without any tool calls."
        calls = parser.parse(text, ToolFormat.XML)
        assert len(calls) == 0

    def test_parse_xml_multiple_params(self):
        """Test parsing tool call with multiple parameters."""
        parser = ToolCallParser()
        text = '''<tool_call name="search">
  <query>test query</query>
  <limit>10</limit>
  <sort>relevance</sort>
</tool_call>'''
        calls = parser.parse(text, ToolFormat.XML)
        assert len(calls) == 1
        assert calls[0].arguments["query"] == "test query"
        assert calls[0].arguments["limit"] == 10  # JSON-parsed as integer
        assert calls[0].arguments["sort"] == "relevance"

    def test_parse_json_single_call(self):
        """Test parsing JSON tool call."""
        parser = ToolCallParser()
        text = '{"tool": "calculator", "arguments": {"expression": "2+2"}}'
        calls = parser.parse(text, ToolFormat.JSON)
        assert len(calls) >= 1
        assert calls[0].name == "calculator"

    def test_remove_tool_calls_xml(self):
        """Test removing XML tool calls from text."""
        parser = ToolCallParser()
        text = '''Before call.
<tool_call name="calc">
  <x>1</x>
</tool_call>
After call.'''
        remaining = parser.remove_tool_calls(text, ToolFormat.XML)
        assert "<tool_call" not in remaining
        assert "Before call" in remaining
        assert "After call" in remaining


# ========== Test ToolCallHandler ==========

class TestToolCallHandler:
    """Tests for ToolCallHandler."""

    def test_process_with_executor(self):
        """Test processing with an executor."""
        def mock_executor(name, args):
            return f"Result for {name}"

        handler = ToolCallHandler(executor=mock_executor)
        text = '<tool_call name="test"><arg>val</arg></tool_call>'
        result = handler.process(text)

        assert result.has_calls
        assert len(result.calls) == 1
        assert len(result.results) == 1
        assert result.results[0].result == "Result for test"

    def test_process_without_executor(self):
        """Test processing without an executor."""
        handler = ToolCallHandler(executor=None)
        text = '<tool_call name="test"><arg>val</arg></tool_call>'
        result = handler.process(text)

        assert result.has_calls
        assert result.results[0].error == "No executor configured"

    def test_process_no_calls(self):
        """Test processing text without tool calls."""
        handler = ToolCallHandler()
        result = handler.process("Just regular text.")
        assert not result.has_calls
        assert len(result.calls) == 0

    def test_process_executor_exception(self):
        """Test handling executor exceptions."""
        def failing_executor(name, args):
            raise ValueError("Execution failed")

        handler = ToolCallHandler(executor=failing_executor)
        text = '<tool_call name="test"><arg>val</arg></tool_call>'
        result = handler.process(text)

        assert result.has_calls
        assert result.results[0].error == "Execution failed"


# ========== Test ToolExecutionResult ==========

class TestToolExecutionResult:
    """Tests for ToolExecutionResult."""

    def test_empty_result(self):
        """Test empty result."""
        result = ToolExecutionResult()
        assert not result.has_calls
        assert result.all_succeeded is True  # vacuously true

    def test_with_calls(self):
        """Test result with calls."""
        result = ToolExecutionResult(
            calls=[ToolCall(name="test", arguments={})],
            results=[ToolResult(call_id=None, name="test", result="ok")],
        )
        assert result.has_calls
        assert result.all_succeeded

    def test_with_failed_call(self):
        """Test result with failed call."""
        result = ToolExecutionResult(
            calls=[ToolCall(name="test", arguments={})],
            results=[ToolResult(call_id=None, name="test", result=None, error="fail")],
        )
        assert result.has_calls
        assert not result.all_succeeded

    def test_format_results_xml(self):
        """Test formatting results as XML."""
        result = ToolExecutionResult(
            results=[ToolResult(call_id=None, name="calc", result="42")],
        )
        formatted = result.format_results(ToolFormat.XML)
        assert '<tool_result name="calc">42</tool_result>' in formatted


# ========== Test MCPToolRegistry ==========

class TestMCPToolRegistry:
    """Tests for MCPToolRegistry."""

    def test_register_tool(self):
        """Test registering a tool."""
        registry = MCPToolRegistry()
        tool = ToolDefinition(name="test", description="Test tool")
        registry.register_tool(tool, server_name="server1")

        assert "test" in registry
        assert len(registry) == 1
        assert registry.get_tool("test") == tool
        assert registry.get_server("test") == "server1"

    def test_register_multiple_tools(self):
        """Test registering multiple tools."""
        registry = MCPToolRegistry()
        tools = [
            ToolDefinition(name="tool1", description="Tool 1"),
            ToolDefinition(name="tool2", description="Tool 2"),
        ]
        registry.register_tools(tools, server_name="server")

        assert len(registry) == 2
        assert "tool1" in registry
        assert "tool2" in registry

    def test_unregister_server(self):
        """Test unregistering a server."""
        registry = MCPToolRegistry()
        registry.register_tool(
            ToolDefinition(name="t1", description=""),
            server_name="s1",
        )
        registry.register_tool(
            ToolDefinition(name="t2", description=""),
            server_name="s2",
        )

        registry.unregister_server("s1")
        assert "t1" not in registry
        assert "t2" in registry

    def test_format_tools_prompt_xml(self):
        """Test XML prompt formatting."""
        registry = MCPToolRegistry()
        registry.register_tool(
            ToolDefinition(name="calc", description="Calculate"),
            server_name="s",
        )

        prompt = registry.format_tools_prompt(ToolFormat.XML)
        assert "<available_tools>" in prompt
        assert '<tool name="calc">' in prompt

    def test_get_tool_use_instruction(self):
        """Test getting tool use instructions."""
        registry = MCPToolRegistry()

        xml_inst = registry.get_tool_use_instruction(ToolFormat.XML)
        assert "<tool_call" in xml_inst

        json_inst = registry.get_tool_use_instruction(ToolFormat.JSON)
        assert '"tool"' in json_inst


# ========== Test create_tool_from_mcp ==========

class TestCreateToolFromMCP:
    """Tests for create_tool_from_mcp."""

    def test_create_from_mcp_schema(self):
        """Test creating tool from MCP schema."""
        mcp_tool = {
            "name": "search",
            "description": "Search for documents",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        }
        tool = create_tool_from_mcp(mcp_tool)

        assert tool.name == "search"
        assert tool.description == "Search for documents"
        assert "query" in tool.parameters
        assert tool.required == ["query"]

    def test_create_from_minimal_schema(self):
        """Test creating tool from minimal schema."""
        mcp_tool = {"name": "simple"}
        tool = create_tool_from_mcp(mcp_tool)

        assert tool.name == "simple"
        assert tool.description == ""
        assert tool.parameters == {}


# ========== Test MockMCPClientManager ==========

class TestMockMCPClientManager:
    """Tests for MockMCPClientManager."""

    def test_register_mock_tool(self):
        """Test registering a mock tool."""
        manager = MockMCPClientManager()
        manager.register_mock_tool(
            name="calculator",
            description="Calculate",
            response="42",
        )

        assert len(manager) == 1
        tools = manager.get_available_tools()
        assert len(tools) == 1
        assert tools[0].name == "calculator"

    @pytest.mark.asyncio
    async def test_call_mock_tool(self):
        """Test calling a mock tool."""
        manager = MockMCPClientManager()
        manager.register_mock_tool(
            name="double",
            response=lambda args: args.get("x", 0) * 2,
        )
        await manager.start()

        result = await manager.call_tool("double", {"x": 5})
        assert result.success
        assert result.result == 10

    @pytest.mark.asyncio
    async def test_call_unknown_tool(self):
        """Test calling an unknown tool."""
        manager = MockMCPClientManager()
        await manager.start()

        result = await manager.call_tool("unknown", {})
        assert not result.success
        assert "not found" in result.error

    def test_call_history(self):
        """Test call history tracking."""
        manager = MockMCPClientManager()
        manager.register_mock_tool(name="test", response="ok")

        import asyncio
        asyncio.run(manager.start())
        asyncio.run(manager.call_tool("test", {"a": 1}))
        asyncio.run(manager.call_tool("test", {"b": 2}))

        history = manager.get_call_history()
        assert len(history) == 2
        assert history[0]["arguments"] == {"a": 1}


# ========== Test ToolContext ==========

class TestToolContext:
    """Tests for ToolContext."""

    def test_create_mock_context(self):
        """Test creating a mock context."""
        context = ToolContext.mock({
            "calc": {"description": "Calculate", "response": "42"},
        })

        assert context.has_tools
        assert len(context) == 1

    def test_inject_tools_into_prompt(self):
        """Test injecting tools into prompt."""
        context = ToolContext.mock({
            "search": {"description": "Search the web", "response": "results"},
        })

        prompt = "Solve this problem."
        augmented = context.inject_tools_into_prompt(prompt)

        assert "Solve this problem" in augmented
        assert "<available_tools>" in augmented
        assert "search" in augmented

    def test_process_response_with_tool_calls(self):
        """Test processing response with tool calls."""
        context = ToolContext.mock({
            "calc": {"response": "42"},
        })
        import asyncio
        asyncio.run(context.start())

        response = '<tool_call name="calc"><expr>2+2</expr></tool_call>'
        result = context.process_response(response)

        assert result.has_calls
        assert result.results[0].result == "42"

    def test_process_response_without_tool_calls(self):
        """Test processing response without tool calls."""
        context = ToolContext.mock({})

        result = context.process_response("Just regular text.")
        assert not result.has_calls

    def test_augment_state_with_results(self):
        """Test augmenting state with tool results."""
        context = ToolContext.mock({"calc": {"response": "42"}})
        import asyncio
        asyncio.run(context.start())

        result = context.process_response('<tool_call name="calc"><x>1</x></tool_call>')
        augmented = context.augment_state_with_results("State", result)

        assert "State" in augmented
        assert "[Tool Results]" in augmented
        assert "42" in augmented

    def test_context_without_tools(self):
        """Test context without any tools."""
        context = ToolContext.mock({})

        assert not context.has_tools
        assert len(context) == 0

        prompt = context.inject_tools_into_prompt("Test")
        assert prompt == "Test"


# ========== Test ToolAwareGenerator ==========

class TestToolAwareGenerator:
    """Tests for ToolAwareGenerator."""

    def test_generate_without_context(self):
        """Test generating without tool context."""
        base_gen = MockGenerator()
        tool_gen = ToolAwareGenerator(base_generator=base_gen)

        result = tool_gen.generate("What is 2+2?", "Initial state", n=1)

        assert len(result) == 1
        assert isinstance(result[0], Continuation)

    def test_generate_with_mock_context(self):
        """Test generating with mock tool context."""
        base_gen = MockGenerator()
        context = ToolContext.mock({
            "calc": {"description": "Calculate", "response": "4"},
        })
        import asyncio
        asyncio.run(context.start())

        tool_gen = ToolAwareGenerator(
            base_generator=base_gen,
            tool_context=context,
        )

        result = tool_gen.generate("What is 2+2?", "Initial state", n=1)
        assert len(result) == 1

    def test_wrap_generator_with_tools(self):
        """Test wrap_generator_with_tools convenience function."""
        base_gen = MockGenerator()
        context = ToolContext.mock({})

        tool_gen = wrap_generator_with_tools(base_gen, context)

        assert isinstance(tool_gen, ToolAwareGenerator)
        assert tool_gen.base_generator is base_gen
        assert tool_gen.tool_context is context

    def test_inject_tools(self):
        """Test injecting tools into prompt."""
        base_gen = MockGenerator()
        context = ToolContext.mock({
            "search": {"description": "Search", "response": "results"},
        })

        tool_gen = ToolAwareGenerator(
            base_generator=base_gen,
            tool_context=context,
        )

        prompt = tool_gen.inject_tools("Solve this.")
        assert "<available_tools>" in prompt
        assert "search" in prompt


# ========== Test ServerConfig ==========

class TestServerConfig:
    """Tests for ServerConfig."""

    def test_create_basic_config(self):
        """Test creating basic server config."""
        config = ServerConfig(
            name="rag",
            command=["python", "-m", "rag_server"],
        )

        assert config.name == "rag"
        assert config.command == ["python", "-m", "rag_server"]
        assert config.env == {}
        assert config.args == []

    def test_create_config_with_env(self):
        """Test creating config with environment variables."""
        config = ServerConfig(
            name="test",
            command=["./server"],
            env={"API_KEY": "secret"},
            args=["--verbose"],
        )

        assert config.env == {"API_KEY": "secret"}
        assert config.args == ["--verbose"]


# ========== Enhanced Parsing Tests ==========

class TestEnhancedParsing:
    """Tests for enhanced tool call parsing."""

    def test_parse_alternative_xml_format(self):
        """Test parsing alternative XML formats like function_call."""
        parser = ToolCallParser()
        text = '''<function_call name="search">
  <query>test</query>
</function_call>'''
        calls = parser.parse(text, ToolFormat.XML)
        assert len(calls) == 1
        assert calls[0].name == "search"

    def test_parse_tool_use_format(self):
        """Test parsing tool_use format."""
        parser = ToolCallParser()
        text = '<tool_use name="calculator"><expression>1+1</expression></tool_use>'
        calls = parser.parse(text, ToolFormat.XML)
        assert len(calls) == 1
        assert calls[0].name == "calculator"

    def test_parse_json_in_code_block(self):
        """Test parsing JSON tool calls from code blocks."""
        parser = ToolCallParser()
        text = '''Here's my tool call:
```json
{"tool": "search", "arguments": {"query": "test"}}
```
'''
        calls = parser.parse(text, ToolFormat.JSON)
        assert len(calls) >= 1
        assert any(c.name == "search" for c in calls)

    def test_parse_anthropic_style_json(self):
        """Test parsing Anthropic-style JSON format with 'name' and 'input'."""
        parser = ToolCallParser()
        text = '{"name": "calculator", "input": {"expression": "2+2"}}'
        calls = parser.parse(text, ToolFormat.JSON)
        assert len(calls) >= 1
        assert calls[0].name == "calculator"
        assert calls[0].arguments.get("expression") == "2+2"

    def test_parse_xml_with_json_content(self):
        """Test parsing XML tool call with JSON as parameter content."""
        parser = ToolCallParser()
        text = '<tool_call name="api">{"endpoint": "/users", "method": "GET"}</tool_call>'
        calls = parser.parse(text, ToolFormat.XML)
        assert len(calls) == 1
        assert calls[0].name == "api"
        # JSON content should be parsed
        assert calls[0].arguments.get("endpoint") == "/users"


# ========== MCTS Integration Tests ==========

class TestMCTSIntegration:
    """Integration tests for ToolAwareGenerator with MCTS components."""

    def test_tool_generator_with_mcts_search(self):
        """Test ToolAwareGenerator works with MCTS search."""
        from mcts_reasoning import MCTS, MockEvaluator

        base_gen = MockGenerator()
        context = ToolContext.mock({
            "hint": {"description": "Get a hint", "response": "Think step by step"},
        })

        tool_gen = ToolAwareGenerator(
            base_generator=base_gen,
            tool_context=context,
        )

        mcts = MCTS(
            generator=tool_gen,
            evaluator=MockEvaluator(),
            max_children_per_node=2,
            max_rollout_depth=3,
        )

        result = mcts.search("What is 2+2?", simulations=5)

        # Should complete without errors
        assert result.root is not None
        assert result.simulations == 5

    def test_tool_generator_preserves_terminal_detection(self):
        """Test that terminal detection works through ToolAwareGenerator."""
        from mcts_reasoning.terminal import MarkerTerminalDetector

        # Generator that produces terminal output
        responses = [
            "Step 1: Let me think...",
            "Step 2: The answer is ANSWER: 4",
        ]
        base_gen = MockGenerator(responses=responses)
        tool_gen = ToolAwareGenerator(base_generator=base_gen)

        # First call - non-terminal
        result1 = tool_gen.generate("What is 2+2?", "Initial", n=1)
        assert not result1[0].is_terminal

        # Second call - terminal
        result2 = tool_gen.generate("What is 2+2?", result1[0].text, n=1)
        assert result2[0].is_terminal
        assert result2[0].answer == "4"

    def test_mcts_with_tool_context_disabled(self):
        """Test MCTS works normally when tool context has no tools."""
        from mcts_reasoning import MCTS, MockEvaluator

        base_gen = MockGenerator()
        # Empty context - should behave like no tools
        context = ToolContext.mock({})

        tool_gen = ToolAwareGenerator(
            base_generator=base_gen,
            tool_context=context,
        )

        mcts = MCTS(
            generator=tool_gen,
            evaluator=MockEvaluator(),
        )

        result = mcts.search("Test", simulations=3)
        assert result.root is not None

    def test_multiple_continuations_with_tools(self):
        """Test generating multiple continuations with tool support."""
        base_gen = MockGenerator()
        context = ToolContext.mock({
            "calc": {"response": "42"},
        })

        tool_gen = ToolAwareGenerator(
            base_generator=base_gen,
            tool_context=context,
        )

        results = tool_gen.generate("What is 6*7?", "Initial state", n=3)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, Continuation)


# ========== End-to-End Tool Usage Tests ==========

class TestToolUsageE2E:
    """End-to-end tests simulating real tool usage patterns."""

    def test_calculator_tool_flow(self):
        """Test a complete flow with calculator tool."""
        import asyncio

        # Set up mock calculator
        def calculate(args):
            expr = args.get("expression", "0")
            try:
                return str(eval(expr))  # Simple eval for testing
            except Exception:
                return "Error"

        context = ToolContext.mock({
            "calculator": {
                "description": "Evaluate math expressions",
                "parameters": {"expression": {"type": "string"}},
                "response": calculate,
            },
        })
        asyncio.run(context.start())

        # Simulate LLM response with tool call
        response = '''Let me calculate that.
<tool_call name="calculator">
  <expression>2 + 2</expression>
</tool_call>'''

        result = context.process_response(response)
        assert result.has_calls
        assert result.results[0].result == "4"

    def test_search_and_reason_flow(self):
        """Test a flow with search tool followed by reasoning."""
        import asyncio

        context = ToolContext.mock({
            "search": {
                "description": "Search knowledge base",
                "response": "The formula for area of circle is A = πr²",
            },
        })
        asyncio.run(context.start())

        # First response: tool call
        response1 = '<tool_call name="search"><query>area of circle formula</query></tool_call>'
        result1 = context.process_response(response1)

        # Augment state with result
        state = "Question: What is the area of a circle with radius 5?"
        augmented = context.augment_state_with_results(state, result1)

        assert "area of circle" in augmented.lower() or "πr²" in augmented

    def test_context_manager_usage(self):
        """Test using ToolContext as context manager."""
        context = ToolContext.mock({
            "test": {"response": "ok"},
        })

        with context:
            assert context.mcp_manager.is_running
            result = context.process_response('<tool_call name="test"></tool_call>')
            assert result.has_calls


# ========== Native Function Calling Tests ==========

class TestNativeFunctionCalling:
    """Tests for native function calling support."""

    def test_supports_native_function_calling_mock(self):
        """Test checking native support for mock provider."""
        from mcts_reasoning.compositional import MockLLMProvider

        provider = MockLLMProvider()
        assert not supports_native_function_calling(provider)

    def test_supports_native_function_calling_openai_name(self):
        """Test checking native support for OpenAI-like provider by name."""
        class FakeOpenAIProvider:
            def get_provider_name(self):
                return "OpenAI-gpt-4"

        provider = FakeOpenAIProvider()
        assert supports_native_function_calling(provider)

    def test_supports_native_function_calling_anthropic_name(self):
        """Test checking native support for Anthropic-like provider by name."""
        class FakeAnthropicProvider:
            def get_provider_name(self):
                return "Anthropic-claude-3"

        provider = FakeAnthropicProvider()
        assert supports_native_function_calling(provider)

    def test_wrap_provider_mock_returns_none(self):
        """Test wrapping mock provider returns None."""
        from mcts_reasoning.compositional import MockLLMProvider

        provider = MockLLMProvider()
        wrapped = wrap_provider_for_native_tools(provider)
        assert wrapped is None

    def test_openai_native_wrapper_structure(self):
        """Test OpenAINativeWrapper structure."""
        class FakeOpenAIProvider:
            def get_provider_name(self):
                return "OpenAI-gpt-4"
            def _get_client(self):
                return None

        provider = FakeOpenAIProvider()
        wrapper = OpenAINativeWrapper(provider)

        assert wrapper.supports_native_tools()
        assert wrapper.provider is provider
        assert "native" in wrapper.get_provider_name().lower()

    def test_anthropic_native_wrapper_structure(self):
        """Test AnthropicNativeWrapper structure."""
        class FakeAnthropicProvider:
            def get_provider_name(self):
                return "Anthropic-claude"
            def _get_client(self):
                return None

        provider = FakeAnthropicProvider()
        wrapper = AnthropicNativeWrapper(provider)

        assert wrapper.supports_native_tools()
        assert wrapper.provider is provider
        assert "native" in wrapper.get_provider_name().lower()

    def test_tool_aware_generator_native_detection(self):
        """Test ToolAwareGenerator detects native provider."""
        base_gen = MockGenerator()
        context = ToolContext.mock({"calc": {"response": "42"}})

        # Without native provider
        gen1 = ToolAwareGenerator(
            base_generator=base_gen,
            tool_context=context,
        )
        assert not gen1._native_available

        # With mock native provider
        class MockNativeProvider:
            def supports_native_tools(self):
                return True

        gen2 = ToolAwareGenerator(
            base_generator=base_gen,
            tool_context=context,
            native_provider=MockNativeProvider(),
        )
        assert gen2._native_available

    def test_create_native_tool_generator_with_mock(self):
        """Test create_native_tool_generator with mock provider."""
        from mcts_reasoning.compositional import MockLLMProvider

        base_gen = MockGenerator()
        context = ToolContext.mock({"calc": {"response": "42"}})
        provider = MockLLMProvider()

        gen = create_native_tool_generator(base_gen, context, provider)

        # Mock provider doesn't support native, so should fall back
        assert isinstance(gen, ToolAwareGenerator)
        assert not gen._native_available

    def test_tool_aware_generator_falls_back_to_parsing(self):
        """Test that generator falls back to parsing when native not available."""
        base_gen = MockGenerator()
        context = ToolContext.mock({
            "calc": {"response": "42"},
        })
        import asyncio
        asyncio.run(context.start())

        gen = ToolAwareGenerator(
            base_generator=base_gen,
            tool_context=context,
            native_provider=None,  # No native provider
        )

        # Should work via parsing mode
        results = gen.generate("What is 2+2?", "Initial", n=1)
        assert len(results) == 1


class TestNativeWrapperIntegration:
    """Integration tests for native wrapper with tool context."""

    def test_wrap_generator_with_native_provider(self):
        """Test wrap_generator_with_tools with native provider."""
        class MockNativeProvider:
            def supports_native_tools(self):
                return True

            def generate_with_tools(self, prompt, tools, max_tokens=1000, temperature=0.7):
                return "Response", []

        base_gen = MockGenerator()
        context = ToolContext.mock({})
        native = MockNativeProvider()

        gen = wrap_generator_with_tools(
            base_gen, context, native_provider=native, use_native=True
        )

        assert gen._native_available
        assert gen.native_provider is native

    def test_wrap_generator_disable_native(self):
        """Test disabling native mode even with native provider."""
        class MockNativeProvider:
            def supports_native_tools(self):
                return True

        base_gen = MockGenerator()
        context = ToolContext.mock({})

        gen = wrap_generator_with_tools(
            base_gen, context,
            native_provider=MockNativeProvider(),
            use_native=False,
        )

        # Native should be disabled
        assert not gen._native_available
