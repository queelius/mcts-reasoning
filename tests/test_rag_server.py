"""
Tests for the RAG MCP Server.

Tests the RAG server functionality without requiring the MCP SDK.
"""

import pytest
import json

from mcts_reasoning.compositional.rag import (
    SolutionRAGStore,
    CompositionalRAGStore,
    CompositionalGuidance,
)
from mcts_reasoning.compositional.examples import Example
from mcts_reasoning.compositional import (
    CognitiveOperation,
    FocusAspect,
    ReasoningStyle,
)


# ========== Test RAG Server Functions ==========

class TestRAGServerFunctions:
    """Test the RAG server tool functions."""

    def test_retrieve_examples_empty_store(self):
        """Test retrieving from empty store."""
        store = SolutionRAGStore()

        # Simulate the retrieve_examples function
        examples = store.retrieve("test query", k=3)
        assert len(examples) == 0

    def test_retrieve_examples_with_data(self):
        """Test retrieving examples with populated store."""
        store = SolutionRAGStore()

        # Add some examples using correct API
        store.add(
            problem="What is 2+2?",
            solution="4",
            reasoning_steps=["Add the numbers", "2+2=4"],
            domain="math",
        )
        store.add(
            problem="Solve x+3=5",
            solution="x=2",
            reasoning_steps=["Subtract 3", "x=5-3=2"],
            domain="math",
        )

        # Retrieve similar examples
        examples = store.retrieve("What is 3+3?", k=2)
        assert len(examples) <= 2
        # Should find the arithmetic example
        assert any("+" in ex.problem or "add" in ex.problem.lower() for ex in examples)

    def test_get_guidance_empty_store(self):
        """Test getting guidance from empty store."""
        store = CompositionalRAGStore()

        # retrieve() returns empty list for empty store
        guidance_list = store.retrieve("solve x^2=4")
        assert len(guidance_list) == 0

    def test_get_guidance_with_patterns(self):
        """Test getting guidance with populated store."""
        store = CompositionalRAGStore()

        # Add guidance pattern
        store.add_guidance(CompositionalGuidance(
            problem_pattern="algebraic equations",
            problem_keywords=["solve", "equation", "x", "variable"],
            recommended_operations=[CognitiveOperation.DECOMPOSE],
            recommended_focuses=[FocusAspect.STRUCTURE],
            recommended_styles=[ReasoningStyle.SYSTEMATIC],
            domain="math",
        ))

        # Query should match
        guidance_list = store.retrieve("solve the equation x+5=10")
        assert len(guidance_list) > 0
        guidance = guidance_list[0]
        assert guidance.domain == "math"
        assert CognitiveOperation.DECOMPOSE in guidance.recommended_operations

    def test_get_recommended_weights(self):
        """Test getting recommended weights."""
        store = CompositionalRAGStore()

        # Add guidance with specific operations
        store.add_guidance(CompositionalGuidance(
            problem_pattern="arithmetic",
            problem_keywords=["add", "sum", "calculate"],
            recommended_operations=[
                CognitiveOperation.DECOMPOSE,
                CognitiveOperation.VERIFY,
            ],
            recommended_focuses=[FocusAspect.DETAILS],
            domain="math",
        ))

        weights = store.get_recommended_weights("calculate the sum of 1+2+3")
        assert weights is not None
        assert "cognitive_op" in weights
        assert CognitiveOperation.DECOMPOSE in weights["cognitive_op"]

    def test_add_example_to_store(self):
        """Test adding examples to store."""
        store = SolutionRAGStore()

        # Use add() with correct API
        store.add(
            problem="What is 5*5?",
            solution="25",
            reasoning_steps=["Multiply 5 by 5", "5*5=25"],
            domain="math",
            difficulty="easy",
        )

        assert len(store) == 1

        # Retrieve to verify
        retrieved = store.retrieve("multiply numbers", k=1)
        assert len(retrieved) == 1
        assert retrieved[0].solution == "25"

    def test_list_domains(self):
        """Test listing domains."""
        store = SolutionRAGStore()

        # Add examples in different domains
        store.add(problem="2+2", solution="4", domain="math")
        store.add(problem="if A then B", solution="modus ponens", domain="logic")
        store.add(problem="3*3", solution="9", domain="math")

        # Count domains
        domains = {}
        for ex in store.examples:
            domain = ex.metadata.get("domain", "general")
            domains[domain] = domains.get(domain, 0) + 1

        assert domains["math"] == 2
        assert domains["logic"] == 1


class TestDefaultGuidance:
    """Test the default guidance patterns."""

    def test_math_guidance_matching(self):
        """Test that math problems match math guidance."""
        store = CompositionalRAGStore()

        # Add math guidance
        store.add_guidance(CompositionalGuidance(
            problem_pattern="algebraic equations",
            problem_keywords=["solve", "equation", "x", "algebra"],
            recommended_operations=[CognitiveOperation.DECOMPOSE],
            domain="math",
        ))

        guidance_list = store.retrieve("solve x^2 + 5x + 6 = 0")
        assert len(guidance_list) > 0
        assert guidance_list[0].domain == "math"

    def test_logic_guidance_matching(self):
        """Test that logic problems match logic guidance."""
        store = CompositionalRAGStore()

        # Add logic guidance
        store.add_guidance(CompositionalGuidance(
            problem_pattern="logical deduction",
            problem_keywords=["if", "then", "therefore", "implies"],
            recommended_operations=[CognitiveOperation.ANALYZE],
            recommended_styles=[ReasoningStyle.FORMAL],
            domain="logic",
        ))

        guidance_list = store.retrieve("if A implies B and B implies C, then what?")
        assert len(guidance_list) > 0
        assert guidance_list[0].domain == "logic"
        assert ReasoningStyle.FORMAL in guidance_list[0].recommended_styles

    def test_coding_guidance_matching(self):
        """Test that coding problems match coding guidance."""
        store = CompositionalRAGStore()

        # Add coding guidance
        store.add_guidance(CompositionalGuidance(
            problem_pattern="algorithm design",
            problem_keywords=["algorithm", "code", "function", "implement"],
            recommended_operations=[CognitiveOperation.DECOMPOSE],
            domain="coding",
        ))

        guidance_list = store.retrieve("implement a sorting algorithm")
        assert len(guidance_list) > 0
        assert guidance_list[0].domain == "coding"


class TestRAGServerIntegration:
    """Integration tests for RAG functionality with tools."""

    def test_example_retrieval_with_tool_context(self):
        """Test using RAG retrieval in a tool-like context."""
        from mcts_reasoning.tools import ToolContext

        # Create a mock tool that uses RAG internally
        store = SolutionRAGStore()
        store.add(
            problem="What is the area of a circle?",
            solution="A = πr²",
            reasoning_steps=["Use the formula", "Area = π * radius²"],
        )

        def retrieve_handler(args):
            query = args.get("query", "")
            examples = store.retrieve(query, k=1)
            if examples:
                return examples[0].to_prompt_string()
            return "No examples found"

        context = ToolContext.mock({
            "retrieve": {
                "description": "Retrieve similar examples",
                "response": retrieve_handler,
            },
        })

        import asyncio
        asyncio.run(context.start())

        # Process a tool call
        response = '<tool_call name="retrieve"><query>circle area</query></tool_call>'
        result = context.process_response(response)

        assert result.has_calls
        # The mock returns the handler result
        assert result.results[0].result is not None

    def test_guidance_with_tool_context(self):
        """Test using compositional guidance in a tool-like context."""
        from mcts_reasoning.tools import ToolContext

        store = CompositionalRAGStore()
        store.add_guidance(CompositionalGuidance(
            problem_pattern="math",
            problem_keywords=["solve", "calculate"],
            recommended_operations=[CognitiveOperation.DECOMPOSE],
            domain="math",
        ))

        def guidance_handler(args):
            query = args.get("query", "")
            guidance_list = store.retrieve(query, k=1)
            if guidance_list:
                guidance = guidance_list[0]
                return json.dumps({
                    "domain": guidance.domain,
                    "operations": [op.value for op in (guidance.recommended_operations or [])],
                })
            return json.dumps({"message": "No guidance found"})

        context = ToolContext.mock({
            "get_guidance": {
                "description": "Get reasoning guidance",
                "response": guidance_handler,
            },
        })

        import asyncio
        asyncio.run(context.start())

        # Process a tool call
        response = '<tool_call name="get_guidance"><query>solve equation</query></tool_call>'
        result = context.process_response(response)

        assert result.has_calls
        result_data = json.loads(result.results[0].result)
        assert result_data["domain"] == "math"


class TestRAGServerCreation:
    """Test RAG server creation (without requiring MCP)."""

    def test_server_creation_requires_mcp(self):
        """Test that server creation fails gracefully without MCP."""
        try:
            from mcts_reasoning.tools.rag_server import HAS_FASTMCP, create_rag_server

            if not HAS_FASTMCP:
                with pytest.raises(ImportError):
                    create_rag_server()
            else:
                # If MCP is available, server creation should work
                server = create_rag_server()
                assert server is not None
        except ImportError:
            # Module import failed - expected without MCP
            pass

    def test_default_guidance_population(self):
        """Test that default guidance is added correctly."""
        from mcts_reasoning.tools.rag_server import _add_default_guidance

        store = CompositionalRAGStore()
        assert len(store) == 0

        _add_default_guidance(store)

        # Should have multiple guidance patterns
        assert len(store) >= 4  # math, logic, coding, general

        # Test that patterns work
        math_guidance = store.retrieve("solve the equation")
        assert len(math_guidance) > 0

        logic_guidance = store.retrieve("if A then B therefore")
        assert len(logic_guidance) > 0
