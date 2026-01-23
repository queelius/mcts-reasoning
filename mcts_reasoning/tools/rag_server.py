"""
RAG MCP Server for MCTS Reasoning.

Exposes SolutionRAGStore and CompositionalRAGStore as MCP tools for
retrieval-augmented reasoning during MCTS search.

Usage:
    # Start the server
    python -m mcts_reasoning.tools.rag_server

    # Or with uvx/npx
    uvx mcp run mcts_reasoning.tools.rag_server

Tools provided:
    - retrieve_examples: Get similar solution examples for few-shot learning
    - get_guidance: Get compositional guidance for a problem type
    - add_example: Add a new example to the solution store
    - list_domains: List available domains in the stores
"""

import json
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

# Try to import FastMCP
try:
    from mcp.server.fastmcp import FastMCP

    HAS_FASTMCP = True
except ImportError:
    HAS_FASTMCP = False
    FastMCP = None

# Import RAG stores
from ..compositional.rag import (
    SolutionRAGStore,
    CompositionalRAGStore,
    CompositionalGuidance,
)
from ..compositional import (
    CognitiveOperation,
    FocusAspect,
    ReasoningStyle,
    ConnectionType,
    OutputFormat,
)


def create_rag_server(
    name: str = "mcts-rag",
    solution_store: Optional[SolutionRAGStore] = None,
    compositional_store: Optional[CompositionalRAGStore] = None,
) -> "FastMCP":
    """
    Create an MCP server exposing RAG functionality.

    Args:
        name: Server name
        solution_store: Pre-populated solution store (or creates empty)
        compositional_store: Pre-populated compositional store (or creates empty)

    Returns:
        FastMCP server instance
    """
    if not HAS_FASTMCP:
        raise ImportError("FastMCP not available. Install with: pip install mcp")

    # Initialize stores
    _solution_store = solution_store or SolutionRAGStore()
    _compositional_store = compositional_store or CompositionalRAGStore()

    # Pre-populate with default guidance if empty
    if len(_compositional_store) == 0:
        _add_default_guidance(_compositional_store)

    # Create server
    mcp = FastMCP(name)

    @mcp.tool()
    def retrieve_examples(
        query: str,
        domain: str = "math",
        k: int = 3,
        include_steps: bool = True,
    ) -> str:
        """
        Retrieve similar solution examples for few-shot learning.

        Args:
            query: The problem or question to find similar examples for
            domain: Domain to search in (math, logic, coding, etc.)
            k: Number of examples to retrieve
            include_steps: Whether to include reasoning steps

        Returns:
            Formatted examples as a string for prompt injection
        """
        # Get examples from the appropriate domain
        examples = _solution_store.retrieve(query, k=k)

        if not examples:
            return f"No similar examples found for domain '{domain}'."

        # Format examples for prompt
        formatted = []
        for i, ex in enumerate(examples, 1):
            formatted.append(f"Example {i}:")
            formatted.append(f"Problem: {ex.problem}")
            if include_steps and ex.reasoning_steps:
                formatted.append("Steps:")
                for j, step in enumerate(ex.reasoning_steps, 1):
                    formatted.append(f"  {j}. {step}")
            formatted.append(f"Solution: {ex.solution}")
            formatted.append("")

        return "\n".join(formatted)

    @mcp.tool()
    def get_guidance(query: str) -> str:
        """
        Get compositional guidance for a problem type.

        Analyzes the problem and returns recommended reasoning approaches
        based on the problem characteristics.

        Args:
            query: The problem or question to get guidance for

        Returns:
            JSON string with recommended operations, focuses, and styles
        """
        # Retrieve matching guidance entries
        guidance_list = _compositional_store.retrieve(query, k=1)

        if not guidance_list:
            return json.dumps(
                {
                    "message": "No specific guidance found for this problem type.",
                    "default_approach": {
                        "operation": "decompose",
                        "focus": "structure",
                        "style": "systematic",
                    },
                }
            )

        guidance = guidance_list[0]  # Best match

        result = {
            "problem_pattern": guidance.problem_pattern,
            "domain": guidance.domain,
            "recommended": {},
        }

        if guidance.recommended_operations:
            result["recommended"]["operations"] = [
                op.value for op in guidance.recommended_operations
            ]

        if guidance.recommended_focuses:
            result["recommended"]["focuses"] = [
                f.value for f in guidance.recommended_focuses
            ]

        if guidance.recommended_styles:
            result["recommended"]["styles"] = [
                s.value for s in guidance.recommended_styles
            ]

        if guidance.recommended_connections:
            result["recommended"]["connections"] = [
                c.value for c in guidance.recommended_connections
            ]

        if guidance.recommended_formats:
            result["recommended"]["formats"] = [
                f.value for f in guidance.recommended_formats
            ]

        return json.dumps(result, indent=2)

    @mcp.tool()
    def get_recommended_weights(query: str) -> str:
        """
        Get weighted sampling recommendations for compositional actions.

        Returns weights that can be used to bias action selection during MCTS.

        Args:
            query: The problem to get weights for

        Returns:
            JSON string with weights for each compositional dimension
        """
        weights = _compositional_store.get_recommended_weights(query)

        if not weights:
            return json.dumps(
                {"message": "No specific weights found. Using uniform sampling."}
            )

        # Convert enum keys to strings for JSON serialization
        serializable = {}
        for dimension, dim_weights in weights.items():
            serializable[dimension] = {
                (k.value if hasattr(k, "value") else str(k)): v
                for k, v in dim_weights.items()
            }

        return json.dumps(serializable, indent=2)

    @mcp.tool()
    def add_example(
        problem: str,
        solution: str,
        reasoning_steps: Optional[List[str]] = None,
        domain: str = "general",
        difficulty: str = "medium",
    ) -> str:
        """
        Add a new example to the solution store.

        Args:
            problem: The problem statement
            solution: The solution
            reasoning_steps: Optional list of reasoning steps
            domain: Problem domain (math, logic, coding, etc.)
            difficulty: Problem difficulty (easy, medium, hard)

        Returns:
            Confirmation message
        """
        _solution_store.add(
            problem=problem,
            solution=solution,
            reasoning_steps=reasoning_steps,
            domain=domain,
            difficulty=difficulty,
        )

        return f"Added example to {domain} domain. Store now has {len(_solution_store)} examples."

    @mcp.tool()
    def list_domains() -> str:
        """
        List available domains and their example counts.

        Returns:
            JSON string with domain statistics
        """
        # Get domain statistics from solution store
        domains = {}
        for example in _solution_store.examples:
            domain = example.metadata.get("domain", "general")
            domains[domain] = domains.get(domain, 0) + 1

        # Get guidance patterns
        patterns = []
        for guidance in _compositional_store.guidance:
            patterns.append(
                {
                    "pattern": guidance.problem_pattern,
                    "domain": guidance.domain,
                    "keywords": guidance.problem_keywords[:5],  # First 5 keywords
                }
            )

        result = {
            "solution_store": {
                "total_examples": len(_solution_store),
                "domains": domains,
            },
            "compositional_store": {
                "total_patterns": len(_compositional_store),
                "patterns": patterns[:10],  # First 10 patterns
            },
        }

        return json.dumps(result, indent=2)

    return mcp


def _add_default_guidance(store: CompositionalRAGStore) -> None:
    """Add default compositional guidance patterns."""

    # Math problems
    store.add_guidance(
        CompositionalGuidance(
            problem_pattern="algebraic equations",
            problem_keywords=["solve", "equation", "x", "variable", "algebra"],
            recommended_operations=[
                CognitiveOperation.DECOMPOSE,
                CognitiveOperation.ANALYZE,
                CognitiveOperation.VERIFY,
            ],
            recommended_focuses=[FocusAspect.STRUCTURE, FocusAspect.SOLUTION],
            recommended_styles=[ReasoningStyle.SYSTEMATIC, ReasoningStyle.FORMAL],
            recommended_connections=[
                ConnectionType.THEREFORE,
                ConnectionType.BUILDING_ON,
            ],
            recommended_formats=[OutputFormat.STEPS, OutputFormat.MATHEMATICAL],
            domain="math",
        )
    )

    store.add_guidance(
        CompositionalGuidance(
            problem_pattern="arithmetic calculations",
            problem_keywords=[
                "calculate",
                "sum",
                "product",
                "add",
                "multiply",
                "divide",
            ],
            recommended_operations=[
                CognitiveOperation.DECOMPOSE,
                CognitiveOperation.VERIFY,
            ],
            recommended_focuses=[FocusAspect.DETAILS, FocusAspect.CORRECTNESS],
            recommended_styles=[ReasoningStyle.SYSTEMATIC],
            recommended_formats=[OutputFormat.STEPS],
            domain="math",
        )
    )

    # Logic problems
    store.add_guidance(
        CompositionalGuidance(
            problem_pattern="logical deduction",
            problem_keywords=["if", "then", "therefore", "implies", "logic", "deduce"],
            recommended_operations=[
                CognitiveOperation.ANALYZE,
                CognitiveOperation.SYNTHESIZE,
                CognitiveOperation.VERIFY,
            ],
            recommended_focuses=[FocusAspect.ASSUMPTIONS, FocusAspect.STRUCTURE],
            recommended_styles=[ReasoningStyle.FORMAL, ReasoningStyle.CRITICAL],
            recommended_connections=[ConnectionType.THEREFORE, ConnectionType.HOWEVER],
            recommended_formats=[OutputFormat.STEPS, OutputFormat.EXPLANATION],
            domain="logic",
        )
    )

    # Coding problems
    store.add_guidance(
        CompositionalGuidance(
            problem_pattern="algorithm design",
            problem_keywords=["algorithm", "code", "function", "implement", "program"],
            recommended_operations=[
                CognitiveOperation.DECOMPOSE,
                CognitiveOperation.GENERATE,
                CognitiveOperation.REFINE,
            ],
            recommended_focuses=[FocusAspect.STRUCTURE, FocusAspect.EFFICIENCY],
            recommended_styles=[ReasoningStyle.SYSTEMATIC, ReasoningStyle.CREATIVE],
            recommended_formats=[OutputFormat.STEPS, OutputFormat.CODE],
            domain="coding",
        )
    )

    # General problem solving
    store.add_guidance(
        CompositionalGuidance(
            problem_pattern="general problem",
            problem_keywords=["problem", "solve", "find", "determine", "what", "how"],
            recommended_operations=[
                CognitiveOperation.DECOMPOSE,
                CognitiveOperation.ANALYZE,
            ],
            recommended_focuses=[FocusAspect.STRUCTURE, FocusAspect.GOAL],
            recommended_styles=[ReasoningStyle.SYSTEMATIC],
            recommended_formats=[OutputFormat.STEPS],
            domain="general",
        )
    )


# Main entry point for running as MCP server
def main():
    """Run the RAG MCP server."""
    if not HAS_FASTMCP:
        print("Error: FastMCP not available. Install with: pip install mcp")
        return 1

    server = create_rag_server()
    server.run()
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
