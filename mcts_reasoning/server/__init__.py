"""
MCTS Reasoning MCP Server.

Exposes MCTS search, exploration, and benchmarking as MCP tools.

Usage:
    python -m mcts_reasoning.server
"""

from __future__ import annotations


def create_server():
    """
    Create and return a FastMCP server with MCTS tools registered.

    Raises:
        ImportError: If the ``mcp`` package is not installed.
    """
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        raise ImportError(
            "MCP server requires the 'mcp' package. "
            "Install with: pip install mcts-reasoning[server]"
        )

    mcp = FastMCP("mcts-reasoning")

    from .tools import (
        mcts_bench_impl,
        mcts_explore_impl,
        mcts_search_impl,
        list_components_impl,
        get_contracts_impl,
    )

    @mcp.tool()
    def mcts_search(
        question: str,
        provider: str = "auto",
        model: str = "",
        simulations: int = 10,
        exploration_constant: float = 1.414,
    ) -> dict:
        """Run MCTS search on a question. Returns best answer with confidence."""
        return mcts_search_impl(
            question,
            provider,
            model or None,
            simulations,
            exploration_constant,
        )

    @mcp.tool()
    def mcts_explore(
        question: str,
        provider: str = "auto",
        model: str = "",
        simulations: int = 10,
    ) -> dict:
        """Run MCTS and return the full reasoning tree for inspection."""
        return mcts_explore_impl(
            question,
            provider,
            model or None,
            simulations,
        )

    @mcp.tool()
    def mcts_bench(
        benchmark: str = "knights",
        provider: str = "auto",
        model: str = "",
        simulations: str = "10",
    ) -> dict:
        """Run benchmark: baseline vs MCTS. Returns accuracy comparison."""
        sim_list = [int(s.strip()) for s in simulations.split(",")]
        return mcts_bench_impl(
            benchmark,
            provider,
            model or None,
            sim_list,
        )

    @mcp.tool()
    def list_components() -> dict:
        """List all available implementations for each MCTS decision point (providers, strategies, benchmarks)."""
        return list_components_impl()

    @mcp.tool()
    def get_contracts(contract_name: str = "") -> dict:
        """Get ABC interface contracts (method signatures + docstrings). Like querying a schema for the API.

        Pass a specific name (e.g., 'PromptStrategy') or leave empty for all contracts.
        A client can read these and write a conforming implementation on the fly."""
        return get_contracts_impl(contract_name or None)

    return mcp


if __name__ == "__main__":
    server = create_server()
    server.run()
