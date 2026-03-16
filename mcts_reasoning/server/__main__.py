"""Allow running the MCP server via ``python -m mcts_reasoning.server``."""

from . import create_server

if __name__ == "__main__":
    server = create_server()
    server.run()
