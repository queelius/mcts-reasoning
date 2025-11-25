#!/usr/bin/env python3
"""
Demo script showing the new diagnostic commands.

This simulates what you'd see in the TUI when using:
- /nodes
- /inspect <index>
- /path <index>
- /export-tree <file>
"""

from mcts_reasoning.compositional.providers import MockLLMProvider
from mcts_reasoning.reasoning import ReasoningMCTS
from mcts_reasoning.tui.session import SessionState
from mcts_reasoning.tui.commands import CommandHandler, CommandParser

def demo_diagnostic_commands():
    """Demo the diagnostic commands."""

    # Create a mock session with a small tree
    print("Creating MCTS tree with mock LLM...")
    print("=" * 70)

    provider = MockLLMProvider()
    mcts = (
        ReasoningMCTS()
        .with_llm(provider)
        .with_question("What are the prime numbers less than 20?")
        .with_exploration(1.414)
        .with_max_rollout_depth(3)
        .with_compositional_actions(enabled=True)
    )

    # Run a short search
    initial_state = "Question: What are the prime numbers less than 20?\n\nLet me solve this step by step."
    mcts.search(initial_state, simulations=5)

    print(f"Tree created with {len(mcts.get_all_nodes())} nodes")
    print()

    # Create session and command handler
    session = SessionState()
    session.mcts = mcts
    session.llm_provider = provider
    handler = CommandHandler(session)

    # Demo 1: /nodes
    print("=" * 70)
    print("Command: /nodes")
    print("=" * 70)
    cmd = CommandParser.parse("/nodes")
    success, msg = handler.execute(cmd)
    print(msg)
    print()

    # Demo 2: /inspect 3
    print("=" * 70)
    print("Command: /inspect 3")
    print("=" * 70)
    cmd = CommandParser.parse("/inspect 3")
    success, msg = handler.execute(cmd)
    print(msg)
    print()

    # Demo 3: /path 3
    print("=" * 70)
    print("Command: /path 3")
    print("=" * 70)
    cmd = CommandParser.parse("/path 3")
    success, msg = handler.execute(cmd)
    print(msg)
    print()

    # Demo 4: /export-tree
    print("=" * 70)
    print("Command: /export-tree demo_tree.json")
    print("=" * 70)
    cmd = CommandParser.parse("/export-tree demo_tree.json")
    success, msg = handler.execute(cmd)
    print(msg)
    print()

    # Show what's in the exported file
    import json
    with open('demo_tree.json', 'r') as f:
        tree_data = json.load(f)

    print("Exported tree structure:")
    print(f"  - Total nodes: {len(tree_data['node_list'])}")
    print(f"  - Max depth: {tree_data['stats']['max_depth']}")
    print(f"  - Best value: {tree_data['stats']['best_value']:.3f}")
    print()
    print("Node list preview:")
    for node_info in tree_data['node_list'][:5]:
        print(f"  Node {node_info['index']}: depth={node_info['depth']}, visits={node_info['visits']}, value={node_info['value']:.2f}")

    print()
    print("=" * 70)
    print("Demo complete! Try these commands in your TUI session:")
    print("  /nodes")
    print("  /inspect <index>")
    print("  /path <index>")
    print("  /export-tree <filename>")
    print("=" * 70)

if __name__ == "__main__":
    demo_diagnostic_commands()
