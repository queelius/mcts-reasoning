"""
Tree visualization for MCTS reasoning trees.

Renders SearchState trees as rich terminal output or structured JSON.
The tree is the primary computational artifact — this module makes
it inspectable.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .node import Node
from .types import SearchState


def render_tree(
    root: Node,
    max_state_length: int = 80,
    show_values: bool = True,
    show_state: bool = True,
    indent: str = "  ",
) -> str:
    """Render a tree as an indented string for terminal display.

    Args:
        root: Root node of the tree.
        max_state_length: Truncate state text to this length.
        show_values: Show value/visits on each node.
        show_state: Show the state text snippet.
        indent: Indentation string per level.

    Returns:
        Multi-line string representation of the tree.
    """
    lines: list[str] = []
    _render_node(root, lines, "", "", max_state_length, show_values, show_state, indent)
    return "\n".join(lines)


def _render_node(
    node: Node,
    lines: list[str],
    prefix: str,
    connector: str,
    max_len: int,
    show_values: bool,
    show_state: bool,
    indent: str,
) -> None:
    """Recursively render a node and its children."""
    # Extract action tag if present
    state_lines = str(node.state).split("\n")
    action_lines = [l.strip() for l in state_lines if l.strip().startswith("[")]
    action_tag = action_lines[-1].split("]")[0] + "]" if action_lines else ""

    # Build the node label
    parts = []
    if action_tag:
        parts.append(action_tag)

    if show_values:
        parts.append(f"v={node.value:.2f} n={node.visits}")

    if node.is_terminal:
        parts.append("[TERMINAL]")
        if node.answer:
            parts.append(f"-> {node.answer}")

    label = " ".join(parts) if parts else "(root)"

    # State snippet
    state_text = ""
    if show_state:
        # Get the last meaningful line of state (usually the latest reasoning)
        meaningful = [l.strip() for l in state_lines if l.strip() and not l.strip().startswith("Question:")]
        if meaningful:
            last = meaningful[-1]
            if len(last) > max_len:
                last = last[:max_len - 3] + "..."
            state_text = f'\n{prefix}{indent}  "{last}"'

    lines.append(f"{prefix}{connector}{label}{state_text}")

    # Children
    children = node.children
    for i, child in enumerate(children):
        is_last = (i == len(children) - 1)
        child_connector = "└─ " if is_last else "├─ "
        child_prefix = prefix + ("   " if is_last else "│  ")
        _render_node(child, lines, child_prefix, child_connector, max_len, show_values, show_state, indent)


def render_summary(state: SearchState) -> str:
    """Render a summary of a SearchState."""
    root = state.root
    lines = [
        f"Question: {state.question[:80]}",
        f"Simulations: {state.simulations_run}",
        f"Nodes: {root.count_nodes()}",
        f"Max depth: {root.max_depth()}",
        f"Terminals: {len(state.terminal_states)}",
        "",
    ]

    if state.terminal_states:
        lines.append("Answers found:")
        # Group by answer
        answer_counts: dict[str, list[float]] = {}
        for t in state.terminal_states:
            ans = t.get("answer", "None")
            score = t.get("score", 0.0)
            answer_counts.setdefault(ans, []).append(score)

        for ans, scores in sorted(answer_counts.items(), key=lambda x: -len(x[1])):
            avg = sum(scores) / len(scores)
            lines.append(f"  {ans:30s}  count={len(scores)}  avg_score={avg:.2f}")

    lines.append("")
    lines.append("Tree:")
    lines.append(render_tree(root))

    return "\n".join(lines)


def tree_to_dot(root: Node, max_label: int = 30) -> str:
    """Export tree as Graphviz DOT format for visual rendering.

    Usage: pipe output to `dot -Tpng -o tree.png` or `dot -Tsvg`.
    """
    lines = [
        "digraph mcts_tree {",
        '  rankdir=TB;',
        '  node [shape=box, style=rounded, fontsize=10, fontname="Courier"];',
        '  edge [color="#888888"];',
        "",
    ]

    def _node_id(node: Node) -> str:
        return f"n{id(node)}"

    def _add_node(node: Node) -> None:
        nid = _node_id(node)

        # Label
        state_lines = str(node.state).split("\n")
        action_lines = [l.strip() for l in state_lines if l.strip().startswith("[")]
        action = action_lines[-1].split("]")[0].replace("[", "") if action_lines else ""

        meaningful = [l.strip() for l in state_lines if l.strip() and not l.strip().startswith("Question:")]
        snippet = meaningful[-1][:max_label] if meaningful else ""
        snippet = snippet.replace('"', '\\"')

        label_parts = []
        if action:
            label_parts.append(action.upper())
        label_parts.append(f"v={node.value:.2f} n={node.visits}")
        if snippet:
            label_parts.append(snippet)
        if node.answer:
            label_parts.append(f"ANSWER: {node.answer}")

        label = "\\n".join(label_parts)

        # Style
        if node.is_terminal:
            color = "#4ade80" if node.value > 0.5 else "#f87171"
            lines.append(f'  {nid} [label="{label}", fillcolor="{color}22", style="rounded,filled"];')
        elif node.visits == 0:
            lines.append(f'  {nid} [label="{label}", color="#888888"];')
        else:
            lines.append(f'  {nid} [label="{label}"];')

        for child in node.children:
            lines.append(f'  {nid} -> {_node_id(child)};')
            _add_node(child)

    _add_node(root)
    lines.append("}")
    return "\n".join(lines)
