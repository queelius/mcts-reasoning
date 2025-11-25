# Tree Diagnostic Commands

New in this version: comprehensive tree inspection and debugging commands for the MCTS TUI.

## Overview

The MCTS reasoning tree can now be inspected in detail using four new commands:

1. `nodes` - List all nodes in the tree
2. `inspect <index>` - Show detailed information about a specific node
3. `path <index>` - Show the reasoning path from root to a node
4. `export-tree <file>` - Export the entire tree to JSON

## Command Reference

### `nodes` - List All Nodes

Shows a tabular view of all nodes in the tree with:
- Index (for use with `/inspect` and `/path`)
- Depth in the tree
- Visit count
- Total value accumulated
- Average value per visit
- Action taken to create this node

**Example output:**
```
Index  Depth  Visits   Value        Avg        Action
==========================================================================================
0      0      33       15.71        0.476      ROOT
1      1      32       15.32        0.479      Action(compare, correctness, systematic)
2      2      31       14.93        0.482      Action(abstract, solution, intuitive)
3      3      30       14.55        0.485      Action(decompose, goal, systematic)
```

**When to use:**
- Get an overview of the tree structure
- Identify which nodes were visited most (higher visit counts)
- Find nodes with high average values (promising reasoning paths)
- Get node indices for use with `/inspect` and `/path`

---

### `inspect <index>` - Inspect Node Details

Shows comprehensive information about a specific node:

- **Depth**: How far from the root
- **Visits**: Number of times visited during search
- **Total Value**: Cumulative value from all visits
- **Avg Value**: Quality score (higher = better reasoning)
- **UCB1**: Upper Confidence Bound (guides search selection)
- **Children**: Number of child nodes
- **Is Leaf**: Whether node can be expanded further
- **Action Taken**: The action that created this node
- **State**: The full reasoning text at this node (last 500 chars shown)

**Example:**
```bash
inspect 5
```

**Example output:**
```
Node 5 Details:
======================================================================
Depth:        3
Visits:       15
Total Value:  12.345
Avg Value:    0.823
UCB1:         0.891
Children:     2
Is Leaf:      False

Action Taken:
Action(decompose, structure, systematic)

State (1523 chars):
======================================================================
Question: What are the prime numbers less than 20?

Step 1: Define primality
A prime number is a natural number greater than 1 that has no
positive divisors other than 1 and itself...
[showing last 500 chars of accumulated reasoning]
======================================================================
```

**When to use:**
- Understand what action was taken at a specific node
- See the complete reasoning state at any point
- Compare UCB1 values to understand why certain nodes were explored
- Debug reasoning issues by examining state evolution

---

### `path <index>` - Show Reasoning Path

Shows the complete path from the root to a specific node, with:
- Each node along the path
- The action that led to each node
- Visit counts and values at each step
- New content added at each step (state diff)

**Example:**
```bash
path 7
```

**Example output:**
```
Path to Node 7 (4 nodes):

[0] ROOT
    visits=33, value=15.71, avg=0.476

  [1] Action(decompose, structure, systematic)
      visits=20, value=10.12, avg=0.506
      Added: Let me break down the problem systematically...

    [2] Action(analyze, correctness, critical)
        visits=12, value=6.45, avg=0.538
        Added: I need to verify each step for accuracy...

      [3] Action(synthesize, solution, systematic)
          visits=8, value=4.23, avg=0.529
          Added: Combining the analysis, the solution is...
```

**When to use:**
- Trace how a specific conclusion was reached
- Understand the sequence of actions in a reasoning path
- See how reasoning evolved step by step
- Debug why certain paths were preferred over others

---

### `export-tree <file>` - Export Tree to JSON

Exports the complete tree structure to a JSON file with:

- Full tree hierarchy with all nodes
- Configuration (exploration constant, max depth, etc.)
- Statistics (total nodes, max depth, best value)
- Node list with indices for easy lookup
- Metadata and timestamp

**Example:**
```bash
export-tree my_reasoning_session.json
```

**JSON structure:**
```json
{
  "root": {
    "state": "Question: ...",
    "visits": 33,
    "value": 15.71,
    "depth": 0,
    "action_taken": null,
    "children": [...]
  },
  "config": {
    "exploration_constant": 1.414,
    "max_rollout_depth": 5,
    "discount_factor": 0.95
  },
  "node_list": [
    {
      "index": 0,
      "depth": 0,
      "visits": 33,
      "value": 15.71,
      "avg_value": 0.476,
      "action": "ROOT",
      "state_length": 234,
      "num_children": 3
    },
    ...
  ],
  "stats": {
    "total_nodes": 33,
    "max_depth": 7,
    "best_value": 0.538
  },
  "timestamp": "2025-10-20T12:34:56.789"
}
```

**When to use:**
- Save reasoning sessions for later analysis
- Share trees with collaborators
- Create datasets for research
- Perform offline analysis with custom tools
- Visualize trees using external software

---

## Workflow Examples

### Example 1: Debugging Low-Quality Results

```bash
# Run search
ask What is the sum of primes less than 20?
search 50

# Check tree structure
nodes

# Find node with highest average value
# (Look for high "Avg" column in nodes output)

# Inspect the best node
inspect 12

# Trace how we got there
path 12

# Export for further analysis
export-tree prime_analysis.json
```

### Example 2: Understanding Action Selection

```bash
# After search
nodes

# Compare sibling nodes (same depth, different actions)
inspect 5
inspect 6
inspect 7

# See which action sequences lead to better results
path 5   # Best performing path
path 7   # Alternative path
```

### Example 3: Research Analysis

```bash
# Run experiment
ask [Your research question]
search 100

# Export detailed tree
export-tree experiment_1_tree.json

# Later, analyze with Python:
import json
with open('experiment_1_tree.json') as f:
    tree = json.load(f)

# Analyze node distribution, action frequencies, etc.
```

---

## Tips and Best Practices

### Understanding Node Values

- **High visits, low value**: Node was explored a lot but didn't lead to good results
- **Low visits, high value**: Promising but under-explored node
- **High UCB1**: Node will likely be explored next (high uncertainty or promise)

### Identifying Patterns

Use `nodes` to spot patterns:
- Are certain depths over-represented? (may indicate stuck reasoning)
- Do certain actions appear more frequently? (action bias)
- Is value increasing with depth? (good) or decreasing? (bad)

### Exporting for Visualization

The exported JSON can be processed with:
- Python libraries (networkx, graphviz)
- D3.js for interactive web visualizations
- Custom analysis scripts

Example Python visualization:
```python
import json
import networkx as nx
import matplotlib.pyplot as plt

with open('tree.json') as f:
    data = json.load(f)

# Build graph from node list
G = nx.DiGraph()
for node in data['node_list']:
    G.add_node(node['index'],
               visits=node['visits'],
               value=node['avg_value'])

# Draw
nx.draw(G, with_labels=True)
plt.show()
```

---

## API Access (Python)

These commands are also available programmatically:

```python
from mcts_reasoning.reasoning import ReasoningMCTS

# After running search
mcts = ReasoningMCTS()
# ... search ...

# Get all nodes
nodes = mcts.get_all_nodes()

# Get specific node
node = mcts.get_node_by_index(5)

# Get node details
details = mcts.get_node_details(node)

# Get path to node
path = node.path_to_root

# Export tree
tree_json = mcts.to_json()
```

---

## See Also

- [TUI Guide](../guides/tui-guide.md) - Complete TUI command reference
- [MCP Integration](../advanced/mcp-integration.md) - Using tools with MCTS
- [Prompt Features](prompt-features.md) - Compositional prompting details
