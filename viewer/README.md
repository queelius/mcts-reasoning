# MCTS Live Tree Viewer

Real-time visualization of Monte Carlo Tree Search with compositional actions.

## Features

- **Real-time Updates**: Watch the MCTS tree grow as it explores
- **Phase Visualization**: See Selection, Expansion, Rollout, and Backpropagation phases
- **Interactive Tree**: Click nodes to see details (state, value, visits, actions)
- **Value Heatmap**: Nodes colored from green (high value) to red (low value)
- **Statistics Panel**: Track simulations, nodes, and best path
- **Value History Chart**: See how node values evolve over time

## Architecture

```
MCTS Process (Python)          Viewer Server (Python)         Browser (JS)
     |                               |                            |
     |--[TCP/IPC Events]----------->|                            |
     |  (port 9999)                  |--[WebSocket]------------->|
     |                               |  (port 8000)              |
     |                               |                            |
  [Events]                      [Processes]                 [Displays]
  - node.created                - Parse events              - vis.js tree
  - node.updated                - Update tree               - Statistics
  - phase.start/end             - Broadcast                 - Node details
  - action.executed             - Store history             - Value chart
```

## Quick Start

### Option 1: Using the launcher (recommended)
```bash
cd integrations/mcts_live_viewer
./launch.sh
```

### Option 2: Manual setup
```bash
# Terminal 1: Start the viewer server
cd integrations/mcts_live_viewer
python server.py

# Terminal 2: Run MCTS with IPC
python test_mcts_with_viewer.py --test math
```

Then open http://localhost:8000 in your browser.

## Test Options

The test script provides different scenarios:

```bash
# Math problem (quick, clear solution)
python test_mcts_with_viewer.py --test math

# Logic puzzle (requires reasoning)
python test_mcts_with_viewer.py --test logic

# Compositional action exploration
python test_mcts_with_viewer.py --test compositional

# Run all tests
python test_mcts_with_viewer.py --test all
```

## Understanding the Visualization

### Node Appearance
- **Size**: Proportional to visit count (more visits = larger node)
- **Color**: Value-based gradient
  - 🟢 Green: High value (good paths)
  - 🟡 Yellow: Medium value
  - 🔴 Red: Low value (poor paths)
- **Labels**: Show action taken or "ROOT" for initial node

### MCTS Phases
Watch the current phase indicator:
1. **Selection** (🔍): Traversing tree using UCB1
2. **Expansion** (➕): Adding new child node
3. **Rollout** (🎲): Simulating to terminal state
4. **Backpropagation** (⬆️): Updating statistics

### Statistics Panel
- **Simulations**: Total MCTS iterations completed
- **Nodes**: Total nodes in the tree
- **Max Depth**: Deepest path explored
- **Avg Value**: Average value across all nodes

### Node Info Panel
Click any node to see:
- Node ID and depth
- State preview (first/last 200 chars)
- Action that led to this node
- Visit count and value
- Parent and children information

## Compositional Actions

The system uses compositional prompting with 5 dimensions:
- **Operation**: analyze, decompose, solve, verify, synthesize
- **Focus**: problem, solution, assumptions, constraints, approach
- **Style**: systematic, intuitive, formal

This creates a rich action space for exploration.

## Customization

### Adjust MCTS Parameters
In `test_mcts_with_viewer.py`:
```python
mcts = ReasoningMCTSWithIPC(
    exploration_constant=1.414,  # UCB1 exploration
    max_rollout_depth=5,         # Rollout depth limit
    use_compositional=True,      # Enable compositional actions
)
```

### Change Visualization Layout
In `server.py`:
```python
layout: {
    hierarchical: {
        direction: 'UD',        # UD, DU, LR, RL
        nodeSpacing: 150,       # Space between nodes
        levelSeparation: 150,   # Space between levels
    }
}
```

## Troubleshooting

### "Connection refused" error
- Ensure viewer server is running first
- Check ports 8000 (HTTP) and 9999 (TCP) are free

### Tree not updating
- Verify IPC is enabled in MCTS: `enable_ipc=True`
- Check browser console for WebSocket errors

### Nodes overlapping
- Adjust `nodeSpacing` in server.py
- Use physics simulation: set `physics: { enabled: true }`

## Implementation Notes

### IPC Protocol
Events are sent as length-prefixed JSON:
```
[4 bytes: message length][JSON message]
```

### Event Types
- `mcts.initialized`: Search started
- `node.created`: New node added
- `node.updated`: Node statistics changed  
- `phase.*`: MCTS phase transitions
- `action.executing`: Action being applied
- `terminal.detected`: Terminal state reached

### Memory Management
- Tree pruning for large graphs (>1000 nodes)
- Event buffer limits (last 1000 events)
- State preview truncation (200 chars)