"""
Core MCTS implementation with fluent API and JSON serialization.
"""

import json
import math
import random
from typing import Optional, List, Dict, Any, Tuple, Union
from dataclasses import dataclass, field, asdict
from pathlib import Path
from datetime import datetime


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""
    state: str
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    visits: int = 0
    value: float = 0.0
    action_taken: Optional[Any] = None
    
    @property
    def is_leaf(self) -> bool:
        """Check if this is a leaf node."""
        return len(self.children) == 0
    
    @property
    def is_root(self) -> bool:
        """Check if this is the root node."""
        return self.parent is None
    
    @property
    def depth(self) -> int:
        """Get depth of this node."""
        return 0 if self.is_root else self.parent.depth + 1
    
    @property
    def path_to_root(self) -> List['MCTSNode']:
        """Get path from this node to root."""
        path = []
        node = self
        while node:
            path.append(node)
            node = node.parent
        return list(reversed(path))
    
    def ucb1(self, exploration_constant: float = 1.414) -> float:
        """Calculate UCB1 value for this node."""
        if self.visits == 0:
            return float('inf')
        if self.parent is None:
            return self.value
        
        exploitation = self.value / self.visits
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary (for JSON serialization)."""
        return {
            'state': self.state,
            'visits': self.visits,
            'value': self.value,
            'action_taken': str(self.action_taken) if self.action_taken else None,
            'children': [child.to_dict() for child in self.children]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], parent: Optional['MCTSNode'] = None) -> 'MCTSNode':
        """Create node from dictionary."""
        node = cls(
            state=data['state'],
            parent=parent,
            visits=data['visits'],
            value=data['value'],
            action_taken=data.get('action_taken')
        )
        
        # Recursively create children
        for child_data in data.get('children', []):
            child = cls.from_dict(child_data, parent=node)
            node.children.append(child)
        
        return node


class MCTS:
    """
    Monte Carlo Tree Search with fluent API and JSON serialization.
    
    Example:
        mcts = (
            MCTS()
            .with_llm(llm_adapter)
            .with_exploration(1.414)
            .search("initial state", simulations=100)
        )
        
        # Get results
        best_path = mcts.best_path
        solution = mcts.solution
        
        # Save/load
        mcts.save("checkpoint.json")
        mcts2 = MCTS.load("checkpoint.json")
    """
    
    def __init__(self):
        self.root: Optional[MCTSNode] = None
        self.llm = None
        self.exploration_constant = 1.414
        self.max_rollout_depth = 5
        self.discount_factor = 0.95
        self._metadata = {}
        
    # ========== Fluent Configuration Methods ==========
    
    def with_llm(self, llm) -> 'MCTS':
        """Set the LLM adapter."""
        self.llm = llm
        return self
    
    def with_exploration(self, constant: float) -> 'MCTS':
        """Set exploration constant for UCB1."""
        self.exploration_constant = constant
        return self
    
    def with_max_rollout_depth(self, depth: int) -> 'MCTS':
        """Set maximum rollout depth."""
        self.max_rollout_depth = depth
        return self
    
    def with_discount(self, factor: float) -> 'MCTS':
        """Set discount factor for rewards."""
        self.discount_factor = factor
        return self
    
    def with_metadata(self, **kwargs) -> 'MCTS':
        """Add metadata (for tracking/debugging)."""
        self._metadata.update(kwargs)
        return self
    
    # ========== Core MCTS Methods ==========
    
    def search(self, initial_state: str, simulations: int = 100) -> 'MCTS':
        """
        Run MCTS search.
        
        Args:
            initial_state: Starting state string
            simulations: Number of simulations to run
            
        Returns:
            self (for chaining)
        """
        if self.llm is None:
            raise ValueError("LLM not set. Use .with_llm() first.")
        
        # Initialize root if needed
        if self.root is None:
            self.root = MCTSNode(state=initial_state)
        
        # Run simulations
        for _ in range(simulations):
            self._simulate()
        
        return self
    
    def _simulate(self):
        """Run one MCTS simulation."""
        # Selection
        node = self._select()
        
        # Expansion
        if not self._is_terminal(node) and node.visits > 0:
            node = self._expand(node)
        
        # Rollout
        reward = self._rollout(node)
        
        # Backpropagation
        self._backpropagate(node, reward)
    
    def _select(self) -> MCTSNode:
        """Select node to expand using UCB1."""
        node = self.root
        
        while node.children:
            # Select child with highest UCB1 value
            node = max(node.children, 
                      key=lambda n: n.ucb1(self.exploration_constant))
        
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand node by adding one child."""
        # Get possible actions (to be implemented by subclass)
        actions = self._get_actions(node.state)
        
        if not actions:
            return node
        
        # Add one random unexplored action
        action = random.choice(actions)
        new_state = self._take_action(node.state, action)
        
        child = MCTSNode(
            state=new_state,
            parent=node,
            action_taken=action
        )
        node.children.append(child)
        
        return child
    
    def _rollout(self, node: MCTSNode) -> float:
        """Rollout from node to estimate value."""
        state = node.state
        depth = 0
        
        while depth < self.max_rollout_depth and not self._is_terminal_state(state):
            actions = self._get_actions(state)
            if not actions:
                break
            
            action = random.choice(actions)
            state = self._take_action(state, action)
            depth += 1
        
        # Evaluate terminal state
        value = self._evaluate_state(state)
        
        # Apply discount based on depth
        return value * (self.discount_factor ** depth)
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Update statistics up the tree."""
        while node:
            node.visits += 1
            node.value += reward
            node = node.parent
    
    # ========== Methods to Override ==========
    
    def _get_actions(self, state: str) -> List[Any]:
        """Get possible actions from state. Override in subclass."""
        return ["action1", "action2", "action3"]
    
    def _take_action(self, state: str, action: Any) -> str:
        """Apply action to state. Override in subclass."""
        if self.llm:
            prompt = f"State: {state}\nAction: {action}\nNext state:"
            return self.llm.generate(prompt, max_tokens=150)
        return state + f"\n{action}"
    
    def _is_terminal(self, node: MCTSNode) -> bool:
        """Check if node is terminal. Override in subclass."""
        return self._is_terminal_state(node.state)
    
    def _is_terminal_state(self, state: str) -> bool:
        """Check if state is terminal. Override in subclass."""
        return False  # Default: never terminal
    
    def _evaluate_state(self, state: str) -> float:
        """Evaluate a state. Override in subclass."""
        if self.llm:
            prompt = f"Evaluate this state (0-1):\n{state}\nScore:"
            try:
                response = self.llm.generate(prompt, max_tokens=10)
                return float(response.strip())
            except:
                return 0.5
        return random.random()
    
    # ========== Property Access ==========
    
    @property
    def best_node(self) -> Optional[MCTSNode]:
        """Get best child of root (most visits)."""
        if not self.root or not self.root.children:
            return None
        return max(self.root.children, key=lambda n: n.visits)
    
    @property
    def best_value(self) -> float:
        """Get value of best path."""
        node = self.best_node
        return node.value / node.visits if node and node.visits > 0 else 0.0
    
    @property
    def best_path(self) -> List[Tuple[Any, str]]:
        """Get best path as list of (action, state) tuples."""
        path = []
        node = self.root
        
        while node and node.children:
            # Follow most visited child
            node = max(node.children, key=lambda n: n.visits)
            path.append((node.action_taken, node.state))
        
        return path
    
    @property
    def solution(self) -> str:
        """Extract solution from best path."""
        if not self.best_path:
            return self.root.state if self.root else ""
        
        # Return final state of best path
        return self.best_path[-1][1]
    
    @property
    def stats(self) -> Dict[str, Any]:
        """Get tree statistics."""
        if not self.root:
            return {}
        
        def count_nodes(node):
            return 1 + sum(count_nodes(c) for c in node.children)
        
        def max_depth(node, d=0):
            if not node.children:
                return d
            return max(max_depth(c, d+1) for c in node.children)
        
        return {
            'total_nodes': count_nodes(self.root),
            'max_depth': max_depth(self.root),
            'root_visits': self.root.visits,
            'root_value': self.root.value,
            'best_value': self.best_value,
            'num_children': len(self.root.children),
            'metadata': self._metadata
        }
    
    # ========== Serialization ==========
    
    def to_json(self) -> Dict[str, Any]:
        """Convert entire tree to JSON-serializable dict."""
        return {
            'root': self.root.to_dict() if self.root else None,
            'config': {
                'exploration_constant': self.exploration_constant,
                'max_rollout_depth': self.max_rollout_depth,
                'discount_factor': self.discount_factor,
            },
            'metadata': self._metadata,
            'timestamp': datetime.now().isoformat(),
            'stats': self.stats
        }
    
    @classmethod
    def from_json(cls, data: Dict[str, Any]) -> 'MCTS':
        """Create MCTS instance from JSON dict."""
        mcts = cls()
        
        # Restore config
        config = data.get('config', {})
        mcts.exploration_constant = config.get('exploration_constant', 1.414)
        mcts.max_rollout_depth = config.get('max_rollout_depth', 5)
        mcts.discount_factor = config.get('discount_factor', 0.95)
        
        # Restore metadata
        mcts._metadata = data.get('metadata', {})
        
        # Restore tree
        if data.get('root'):
            mcts.root = MCTSNode.from_dict(data['root'])
        
        return mcts
    
    def save(self, filepath: Union[str, Path]) -> 'MCTS':
        """Save tree to JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(self.to_json(), f, indent=2)
        
        return self
    
    @classmethod
    def load(cls, filepath: Union[str, Path]) -> 'MCTS':
        """Load tree from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_json(data)
    
    # ========== Context Manager Support ==========
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit (for cleanup)."""
        # Could close IPC connections, save checkpoints, etc.
        pass
    
    # ========== String Representation ==========
    
    def __repr__(self):
        if not self.root:
            return "MCTS(uninitialized)"
        
        stats = self.stats
        return (f"MCTS(nodes={stats['total_nodes']}, "
                f"depth={stats['max_depth']}, "
                f"best_value={stats['best_value']:.3f})")
    
    def __str__(self):
        """Pretty string representation."""
        if not self.root:
            return "Empty MCTS tree"
        
        lines = ["MCTS Tree:"]
        stats = self.stats
        lines.append(f"  Nodes: {stats['total_nodes']}")
        lines.append(f"  Max depth: {stats['max_depth']}")
        lines.append(f"  Best value: {stats['best_value']:.3f}")
        
        if self.best_path:
            lines.append(f"  Best path length: {len(self.best_path)}")
        
        return "\n".join(lines)