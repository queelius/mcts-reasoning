"""
Clean, Strict MCTS Implementation for LLM Reasoning

This is a proper MCTS with the four canonical phases:
1. Selection - UCB1 to traverse tree
2. Expansion - Add one child node
3. Simulation - Rollout to terminal state
4. Backpropagation - Update statistics

No unnecessary complications.
"""

import math
import random
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass, field


@dataclass
class MCTSNode:
    """A node in the MCTS tree."""
    
    # State
    state: str  # The reasoning state at this node
    action_taken: Optional[Any] = None  # Action that led to this node
    
    # Tree structure
    parent: Optional['MCTSNode'] = None
    children: List['MCTSNode'] = field(default_factory=list)
    untried_actions: List[Any] = field(default_factory=list)
    
    # MCTS statistics
    visits: int = 0
    total_reward: float = 0.0
    
    @property
    def value(self) -> float:
        """Average reward (Q-value)"""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits
    
    @property
    def is_fully_expanded(self) -> bool:
        """Check if all actions have been tried"""
        return len(self.untried_actions) == 0
    
    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal node"""
        # Will be determined by the MCTS class
        return False
    
    def ucb1(self, exploration_constant: float = math.sqrt(2)) -> float:
        """Upper Confidence Bound formula for selection"""
        if self.visits == 0:
            return float('inf')
        
        exploitation = self.value
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits) / self.visits
        )
        return exploitation + exploration
    
    def best_child(self, exploration_constant: float = math.sqrt(2)) -> 'MCTSNode':
        """Select best child using UCB1"""
        return max(self.children, key=lambda c: c.ucb1(exploration_constant))
    
    def add_child(self, state: str, action: Any) -> 'MCTSNode':
        """Add a child node"""
        child = MCTSNode(
            state=state,
            action_taken=action,
            parent=self,
            untried_actions=[]  # Will be filled by MCTS class
        )
        self.children.append(child)
        self.untried_actions.remove(action)
        return child
    
    def update(self, reward: float):
        """Update node statistics"""
        self.visits += 1
        self.total_reward += reward
    
    def get_path_to_root(self) -> List['MCTSNode']:
        """Get path from this node to root"""
        path = []
        node = self
        while node is not None:
            path.append(node)
            node = node.parent
        return list(reversed(path))


class MCTS:
    """
    Strict Monte Carlo Tree Search implementation.
    
    This is domain-agnostic - subclasses implement:
    - get_actions(state) -> List[action]
    - take_action(state, action) -> new_state
    - is_terminal(state) -> bool
    - evaluate_terminal(state) -> reward
    - rollout_policy(state) -> action
    """
    
    def __init__(
        self,
        exploration_constant: float = math.sqrt(2),
        max_rollout_depth: int = 10
    ):
        self.exploration_constant = exploration_constant
        self.max_rollout_depth = max_rollout_depth
        self.root: Optional[MCTSNode] = None
    
    def search(
        self,
        initial_state: str,
        num_simulations: int = 1000
    ) -> MCTSNode:
        """
        Run MCTS search from initial state.
        
        Returns the root node with a built tree.
        """
        # Initialize root
        self.root = MCTSNode(
            state=initial_state,
            untried_actions=self.get_actions(initial_state)
        )
        
        # Run simulations
        for _ in range(num_simulations):
            self._simulate()
        
        return self.root
    
    def _simulate(self):
        """Run one simulation of MCTS"""
        
        # 1. Selection: Traverse tree using UCB1
        node = self._select()
        
        # 2. Expansion: Add new child if not terminal
        if not self.is_terminal(node.state) and node.is_fully_expanded:
            node = self._expand(node)
        
        # 3. Simulation: Rollout to terminal state
        reward = self._rollout(node)
        
        # 4. Backpropagation: Update statistics
        self._backpropagate(node, reward)
    
    def _select(self) -> MCTSNode:
        """
        Selection phase: Navigate to a leaf using UCB1.
        
        Returns a node that is either:
        - Not fully expanded (has untried actions)
        - Terminal
        """
        node = self.root
        
        while node.is_fully_expanded and not self.is_terminal(node.state):
            if not node.children:
                break
            node = node.best_child(self.exploration_constant)
        
        return node
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """
        Expansion phase: Add one new child.
        
        Picks a random untried action and creates a child.
        """
        if len(node.untried_actions) == 0:
            # Get available actions for this state
            node.untried_actions = self.get_actions(node.state)
        
        if len(node.untried_actions) == 0:
            return node  # No actions available
        
        # Pick random untried action
        action = random.choice(node.untried_actions)
        
        # Take action to get new state
        new_state = self.take_action(node.state, action)
        
        # Add child
        child = node.add_child(new_state, action)
        
        # Set child's untried actions
        if not self.is_terminal(new_state):
            child.untried_actions = self.get_actions(new_state)
        
        return child
    
    def _rollout(self, node: MCTSNode) -> float:
        """
        Simulation phase: Random rollout to terminal state.
        
        Uses rollout_policy to select actions.
        """
        state = node.state
        depth = 0
        
        while not self.is_terminal(state) and depth < self.max_rollout_depth:
            # Get action from rollout policy
            action = self.rollout_policy(state)
            if action is None:
                break
            
            # Take action
            state = self.take_action(state, action)
            depth += 1
        
        # Evaluate terminal state
        return self.evaluate_terminal(state)
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """
        Backpropagation phase: Update statistics up the tree.
        """
        while node is not None:
            node.update(reward)
            node = node.parent
    
    def get_best_action(self) -> Any:
        """Get the best action from root based on visit counts"""
        if not self.root or not self.root.children:
            return None
        
        # Choose child with most visits (most robust)
        best_child = max(self.root.children, key=lambda c: c.visits)
        return best_child.action_taken
    
    def get_best_path(self) -> List[Tuple[Any, str]]:
        """Get the best path through the tree"""
        path = []
        node = self.root
        
        while node.children:
            best_child = max(node.children, key=lambda c: c.visits)
            path.append((best_child.action_taken, best_child.state))
            node = best_child
        
        return path
    
    # Abstract methods to be implemented by subclasses
    
    def get_actions(self, state: str) -> List[Any]:
        """Get available actions for a state"""
        raise NotImplementedError
    
    def take_action(self, state: str, action: Any) -> str:
        """Apply action to state, return new state"""
        raise NotImplementedError
    
    def is_terminal(self, state: str) -> bool:
        """Check if state is terminal"""
        raise NotImplementedError
    
    def evaluate_terminal(self, state: str) -> float:
        """Evaluate a terminal state"""
        raise NotImplementedError
    
    def rollout_policy(self, state: str) -> Any:
        """Select action during rollout (can be random or guided)"""
        raise NotImplementedError