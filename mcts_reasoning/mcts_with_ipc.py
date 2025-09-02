"""
MCTS with IPC Support for Live Visualization

Extends the clean MCTS implementation with IPC events for real-time monitoring.
"""

import json
import socket
import struct
from typing import Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass, asdict

from .mcts_core import MCTS, MCTSNode
from .reasoning_mcts import ReasoningMCTS, CompositionalAction


@dataclass
class MCTSEvent:
    """An event in the MCTS process"""
    event_type: str
    timestamp: float
    data: Dict[str, Any]
    
    def to_json(self) -> str:
        return json.dumps({
            'event_type': self.event_type,
            'timestamp': self.timestamp,
            'data': self.data
        })


class IPCTransport:
    """Simple TCP transport for IPC"""
    
    def __init__(self, host: str = "localhost", port: int = 9999):
        self.host = host
        self.port = port
        self.socket = None
        self.connected = False
    
    def connect(self) -> bool:
        """Connect to IPC server"""
        try:
            self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.socket.connect((self.host, self.port))
            self.connected = True
            return True
        except Exception as e:
            print(f"IPC connection failed: {e}")
            self.connected = False
            return False
    
    def send_event(self, event: MCTSEvent) -> bool:
        """Send event via IPC"""
        if not self.connected:
            return False
        
        try:
            # Serialize event
            data = event.to_json().encode('utf-8')
            
            # Send length prefix then data
            length = struct.pack('!I', len(data))
            self.socket.sendall(length + data)
            return True
        except Exception as e:
            print(f"IPC send error: {e}")
            self.connected = False
            return False
    
    def close(self):
        """Close connection"""
        if self.socket:
            self.socket.close()
            self.connected = False


class MCTSWithIPC(MCTS):
    """
    MCTS with IPC event broadcasting.
    
    Sends events for:
    - Tree initialization
    - Node selection
    - Node expansion
    - Rollout start/end
    - Backpropagation
    - Best path updates
    """
    
    def __init__(
        self,
        exploration_constant: float = 1.414,
        max_rollout_depth: int = 10,
        ipc_host: str = "localhost",
        ipc_port: int = 9999,
        enable_ipc: bool = True
    ):
        super().__init__(exploration_constant, max_rollout_depth)
        
        self.enable_ipc = enable_ipc
        self.ipc = None
        
        if enable_ipc:
            self.ipc = IPCTransport(ipc_host, ipc_port)
            if self.ipc.connect():
                print(f"📡 IPC connected to {ipc_host}:{ipc_port}")
            else:
                print("⚠️ IPC connection failed, continuing without visualization")
                self.enable_ipc = False
    
    def search(self, initial_state: str, num_simulations: int = 1000) -> MCTSNode:
        """Run MCTS search with IPC events"""
        
        # Send initialization event
        self._send_event("mcts.initialized", {
            "initial_state": initial_state[:200],  # First 200 chars
            "num_simulations": num_simulations,
            "exploration_constant": self.exploration_constant
        })
        
        # Run normal search
        result = super().search(initial_state, num_simulations)
        
        # Send completion event
        self._send_event("mcts.completed", {
            "total_nodes": self._count_nodes(result),
            "root_value": result.value,
            "root_visits": result.visits
        })
        
        return result
    
    def _simulate(self):
        """Run one simulation with IPC events"""
        
        # Selection phase
        self._send_event("phase.selection_start", {})
        node = self._select()
        self._send_event("phase.selection_end", {
            "selected_node_id": id(node),
            "node_value": node.value,
            "node_visits": node.visits
        })
        
        # Expansion phase
        if not self.is_terminal(node.state) and not node.is_fully_expanded:
            self._send_event("phase.expansion_start", {
                "parent_node_id": id(node)
            })
            node = self._expand(node)
            self._send_event("phase.expansion_end", {
                "new_node_id": id(node),
                "action": str(node.action_taken) if node.action_taken else None
            })
        
        # Simulation phase (rollout)
        self._send_event("phase.rollout_start", {
            "start_node_id": id(node),
            "depth": self._get_depth(node)
        })
        reward = self._rollout(node)
        self._send_event("phase.rollout_end", {
            "reward": reward,
            "terminal_reached": self.is_terminal(node.state)
        })
        
        # Backpropagation phase
        self._send_event("phase.backprop_start", {
            "start_node_id": id(node),
            "reward": reward
        })
        self._backpropagate(node, reward)
        self._send_event("phase.backprop_end", {})
    
    def _expand(self, node: MCTSNode) -> MCTSNode:
        """Expand with IPC events"""
        child = super()._expand(node)
        
        # Send node creation event
        self._send_event("node.created", {
            "node_id": id(child),
            "parent_id": id(node),
            "action": str(child.action_taken) if child.action_taken else None,
            "state_preview": child.state[:100],  # First 100 chars
            "depth": self._get_depth(child)
        })
        
        return child
    
    def _backpropagate(self, node: MCTSNode, reward: float):
        """Backpropagate with IPC events"""
        super()._backpropagate(node, reward)
        
        # Send value update events
        current = node
        while current is not None:
            self._send_event("node.updated", {
                "node_id": id(current),
                "visits": current.visits,
                "value": current.value,
                "total_reward": current.total_reward
            })
            current = current.parent
    
    def _send_event(self, event_type: str, data: Dict[str, Any]):
        """Send IPC event"""
        if not self.enable_ipc or not self.ipc:
            return
        
        event = MCTSEvent(
            event_type=event_type,
            timestamp=datetime.now().timestamp(),
            data=data
        )
        
        self.ipc.send_event(event)
    
    def _get_depth(self, node: MCTSNode) -> int:
        """Get depth of node in tree"""
        depth = 0
        current = node
        while current.parent is not None:
            depth += 1
            current = current.parent
        return depth
    
    def _count_nodes(self, node: MCTSNode) -> int:
        """Count total nodes in subtree"""
        count = 1
        for child in node.children:
            count += self._count_nodes(child)
        return count
    
    def close(self):
        """Close IPC connection"""
        if self.ipc:
            self.ipc.close()


class ReasoningMCTSWithIPC(ReasoningMCTS, MCTSWithIPC):
    """
    Reasoning MCTS with IPC support.
    
    Combines compositional actions with live visualization.
    """
    
    def __init__(
        self,
        llm_client,
        original_question: str,
        exploration_constant: float = 1.414,
        max_rollout_depth: int = 5,
        use_compositional: bool = True,
        ipc_host: str = "localhost",
        ipc_port: int = 9999,
        enable_ipc: bool = True
    ):
        # Initialize ReasoningMCTS part
        ReasoningMCTS.__init__(
            self,
            llm_client=llm_client,
            original_question=original_question,
            exploration_constant=exploration_constant,
            max_rollout_depth=max_rollout_depth,
            use_compositional=use_compositional
        )
        
        # Initialize IPC part
        self.enable_ipc = enable_ipc
        self.ipc = None
        
        if enable_ipc:
            self.ipc = IPCTransport(ipc_host, ipc_port)
            if self.ipc.connect():
                print(f"📡 IPC connected to {ipc_host}:{ipc_port}")
                # Send initialization with question
                self._send_event("reasoning.initialized", {
                    "question": original_question,
                    "use_compositional": use_compositional,
                    "action_space_size": len(self.get_actions(""))
                })
            else:
                print("⚠️ IPC connection failed, continuing without visualization")
                self.enable_ipc = False
    
    def take_action(self, state: str, action: CompositionalAction) -> str:
        """Take action with IPC event"""
        # Send action event
        self._send_event("action.executing", {
            "action": str(action),
            "operation": action.operation.value,
            "focus": action.focus.value,
            "style": action.style.value
        })
        
        # Execute action
        new_state = super().take_action(state, action)
        
        # Send result event
        self._send_event("action.completed", {
            "action": str(action),
            "state_length": len(new_state),
            "new_content": new_state[len(state):][:200]  # New content (first 200 chars)
        })
        
        return new_state
    
    def is_terminal(self, state: str) -> bool:
        """Check terminal with IPC event"""
        is_term = super().is_terminal(state)
        
        if is_term:
            self._send_event("terminal.detected", {
                "state_preview": state[-200:],  # Last 200 chars
                "reason": "solution_found" if "answer" in state.lower() else "other"
            })
        
        return is_term
    
    def evaluate_terminal(self, state: str) -> float:
        """Evaluate terminal with IPC event"""
        value = super().evaluate_terminal(state)
        
        self._send_event("terminal.evaluated", {
            "value": value,
            "state_length": len(state)
        })
        
        return value