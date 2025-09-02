"""
Sampling strategies for MCTS trees.

Provides various methods to sample paths from MCTS trees:
- Temperature-based sampling
- Visit-based sampling
- Top-K sampling
- Diverse sampling
- Consistency checking
"""

import math
import random
from typing import List, Tuple, Dict, Any, Optional, Set
from collections import Counter
from dataclasses import dataclass

from .core import MCTS, MCTSNode


@dataclass
class SampledPath:
    """A sampled path from the MCTS tree."""
    nodes: List[MCTSNode]
    actions: List[Any]
    states: List[str]
    total_value: float
    total_visits: int
    
    @property
    def final_state(self) -> str:
        """Get the final state of the path."""
        return self.states[-1] if self.states else ""
    
    @property
    def length(self) -> int:
        """Get path length."""
        return len(self.nodes)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'actions': [str(a) for a in self.actions],
            'final_state': self.final_state,
            'length': self.length,
            'value': self.total_value,
            'visits': self.total_visits
        }


class MCTSSampler:
    """Sampling strategies for MCTS trees."""
    
    def __init__(self, mcts: MCTS):
        self.mcts = mcts
        self.root = mcts.root
    
    def sample_by_value(self, temperature: float = 1.0, 
                       from_node: Optional[MCTSNode] = None) -> SampledPath:
        """
        Sample a path using softmax over node values.
        
        Args:
            temperature: Controls randomness (0 = greedy, ∞ = uniform)
            from_node: Starting node (default: root)
            
        Returns:
            SampledPath object
        """
        node = from_node or self.root
        if not node:
            raise ValueError("No tree to sample from")
        
        path_nodes = [node]
        path_actions = []
        path_states = [node.state]
        
        while node.children:
            # Calculate softmax probabilities
            values = [
                (child.value / child.visits if child.visits > 0 else 0.0)
                for child in node.children
            ]
            
            if temperature == 0:
                # Greedy selection
                idx = values.index(max(values))
            else:
                # Softmax sampling
                exp_values = [math.exp(v / temperature) for v in values]
                total = sum(exp_values)
                probs = [e / total for e in exp_values]
                
                # Sample child
                idx = random.choices(range(len(node.children)), weights=probs)[0]
            
            node = node.children[idx]
            path_nodes.append(node)
            path_actions.append(node.action_taken)
            path_states.append(node.state)
        
        # Calculate path statistics
        total_value = sum(n.value for n in path_nodes)
        total_visits = sum(n.visits for n in path_nodes)
        
        return SampledPath(
            nodes=path_nodes,
            actions=path_actions,
            states=path_states,
            total_value=total_value,
            total_visits=total_visits
        )
    
    def sample_by_visits(self, from_node: Optional[MCTSNode] = None) -> SampledPath:
        """
        Sample a path proportional to visit counts (MCTS-style).
        
        This is what AlphaGo uses after search completes.
        
        Args:
            from_node: Starting node (default: root)
            
        Returns:
            SampledPath object
        """
        node = from_node or self.root
        if not node:
            raise ValueError("No tree to sample from")
        
        path_nodes = [node]
        path_actions = []
        path_states = [node.state]
        
        while node.children:
            # Sample proportional to visits
            visits = [child.visits for child in node.children]
            total_visits = sum(visits)
            
            if total_visits == 0:
                # Uniform if no visits
                node = random.choice(node.children)
            else:
                probs = [v / total_visits for v in visits]
                idx = random.choices(range(len(node.children)), weights=probs)[0]
                node = node.children[idx]
            
            path_nodes.append(node)
            path_actions.append(node.action_taken)
            path_states.append(node.state)
        
        # Calculate path statistics
        total_value = sum(n.value for n in path_nodes)
        total_visits = sum(n.visits for n in path_nodes)
        
        return SampledPath(
            nodes=path_nodes,
            actions=path_actions,
            states=path_states,
            total_value=total_value,
            total_visits=total_visits
        )
    
    def sample_top_k(self, k: int = 5, 
                    criterion: str = "value") -> List[SampledPath]:
        """
        Get top-K paths based on criterion.
        
        Args:
            k: Number of paths to return
            criterion: "value", "visits", or "depth"
            
        Returns:
            List of top-K paths
        """
        # Collect all paths to leaves
        all_paths = self._get_all_paths(self.root)
        
        # Sort by criterion
        if criterion == "value":
            key_func = lambda p: p.total_value / max(p.total_visits, 1)
        elif criterion == "visits":
            key_func = lambda p: p.total_visits
        elif criterion == "depth":
            key_func = lambda p: p.length
        else:
            raise ValueError(f"Unknown criterion: {criterion}")
        
        # Sort and return top-K
        sorted_paths = sorted(all_paths, key=key_func, reverse=True)
        return sorted_paths[:k]
    
    def sample_diverse(self, n: int = 5, 
                      min_distance: float = 0.3,
                      temperature: float = 1.0) -> List[SampledPath]:
        """
        Sample diverse paths (syntactically different).
        
        Args:
            n: Number of paths to sample
            min_distance: Minimum edit distance ratio between paths
            temperature: Temperature for sampling
            
        Returns:
            List of diverse paths
        """
        diverse_paths = []
        attempts = 0
        max_attempts = n * 20  # Avoid infinite loop
        
        while len(diverse_paths) < n and attempts < max_attempts:
            attempts += 1
            
            # Sample a path
            path = self.sample_by_value(temperature=temperature)
            
            # Check if it's different enough from existing paths
            is_diverse = True
            for existing_path in diverse_paths:
                distance = self._path_distance(path, existing_path)
                if distance < min_distance:
                    is_diverse = False
                    break
            
            if is_diverse:
                diverse_paths.append(path)
        
        return diverse_paths
    
    def sample_multiple(self, n: int = 10, 
                       strategy: str = "value",
                       temperature: float = 1.0) -> List[SampledPath]:
        """
        Sample multiple paths using specified strategy.
        
        Args:
            n: Number of paths to sample
            strategy: "value", "visits", or "mixed"
            temperature: Temperature for value-based sampling
            
        Returns:
            List of sampled paths
        """
        paths = []
        
        for _ in range(n):
            if strategy == "value":
                path = self.sample_by_value(temperature=temperature)
            elif strategy == "visits":
                path = self.sample_by_visits()
            elif strategy == "mixed":
                # Alternate between strategies
                if len(paths) % 2 == 0:
                    path = self.sample_by_value(temperature=temperature)
                else:
                    path = self.sample_by_visits()
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            paths.append(path)
        
        return paths
    
    def get_consistent_solution(self, n_samples: int = 10,
                               temperature: float = 1.0,
                               llm = None) -> Dict[str, Any]:
        """
        Sample multiple solutions and find the most consistent one.
        
        Args:
            n_samples: Number of paths to sample
            temperature: Temperature for sampling
            llm: LLM adapter for comparing solutions (optional)
            
        Returns:
            Dictionary with consistent solution and statistics
        """
        # Sample multiple paths
        paths = self.sample_multiple(n_samples, strategy="value", 
                                    temperature=temperature)
        
        # Extract final states (solutions)
        solutions = [p.final_state for p in paths]
        
        if llm:
            # Use LLM to cluster similar solutions
            clusters = self._cluster_solutions_with_llm(solutions, llm)
        else:
            # Simple exact match clustering
            clusters = self._cluster_solutions_simple(solutions)
        
        # Find largest cluster (most consistent)
        best_cluster = max(clusters, key=lambda c: c['count'])
        
        # Get best solution from cluster
        best_path_idx = solutions.index(best_cluster['solutions'][0])
        best_path = paths[best_path_idx]
        
        return {
            'solution': best_cluster['representative'],
            'confidence': best_cluster['count'] / n_samples,
            'support': best_cluster['count'],
            'total_samples': n_samples,
            'path': best_path,
            'clusters': clusters
        }
    
    def _get_all_paths(self, node: MCTSNode, 
                      current_path: Optional[List] = None) -> List[SampledPath]:
        """Recursively get all paths to leaves."""
        if current_path is None:
            current_path = []
        
        current_path = current_path + [node]
        
        if not node.children:
            # Leaf node - create path
            nodes = current_path
            actions = [n.action_taken for n in nodes[1:]]
            states = [n.state for n in nodes]
            total_value = sum(n.value for n in nodes)
            total_visits = sum(n.visits for n in nodes)
            
            return [SampledPath(
                nodes=nodes,
                actions=actions,
                states=states,
                total_value=total_value,
                total_visits=total_visits
            )]
        
        # Recursive case
        all_paths = []
        for child in node.children:
            child_paths = self._get_all_paths(child, current_path)
            all_paths.extend(child_paths)
        
        return all_paths
    
    def _path_distance(self, path1: SampledPath, path2: SampledPath) -> float:
        """
        Calculate distance between two paths (0 = identical, 1 = completely different).
        
        Uses action sequence similarity.
        """
        actions1 = [str(a) for a in path1.actions]
        actions2 = [str(a) for a in path2.actions]
        
        # Levenshtein distance normalized by max length
        distance = self._levenshtein(actions1, actions2)
        max_len = max(len(actions1), len(actions2))
        
        return distance / max_len if max_len > 0 else 0.0
    
    def _levenshtein(self, s1: List, s2: List) -> int:
        """Calculate Levenshtein distance between two sequences."""
        if len(s1) < len(s2):
            return self._levenshtein(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                # j+1 instead of j since previous_row and current_row are one character longer
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _cluster_solutions_simple(self, solutions: List[str]) -> List[Dict]:
        """Simple clustering based on exact match."""
        counter = Counter(solutions)
        clusters = []
        
        for solution, count in counter.most_common():
            clusters.append({
                'representative': solution,
                'solutions': [solution] * count,
                'count': count
            })
        
        return clusters
    
    def _cluster_solutions_with_llm(self, solutions: List[str], llm) -> List[Dict]:
        """Use LLM to cluster semantically similar solutions."""
        clusters = []
        clustered = set()
        
        for i, sol1 in enumerate(solutions):
            if i in clustered:
                continue
            
            cluster = {
                'representative': sol1,
                'solutions': [sol1],
                'count': 1
            }
            clustered.add(i)
            
            # Find similar solutions
            for j, sol2 in enumerate(solutions[i+1:], start=i+1):
                if j in clustered:
                    continue
                
                # Ask LLM if solutions are equivalent
                prompt = f"""
                Are these two solutions essentially the same?
                
                Solution 1: {sol1[:500]}
                
                Solution 2: {sol2[:500]}
                
                Answer YES or NO:
                """
                
                response = llm.generate(prompt, max_tokens=10)
                
                if "YES" in response.upper():
                    cluster['solutions'].append(sol2)
                    cluster['count'] += 1
                    clustered.add(j)
            
            clusters.append(cluster)
        
        return sorted(clusters, key=lambda c: c['count'], reverse=True)


# ========== Extend MCTS with Sampling Methods ==========

class SamplingMCTS(MCTS):
    """MCTS extended with sampling capabilities."""
    
    def sample(self, n: int = 1, temperature: float = 1.0, 
              strategy: str = "value") -> List[SampledPath]:
        """
        Sample paths from the tree.
        
        Args:
            n: Number of paths to sample
            temperature: Temperature for value-based sampling
            strategy: "value", "visits", or "diverse"
            
        Returns:
            List of sampled paths (or single path if n=1)
        """
        sampler = MCTSSampler(self)
        
        if strategy == "diverse":
            paths = sampler.sample_diverse(n, temperature=temperature)
        else:
            paths = sampler.sample_multiple(n, strategy=strategy, 
                                           temperature=temperature)
        
        return paths[0] if n == 1 and paths else paths
    
    def get_top_k(self, k: int = 5, criterion: str = "value") -> List[SampledPath]:
        """Get top-K paths."""
        sampler = MCTSSampler(self)
        return sampler.sample_top_k(k, criterion)
    
    def check_consistency(self, n_samples: int = 10, 
                         temperature: float = 1.0) -> Dict[str, Any]:
        """Check solution consistency across samples."""
        sampler = MCTSSampler(self)
        return sampler.get_consistent_solution(
            n_samples, temperature, self.llm
        )
    
    @property
    def all_solutions(self) -> List[str]:
        """Get all unique solutions (final states) in the tree."""
        sampler = MCTSSampler(self)
        all_paths = sampler._get_all_paths(self.root)
        return list(set(p.final_state for p in all_paths))