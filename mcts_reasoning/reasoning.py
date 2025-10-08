"""
Reasoning-specific MCTS with compositional actions and all features integrated.
"""

from typing import List, Dict, Any, Optional, Tuple
import random

from .core import MCTS, MCTSNode
from .sampling import MCTSSampler, SampledPath
from .compositional import smart_termination
from .compositional.actions import CompositionalAction, ActionSelector


class ReasoningMCTS(MCTS):
    """
    MCTS for LLM-based reasoning with all features integrated.

    Features:
    - Fluent API
    - JSON serialization
    - Multiple sampling strategies
    - Compositional actions with advanced prompt engineering
    - Consistency checking
    - Smart termination detection

    Example:
        from mcts_reasoning import ReasoningMCTS, get_llm

        llm = get_llm("openai", model="gpt-4")

        mcts = (
            ReasoningMCTS()
            .with_llm(llm)
            .with_exploration(1.414)
            .with_compositional_actions()
            .with_question("What is the sum of all prime numbers less than 20?")
            .search("Let's solve this problem:", simulations=100)
        )

        # Get best solution
        print(f"Solution: {mcts.solution}")

        # Sample diverse solutions
        paths = mcts.sample(n=5, strategy="diverse")

        # Check consistency
        result = mcts.check_consistency(n_samples=10)
        print(f"Most consistent solution (confidence={result['confidence']}):")
        print(result['solution'])
    """

    def __init__(self):
        super().__init__()
        self.use_compositional = False
        self.original_question = ""
        self.terminal_check_with_llm = True
        self.action_selector = None  # Will be initialized when compositional is enabled

    def with_compositional_actions(self, enabled: bool = True,
                                   exploration_constant: Optional[float] = None) -> 'ReasoningMCTS':
        """
        Enable/disable compositional action space.

        Args:
            enabled: Whether to use compositional actions
            exploration_constant: UCB1 constant for action selection (defaults to tree exploration constant)

        Returns:
            self (for chaining)
        """
        self.use_compositional = enabled
        if enabled:
            exp_const = exploration_constant if exploration_constant is not None else self.exploration_constant
            self.action_selector = ActionSelector(exploration_constant=exp_const)
        return self

    def with_question(self, question: str) -> 'ReasoningMCTS':
        """Set the original question being solved."""
        self.original_question = question
        self._metadata['question'] = question
        return self

    def with_terminal_detection(self, use_llm: bool = True) -> 'ReasoningMCTS':
        """Configure terminal state detection."""
        self.terminal_check_with_llm = use_llm
        return self

    # ========== Override MCTS Methods ==========

    def _get_actions(self, state: str) -> List[Any]:
        """Get possible reasoning actions."""
        if self.use_compositional:
            return self._get_compositional_actions(state)
        else:
            return self._get_simple_actions(state)

    def _get_simple_actions(self, state: str) -> List[str]:
        """Simple action space."""
        return [
            "Analyze the problem",
            "Break down into subproblems",
            "Apply a specific technique",
            "Verify the approach",
            "Synthesize the solution",
            "Consider alternative methods",
            "Check assumptions",
            "Refine the solution"
        ]

    def _get_compositional_actions(self, state: str) -> List[CompositionalAction]:
        """Get compositional actions using the new system."""
        if not self.action_selector:
            self.action_selector = ActionSelector(exploration_constant=self.exploration_constant)

        # Get previous action if available (for connection type determination)
        previous_action = None
        if hasattr(self, '_last_action'):
            previous_action = self._last_action

        # Get valid actions from the action selector
        return self.action_selector.get_valid_actions(
            current_state=state,
            previous_action=previous_action,
            n_samples=15
        )
    
    def _take_action(self, state: str, action: Any) -> str:
        """Apply a reasoning action using LLM."""
        if not self.llm:
            return state + f"\n[{action}]"

        # Check if it's a CompositionalAction
        if isinstance(action, CompositionalAction):
            # Store for next action's context
            self._last_action = action
            # Execute compositional action
            return action.execute(
                llm=self.llm,
                current_state=state,
                original_question=self.original_question
            )
        else:
            # Simple string action
            prompt = f"""
Current reasoning:
{state}

Task: {action}

Continue the reasoning:
"""
            response = self.llm.generate(prompt, max_tokens=200)
            return state + f"\n\n{action}:\n{response}"
    
    def _is_terminal_state(self, state: str) -> bool:
        """Check if reasoning has reached a terminal state."""
        # Use smart_termination from compositional module
        llm_provider = self.llm if self.terminal_check_with_llm else None
        return smart_termination(
            state=state,
            llm_provider=llm_provider,
            pattern_only=not self.terminal_check_with_llm
        )
    
    def _evaluate_state(self, state: str) -> float:
        """Evaluate quality of reasoning state."""
        if not self.llm:
            return random.random()
        
        prompt = f"""
Evaluate the quality of this reasoning on a scale of 0 to 1.

Original question: {self.original_question}

Reasoning:
{state[-1500:]}  # Last 1500 chars

Consider:
- Correctness of approach
- Logical consistency
- Progress toward solution
- Clarity of reasoning

Quality score (0-1):
"""
        
        try:
            response = self.llm.generate(prompt, max_tokens=10)
            # Extract number from response
            import re
            numbers = re.findall(r'0?\.\d+|1\.0|0|1', response)
            if numbers:
                return float(numbers[0])
        except:
            pass
        
        return 0.5  # Default
    
    # ========== Sampling Methods ==========
    
    def sample(self, n: int = 1, temperature: float = 1.0, 
              strategy: str = "value") -> List[SampledPath]:
        """
        Sample reasoning paths from the tree.
        
        Args:
            n: Number of paths to sample
            temperature: Temperature for value-based sampling (0=greedy, higher=more random)
            strategy: "value", "visits", or "diverse"
            
        Returns:
            List of sampled paths (or single path if n=1)
        """
        if not self.root:
            raise ValueError("No search tree. Run .search() first.")
        
        sampler = MCTSSampler(self)
        
        if strategy == "diverse":
            paths = sampler.sample_diverse(n, temperature=temperature)
        else:
            paths = sampler.sample_multiple(n, strategy=strategy, 
                                           temperature=temperature)
        
        return paths[0] if n == 1 and paths else paths
    
    def get_top_solutions(self, k: int = 5) -> List[str]:
        """Get top-K solutions by value."""
        if not self.root:
            return []
        
        sampler = MCTSSampler(self)
        paths = sampler.sample_top_k(k, criterion="value")
        return [p.final_state for p in paths]
    
    def check_consistency(self, n_samples: int = 10, 
                         temperature: float = 1.0) -> Dict[str, Any]:
        """
        Check solution consistency across multiple samples.
        
        Args:
            n_samples: Number of paths to sample for consistency check
            temperature: Temperature for sampling
            
        Returns:
            Dictionary with:
            - solution: Most consistent solution
            - confidence: Fraction of samples agreeing
            - support: Number of samples with this solution
            - clusters: All solution clusters
        """
        if not self.root:
            raise ValueError("No search tree. Run .search() first.")
        
        sampler = MCTSSampler(self)
        return sampler.get_consistent_solution(
            n_samples, temperature, self.llm
        )
    
    # ========== Extended Properties ==========
    
    @property
    def solution_with_confidence(self) -> Tuple[str, float]:
        """Get best solution with confidence score."""
        if not self.best_node:
            return "", 0.0
        
        solution = self.solution
        confidence = self.best_value
        
        return solution, confidence
    
    @property
    def reasoning_depth(self) -> int:
        """Get maximum reasoning depth reached."""
        if not self.root:
            return 0
        
        def max_depth(node, d=0):
            if not node.children:
                return d
            return max(max_depth(c, d+1) for c in node.children)
        
        return max_depth(self.root)
    
    @property
    def exploration_breadth(self) -> float:
        """Measure of how broadly the tree explored (0-1)."""
        if not self.root or not self.root.children:
            return 0.0
        
        # Average branching factor
        total_nodes = 0
        total_children = 0
        
        def count_branches(node):
            nonlocal total_nodes, total_children
            if node.children:
                total_nodes += 1
                total_children += len(node.children)
                for child in node.children:
                    count_branches(child)
        
        count_branches(self.root)
        
        if total_nodes == 0:
            return 0.0
        
        avg_branching = total_children / total_nodes
        # Normalize (assume max branching of 10)
        return min(avg_branching / 10.0, 1.0)
    
    # ========== Utility Methods ==========
    
    def explain_reasoning(self) -> str:
        """Generate explanation of the reasoning process."""
        if not self.root:
            return "No reasoning performed yet."
        
        lines = [
            f"Reasoning for: {self.original_question}",
            f"Explored {self.stats['total_nodes']} reasoning paths",
            f"Maximum depth: {self.reasoning_depth} steps",
            f"Exploration breadth: {self.exploration_breadth:.2%}",
            f"Best path value: {self.best_value:.3f}",
            "",
            "Best reasoning chain:"
        ]
        
        # Add best path
        for i, (action, state) in enumerate(self.best_path, 1):
            lines.append(f"\nStep {i}: {action}")
            # Add abbreviated state (first/last 100 chars)
            if len(state) > 200:
                abbrev = state[:100] + "\n...\n" + state[-100:]
            else:
                abbrev = state
            lines.append(abbrev)
        
        return "\n".join(lines)
    
    def to_markdown(self) -> str:
        """Export reasoning tree to Markdown format."""
        lines = [
            f"# MCTS Reasoning",
            f"",
            f"**Question**: {self.original_question}",
            f"",
            f"## Statistics",
            f"- Nodes explored: {self.stats['total_nodes']}",
            f"- Max depth: {self.reasoning_depth}",
            f"- Best value: {self.best_value:.3f}",
            f"",
            f"## Best Solution",
            f"```",
            f"{self.solution}",
            f"```",
            f"",
            f"## Reasoning Path"
        ]
        
        for i, (action, state) in enumerate(self.best_path, 1):
            lines.append(f"\n### Step {i}: {action}")
            lines.append(f"{state[-500:]}")  # Last 500 chars
        
        return "\n".join(lines)