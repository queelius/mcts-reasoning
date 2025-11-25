"""
Reasoning-specific MCTS with compositional actions and all features integrated.
"""

from typing import List, Dict, Any, Optional, Tuple
import random
import logging

logger = logging.getLogger(__name__)

from .core import MCTS, MCTSNode
from .sampling import MCTSSampler, SampledPath
from .compositional import smart_termination
from .compositional.actions import CompositionalAction, ActionSelector
from .context_manager import ContextManager, ContextConfig, configure_context_for_model
from .solution_detection import (
    SolutionDetector,
    SolutionFinalizer,
    is_finalized_solution,
    should_attempt_finalization
)
from .learning import PathLearner
from .meta_reasoning import MetaReasoner, MetaReasoningActionSelector
from .reflection import ReflectionCritic


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
        self.custom_actions = None  # Custom simple actions (when compositional is disabled)
        self.context_manager = None  # Will be initialized with context config
        self.solution_detector = None  # Will be initialized when enabled
        self.solution_finalizer = None  # Will be initialized when enabled
        self.auto_finalize_solutions = True  # Automatically finalize detected solutions
        self.path_learner = None  # Will be initialized when learning enabled
        self.auto_learn = False  # Automatically learn from search results
        self.meta_reasoner = None  # Will be initialized when meta-reasoning enabled
        self.meta_bias_strength = 3.0  # How strongly to bias toward meta-reasoner suggestions
        self.reflection_critic = None  # Will be initialized when reflection enabled
        self.reflection_threshold = 0.6  # Quality threshold for automatic refinement

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
            # Check for pending RAG store from with_rag_store called before this
            rag_store = getattr(self, '_pending_rag_store', None)
            self.action_selector = ActionSelector(
                exploration_constant=exp_const,
                rag_store=rag_store
            )
        return self

    def with_question(self, question: str) -> 'ReasoningMCTS':
        """Set the original question being solved."""
        self.original_question = question
        self._metadata['question'] = question
        return self

    def with_actions(self, actions: List[str]) -> 'ReasoningMCTS':
        """
        Set custom simple actions (used when compositional actions are disabled).

        Args:
            actions: List of action strings

        Returns:
            self (for chaining)
        """
        self.custom_actions = actions
        return self

    def with_terminal_detection(self, use_llm: bool = True) -> 'ReasoningMCTS':
        """Configure terminal state detection."""
        self.terminal_check_with_llm = use_llm
        return self

    def with_context_config(self, config: Optional[ContextConfig] = None,
                           auto_configure: bool = True) -> 'ReasoningMCTS':
        """
        Configure automatic context management and summarization.

        Args:
            config: Context configuration (defaults based on model if not provided)
            auto_configure: If True and config not provided, auto-detect model limits

        Returns:
            self (for chaining)
        """
        if config:
            self.context_manager = ContextManager(config)
        elif auto_configure and self.llm:
            # Try to auto-detect model from LLM provider
            model_name = getattr(self.llm, 'model', 'unknown')
            config = configure_context_for_model(model_name)
            self.context_manager = ContextManager(config)
        else:
            # Use default config
            self.context_manager = ContextManager()

        # Load tokenizer if LLM is available
        if self.llm:
            model_name = getattr(self.llm, 'model', 'unknown')
            self.context_manager.load_tokenizer(model_name)

        return self

    def with_solution_detection(self, enabled: bool = True,
                                threshold: float = 0.7,
                                auto_finalize: bool = True) -> 'ReasoningMCTS':
        """
        Configure automatic solution detection and finalization.

        When enabled, uses LLM-as-a-judge to detect complete solutions,
        then creates polished final answers and marks nodes as terminal.

        Args:
            enabled: Whether to enable solution detection
            threshold: Confidence threshold for considering something a solution (0-1)
            auto_finalize: Automatically finalize detected solutions

        Returns:
            self (for chaining)
        """
        if enabled and self.llm:
            self.solution_detector = SolutionDetector(self.llm, threshold=threshold)
            self.solution_finalizer = SolutionFinalizer(self.llm)
            self.auto_finalize_solutions = auto_finalize
        else:
            self.solution_detector = None
            self.solution_finalizer = None

        return self

    def with_rag_store(self, rag_store: Any) -> 'ReasoningMCTS':
        """
        Set RAG store for compositional guidance.

        Args:
            rag_store: CompositionalRAGStore for guiding action selection

        Returns:
            self (for chaining)
        """
        if self.action_selector:
            self.action_selector.rag_store = rag_store
        else:
            # Will be set when action_selector is created
            self._pending_rag_store = rag_store
        return self

    def with_learning(self, enabled: bool = True, auto_learn: bool = True) -> 'ReasoningMCTS':
        """
        Enable learning from successful paths.

        When enabled, automatically updates RAG store with patterns from
        successful reasoning paths after each search.

        Args:
            enabled: Whether to enable learning
            auto_learn: Automatically learn at end of each search

        Returns:
            self (for chaining)
        """
        if enabled:
            # Get RAG store - check multiple sources
            rag_store = None

            # First try action selector
            if self.action_selector and hasattr(self.action_selector, 'rag_store'):
                rag_store = self.action_selector.rag_store

            # Then try pending
            if rag_store is None and hasattr(self, '_pending_rag_store'):
                rag_store = self._pending_rag_store

            # Create default if still None
            if rag_store is None:
                from .compositional.rag import CompositionalRAGStore
                rag_store = CompositionalRAGStore()
                # Set it now so action selector can use it
                if self.action_selector:
                    self.action_selector.rag_store = rag_store
                else:
                    self._pending_rag_store = rag_store

            self.path_learner = PathLearner(rag_store)
            self.auto_learn = auto_learn

            logger.info(f"Learning enabled with RAG store ({len(rag_store)} existing entries)")
        else:
            self.path_learner = None
            self.auto_learn = False

        return self

    def with_meta_reasoning(self, enabled: bool = True,
                           temperature: float = 0.2,
                           bias_strength: float = 3.0) -> 'ReasoningMCTS':
        """
        Enable meta-reasoning where LLM suggests next action.

        Instead of purely RAG-weighted or random action selection, the LLM
        analyzes the current reasoning state and suggests which cognitive
        operation would be most productive to try next.

        Args:
            enabled: Whether to enable meta-reasoning
            temperature: Temperature for meta-reasoning (lower = more focused)
            bias_strength: How much to bias toward suggested operations (multiplier)

        Returns:
            self (for chaining)
        """
        if enabled and self.llm:
            self.meta_reasoner = MetaReasoner(self.llm, temperature=temperature)
            self.meta_bias_strength = bias_strength
            logger.info(f"Meta-reasoning enabled (bias_strength={bias_strength})")
        else:
            self.meta_reasoner = None

        return self

    def with_reflection(self, enabled: bool = True,
                       temperature: float = 0.3,
                       quality_threshold: float = 0.6) -> 'ReasoningMCTS':
        """
        Enable reflection/critique for self-improvement.

        When enabled, the LLM critiques its own reasoning and suggests
        improvements, creating a feedback loop for higher quality reasoning.

        Args:
            enabled: Whether to enable reflection
            temperature: Temperature for critique (0.2-0.4 recommended)
            quality_threshold: Automatically refine if quality < threshold

        Returns:
            self (for chaining)
        """
        if enabled and self.llm:
            self.reflection_critic = ReflectionCritic(self.llm, temperature=temperature)
            self.reflection_threshold = quality_threshold
            logger.info(f"Reflection enabled (quality_threshold={quality_threshold})")
        else:
            self.reflection_critic = None

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
        # Return custom actions if specified, otherwise use default actions
        if self.custom_actions is not None:
            return self.custom_actions

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
            # Create action selector with pending RAG store if available
            rag_store = getattr(self, '_pending_rag_store', None)
            self.action_selector = ActionSelector(
                exploration_constant=self.exploration_constant,
                rag_store=rag_store
            )

        # Get previous action if available (for connection type determination)
        previous_action = None
        if hasattr(self, '_last_action'):
            previous_action = self._last_action

        # If meta-reasoning is enabled, use it to bias action selection
        if self.meta_reasoner and self.original_question:
            # Wrap action selector with meta-reasoning
            meta_selector = MetaReasoningActionSelector(
                self.action_selector,
                self.meta_reasoner,
                bias_strength=self.meta_bias_strength
            )
            return meta_selector.get_valid_actions(
                current_state=state,
                previous_action=previous_action,
                n_samples=15,
                problem=self.original_question
            )
        else:
            # Get valid actions from the action selector
            # Pass problem for RAG-guided selection if available
            return self.action_selector.get_valid_actions(
                current_state=state,
                previous_action=previous_action,
                n_samples=15,
                problem=self.original_question if self.original_question else None
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
            new_state = action.execute(
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
            new_state = state + f"\n\n{action}:\n{response}"

        # Automatic context management: check if summarization needed
        if self.context_manager and self.context_manager.should_summarize(new_state):
            new_state = self.context_manager.summarize_state(
                new_state,
                self.llm,
                self.original_question
            )

        # Automatic solution detection and finalization
        if (self.auto_finalize_solutions and
            self.solution_detector and
            self.solution_finalizer and
            not is_finalized_solution(new_state)):

            # Check if this state contains a satisfactory solution
            if should_attempt_finalization(new_state, self.solution_detector, self.original_question):
                # Create polished final answer
                new_state = self.solution_finalizer.finalize_solution(
                    new_state,
                    self.original_question,
                    style="clear"
                )
                # Node will be marked terminal in _is_terminal_state()

        # Automatic reflection and refinement
        if (self.reflection_critic and
            self.original_question and
            not is_finalized_solution(new_state)):

            # Critique the new state
            critique = self.reflection_critic.critique(new_state, self.original_question)

            # If quality is below threshold and refinement needed, refine it
            if critique.quality_score < self.reflection_threshold and critique.needs_refinement:
                logger.debug(f"Reflecting: quality={critique.quality_score:.2f} < {self.reflection_threshold}, refining...")
                new_state = self.reflection_critic.refine(
                    new_state, critique, self.original_question
                )

        return new_state
    
    def _is_terminal_state(self, state: str) -> bool:
        """Check if reasoning has reached a terminal state."""
        # First check if this is a finalized solution (always terminal)
        if is_finalized_solution(state):
            return True

        # Otherwise use smart_termination from compositional module
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

    # ========== Search Override for Learning ==========

    def search(self, initial_state: str, simulations: int = 100) -> 'ReasoningMCTS':
        """
        Run MCTS search with automatic learning.

        Args:
            initial_state: Starting state string
            simulations: Number of simulations to run

        Returns:
            self (for chaining)
        """
        # Call parent search
        super().search(initial_state, simulations)

        # Automatic learning from successful paths
        if self.auto_learn and self.path_learner and self.original_question:
            logger.info("Auto-learning from search results...")
            self.path_learner.learn_from_tree(
                self,
                self.original_question,
                min_value=0.5,
                learn_from_solutions=True
            )

            stats = self.path_learner.get_stats()
            logger.info(f"Learning complete: {stats['learning_count']} patterns learned")

        return self

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