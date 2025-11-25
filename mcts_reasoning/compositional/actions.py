"""
Compositional Actions for MCTS

Integrates the compositional prompting framework with MCTS tree search.
Provides action selection, compatibility rules, and UCB1-based exploration.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass
import random
import numpy as np

from . import (
    ComposingPrompt,
    CognitiveOperation,
    FocusAspect,
    ReasoningStyle,
    ConnectionType,
    OutputFormat
)
from .providers import LLMProvider


# ========== Compositional Action Wrapper ==========

@dataclass
class CompositionalAction:
    """
    A compositional action for MCTS.

    Wraps a ComposingPrompt and provides MCTS-specific functionality:
    - Prompt generation
    - Action execution
    - Compatibility checking
    - Hashable for dictionaries
    """

    # Core components (ω, φ, σ, κ, τ)
    operation: CognitiveOperation
    focus: FocusAspect
    style: ReasoningStyle
    connection: ConnectionType
    output_format: OutputFormat

    def to_prompt(self, current_state: str, original_question: str,
                  previous_response: Optional[str] = None) -> str:
        """
        Build prompt for this action.

        Args:
            current_state: Current reasoning state
            original_question: Original question being solved
            previous_response: Previous reasoning step (optional)

        Returns:
            Prompt string
        """
        prompt_builder = (
            ComposingPrompt()
            .cognitive_op(self.operation)
            .focus(self.focus)
            .style(self.style)
            .connect(self.connection)
            .format(self.output_format)
            .problem_context(original_question)
        )

        # Provide substantial context to LLM (configurable via MAX_CONTEXT_LENGTH)
        # Default 4000 chars should fit in most context windows while preserving reasoning history
        max_context = getattr(self, 'max_context_length', 4000)

        if previous_response:
            prompt_builder.base_prompt(f"Previous reasoning:\n{previous_response[-max_context//2:]}")

        # Include full reasoning history up to max_context chars
        # This prevents context loss as tree deepens
        prompt_builder.base_prompt(f"Current state:\n{current_state[-max_context:]}")

        return prompt_builder.build()

    def execute(self, llm: LLMProvider, current_state: str,
                original_question: str, previous_response: Optional[str] = None) -> str:
        """
        Execute this action using an LLM.

        Args:
            llm: LLM provider
            current_state: Current reasoning state
            original_question: Original question
            previous_response: Previous step (optional)

        Returns:
            New reasoning state after applying this action
        """
        prompt = self.to_prompt(current_state, original_question, previous_response)
        response = llm.generate(prompt, max_tokens=300, temperature=0.7)

        # Build new state based on operation type
        return self._build_new_state(current_state, response)

    def _build_new_state(self, old_state: str, response: str) -> str:
        """
        Build new state based on operation type.

        This implements context management (ψ function from the paper).
        """
        if self.operation == CognitiveOperation.DECOMPOSE:
            # Replace with decomposition
            return f"Problem decomposition:\n{response}"
        elif self.operation == CognitiveOperation.SYNTHESIZE:
            # Compress previous reasoning
            return f"Synthesis of reasoning:\n{response}"
        elif self.operation == CognitiveOperation.REFINE:
            # Replace with refined version
            return f"Refined approach:\n{response}"
        elif self.operation in [CognitiveOperation.ANALYZE, CognitiveOperation.EVALUATE]:
            # Append analysis
            return f"{old_state}\n\nAnalysis:\n{response}"
        else:
            # Default: append
            return f"{old_state}\n\n{self.operation.value.title()}:\n{response}"

    def to_string(self) -> str:
        """Convert to human-readable string."""
        return (f"{self.operation.value}_{self.focus.value}_"
                f"{self.style.value}_{self.connection.value}_"
                f"{self.output_format.value}")

    def to_tuple(self) -> Tuple:
        """Convert to tuple for hashing."""
        return (self.operation, self.focus, self.style,
                self.connection, self.output_format)

    def __hash__(self):
        """Make hashable for use in dictionaries."""
        return hash(self.to_tuple())

    def __eq__(self, other):
        """Equality based on components."""
        if not isinstance(other, CompositionalAction):
            return False
        return self.to_tuple() == other.to_tuple()

    def __repr__(self):
        """String representation."""
        return f"Action({self.operation.value}, {self.focus.value}, {self.style.value})"


# ========== Action Selection and Compatibility ==========

class ActionSelector:
    """
    Selects compositional actions for MCTS exploration.

    Features:
    - Compatibility rules (Bayesian network structure)
    - UCB1-based selection for exploration/exploitation
    - Weighted action sampling
    """

    # Semantic compatibility rules (which combinations make sense)
    OPERATION_FOCUS_COMPATIBILITY = {
        CognitiveOperation.DECOMPOSE: {
            FocusAspect.STRUCTURE, FocusAspect.GOAL, FocusAspect.CONSTRAINTS
        },
        CognitiveOperation.ANALYZE: {
            FocusAspect.DETAILS, FocusAspect.ASSUMPTIONS, FocusAspect.ERRORS,
            FocusAspect.PATTERNS
        },
        CognitiveOperation.SYNTHESIZE: {
            FocusAspect.STRUCTURE, FocusAspect.PROGRESS, FocusAspect.ALTERNATIVES,
            FocusAspect.SOLUTION
        },
        CognitiveOperation.VERIFY: {
            FocusAspect.ERRORS, FocusAspect.ASSUMPTIONS, FocusAspect.CONSTRAINTS,
            FocusAspect.CORRECTNESS
        },
        CognitiveOperation.GENERATE: {
            FocusAspect.ALTERNATIVES, FocusAspect.SOLUTION, FocusAspect.EXAMPLES
        },
        CognitiveOperation.EVALUATE: {
            FocusAspect.CORRECTNESS, FocusAspect.EFFICIENCY, FocusAspect.SOLUTION
        },
    }

    OPERATION_STYLE_COMPATIBILITY = {
        CognitiveOperation.DECOMPOSE: {
            ReasoningStyle.SYSTEMATIC, ReasoningStyle.FORMAL
        },
        CognitiveOperation.GENERATE: {
            ReasoningStyle.CREATIVE, ReasoningStyle.EXPLORATORY
        },
        CognitiveOperation.VERIFY: {
            ReasoningStyle.CRITICAL, ReasoningStyle.FORMAL
        },
        CognitiveOperation.ANALYZE: {
            ReasoningStyle.SYSTEMATIC, ReasoningStyle.CRITICAL
        },
    }

    def __init__(self, exploration_constant: float = 1.414,
                 use_compatibility_rules: bool = True,
                 rag_store: Optional[Any] = None):
        """
        Initialize action selector.

        Args:
            exploration_constant: UCB1 exploration parameter
            use_compatibility_rules: Whether to enforce compatibility rules
            rag_store: Optional CompositionalRAGStore for RAG-guided action selection
        """
        self.exploration_constant = exploration_constant
        self.use_compatibility_rules = use_compatibility_rules
        self.rag_store = rag_store

        # Track action statistics for UCB1
        self.action_visit_counts: Dict[CompositionalAction, int] = {}
        self.action_values: Dict[CompositionalAction, float] = {}

    def get_rag_weights(self, problem: str) -> Optional[Dict[str, Dict[Any, float]]]:
        """
        Get RAG-guided weights for a problem.

        Args:
            problem: Problem statement

        Returns:
            Weight dictionary or None if no RAG store
        """
        if self.rag_store is None:
            return None

        try:
            from .rag import CompositionalRAGStore
            if isinstance(self.rag_store, CompositionalRAGStore):
                return self.rag_store.get_recommended_weights(problem)
        except ImportError:
            pass

        return None

    def get_valid_actions(self, current_state: str,
                         previous_action: Optional[CompositionalAction] = None,
                         n_samples: int = 15,
                         problem: Optional[str] = None) -> List[CompositionalAction]:
        """
        Get valid compositional actions for current state.

        Args:
            current_state: Current reasoning state
            previous_action: Previous action taken (for connection type)
            n_samples: Number of actions to sample (to keep action space manageable)
            problem: Optional problem statement for RAG-guided selection

        Returns:
            List of valid CompositionalAction objects
        """
        # Get RAG weights if available
        rag_weights = self.get_rag_weights(problem) if problem else None

        actions = []

        # Sample diverse operations (bias toward RAG recommendations if available)
        operations = list(CognitiveOperation)
        if rag_weights and 'cognitive_op' in rag_weights:
            # Weighted sampling of operations
            op_weights = rag_weights['cognitive_op']
            total_weight = sum(op_weights.get(op, 1.0) for op in operations)
            probs = [op_weights.get(op, 1.0) / total_weight for op in operations]
            sampled_operations = random.choices(operations, weights=probs, k=min(len(operations), 5))
        else:
            sampled_operations = random.sample(operations, min(len(operations), 5))

        for operation in sampled_operations:
            # Get compatible focuses
            if self.use_compatibility_rules and operation in self.OPERATION_FOCUS_COMPATIBILITY:
                compatible_focuses = self.OPERATION_FOCUS_COMPATIBILITY[operation]
            else:
                compatible_focuses = set(FocusAspect)

            # Get compatible styles
            if self.use_compatibility_rules and operation in self.OPERATION_STYLE_COMPATIBILITY:
                compatible_styles = self.OPERATION_STYLE_COMPATIBILITY[operation]
            else:
                compatible_styles = set(ReasoningStyle)

            # Sample from compatible sets
            focus = random.choice(list(compatible_focuses))
            style = random.choice(list(compatible_styles))

            # Determine connection type based on previous action
            connection = self._get_connection_type(operation, previous_action)

            # Determine output format based on operation
            output_format = self._get_output_format(operation, focus)

            action = CompositionalAction(
                operation=operation,
                focus=focus,
                style=style,
                connection=connection,
                output_format=output_format
            )
            actions.append(action)

        # Add some random diversity
        for _ in range(n_samples - len(actions)):
            if len(actions) >= n_samples:
                break

            action = CompositionalAction(
                operation=random.choice(list(CognitiveOperation)),
                focus=random.choice(list(FocusAspect)),
                style=random.choice(list(ReasoningStyle)),
                connection=random.choice(list(ConnectionType)),
                output_format=random.choice(list(OutputFormat))
            )
            actions.append(action)

        return actions[:n_samples]

    def _get_connection_type(self, current_op: CognitiveOperation,
                            previous_action: Optional[CompositionalAction]) -> ConnectionType:
        """Determine appropriate connection type."""
        if previous_action is None:
            return ConnectionType.CONTINUE

        # Logic based on previous operation
        prev_op = previous_action.operation

        if prev_op == CognitiveOperation.DECOMPOSE:
            return random.choice([ConnectionType.ELABORATE, ConnectionType.CONTINUE])
        elif prev_op == CognitiveOperation.ANALYZE:
            return random.choice([ConnectionType.VERIFY, ConnectionType.BUILDING_ON])
        elif current_op == CognitiveOperation.SYNTHESIZE:
            return random.choice([ConnectionType.SUMMARIZE, ConnectionType.CONCLUDE])
        else:
            return random.choice(list(ConnectionType))

    def _get_output_format(self, operation: CognitiveOperation,
                          focus: FocusAspect) -> OutputFormat:
        """Determine appropriate output format."""
        if operation == CognitiveOperation.DECOMPOSE:
            return random.choice([OutputFormat.LIST, OutputFormat.STEPS])
        elif operation == CognitiveOperation.COMPARE:
            return random.choice([OutputFormat.COMPARISON, OutputFormat.TABLE])
        elif focus == FocusAspect.DETAILS:
            return random.choice([OutputFormat.EXPLANATION, OutputFormat.STEPS])
        elif operation == CognitiveOperation.GENERATE and focus == FocusAspect.ALTERNATIVES:
            return random.choice([OutputFormat.LIST, OutputFormat.COMPARISON])
        else:
            # Default to common formats
            return random.choice([OutputFormat.EXPLANATION, OutputFormat.STEPS, OutputFormat.LIST])

    def select_action_ucb1(self, valid_actions: List[CompositionalAction],
                          parent_visit_count: int,
                          temperature: float = 1.0) -> CompositionalAction:
        """
        Select action using UCB1 formula for MCTS.

        Args:
            valid_actions: List of valid actions to choose from
            parent_visit_count: Number of times parent node has been visited
            temperature: Temperature for exploration (higher = more exploration)

        Returns:
            Selected action
        """
        if not valid_actions:
            raise ValueError("No valid actions to select from")

        # If we haven't tried all actions yet, try an unexplored one
        unexplored = [a for a in valid_actions if a not in self.action_visit_counts]
        if unexplored:
            return random.choice(unexplored)

        # Otherwise, use UCB1 to balance exploration/exploitation
        ucb_scores = []
        for action in valid_actions:
            if action in self.action_values:
                exploitation = self.action_values[action]
                exploration = self.exploration_constant * np.sqrt(
                    np.log(parent_visit_count) / max(self.action_visit_counts[action], 1)
                )
                ucb = exploitation + exploration * temperature
            else:
                ucb = float('inf')  # Unexplored action
            ucb_scores.append(ucb)

        # Select action with highest UCB score
        best_idx = np.argmax(ucb_scores)
        return valid_actions[best_idx]

    def update_action_value(self, action: CompositionalAction, value: float):
        """
        Update action value based on tree exploration results.

        Args:
            action: Action to update
            value: Observed value/reward
        """
        if action not in self.action_visit_counts:
            self.action_visit_counts[action] = 0
            self.action_values[action] = 0.0

        # Incremental update
        n = self.action_visit_counts[action]
        self.action_values[action] = (
            (self.action_values[action] * n + value) / (n + 1)
        )
        self.action_visit_counts[action] += 1

    def sample_weighted(self, weights: Optional[Dict[str, Dict[Any, float]]] = None,
                       n_samples: int = 1) -> List[CompositionalAction]:
        """
        Sample actions with optional weights.

        Args:
            weights: Weight dictionary for biased sampling
            n_samples: Number of actions to sample

        Returns:
            List of sampled actions
        """
        actions = []

        for _ in range(n_samples):
            # Use ComposingPrompt's weighted sampling
            prompt = ComposingPrompt.sample_action(weights)

            action = CompositionalAction(
                operation=prompt._cognitive_op,
                focus=prompt._focus,
                style=prompt._style,
                connection=prompt._connection,
                output_format=prompt._output_format
            )
            actions.append(action)

        return actions


# ========== Export All ==========

__all__ = [
    'CompositionalAction',
    'ActionSelector',
]
