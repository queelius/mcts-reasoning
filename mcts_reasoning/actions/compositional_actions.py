"""
Compositional Action System for Tree-of-Thought MCTS

Based on the compositional prompting framework, but using MCTS and LLM-as-judge
instead of RL/Q-learning for policy optimization.

Action Components:
- Cognitive Operation (ω): What mental operation to perform
- Focus Aspect (φ): What aspect to focus on  
- Reasoning Style (σ): How to approach the reasoning
- Connection Type (κ): How to connect to previous reasoning
- Output Format (τ): How to structure the output
"""

from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import itertools
import random
import numpy as np

from .base_action import BaseAction, ActionResult
from ..llm.ollama_client import LLMManager


class CognitiveOperation(Enum):
    """Cognitive operations that can be performed (ω)"""
    DECOMPOSE = "decompose"          # Break into parts
    ANALYZE = "analyze"              # Examine in detail
    SYNTHESIZE = "synthesize"        # Combine elements
    VERIFY = "verify"                # Check correctness
    ABSTRACT = "abstract"            # Extract general principles
    CONCRETIZE = "concretize"        # Make specific/concrete
    COMPARE = "compare"              # Find similarities/differences
    EVALUATE = "evaluate"            # Assess quality/value
    GENERATE = "generate"            # Create new content
    REFINE = "refine"               # Improve existing content


class FocusAspect(Enum):
    """What aspect to focus on (φ)"""
    STRUCTURE = "structure"          # Overall organization
    DETAILS = "details"              # Specific elements
    ASSUMPTIONS = "assumptions"      # Underlying assumptions
    CONSTRAINTS = "constraints"      # Limitations and requirements
    GOAL = "goal"                   # End objective
    PROGRESS = "progress"            # Current state of solution
    ERRORS = "errors"               # Mistakes or issues
    ALTERNATIVES = "alternatives"    # Other possibilities


class ReasoningStyle(Enum):
    """How to approach the reasoning (σ)"""
    SYSTEMATIC = "systematic"        # Step-by-step, methodical
    INTUITIVE = "intuitive"         # Pattern recognition, insight
    FORMAL = "formal"               # Mathematical/logical rigor
    EXPLORATORY = "exploratory"     # Open-ended investigation
    CRITICAL = "critical"           # Questioning and skeptical
    CREATIVE = "creative"           # Novel and imaginative


class ConnectionType(Enum):
    """How to connect to previous reasoning (κ)"""
    CONTINUE = "continue"           # Build on previous
    CONTRAST = "contrast"           # Show differences
    ELABORATE = "elaborate"         # Add more detail
    SUMMARIZE = "summarize"         # Condense previous
    PIVOT = "pivot"                # Change direction
    VERIFY = "verify"              # Check previous work
    CONCLUDE = "conclude"          # Draw final conclusion
    QUESTION = "question"          # Raise new questions


class OutputFormat(Enum):
    """How to structure the output (τ)"""
    LIST = "list"                  # Bullet points
    STEPS = "steps"                # Numbered sequence
    COMPARISON = "comparison"       # Side-by-side comparison
    EXPLANATION = "explanation"     # Narrative explanation
    SOLUTION = "solution"          # Direct answer
    CODE = "code"                  # Programming code
    MATHEMATICAL = "mathematical"   # Mathematical notation
    FREEFORM = "free-form"         # Unstructured


@dataclass
class ComposedAction:
    """A compositional action tuple (ω, φ, σ, κ, τ)"""
    operation: CognitiveOperation
    focus: FocusAspect
    style: ReasoningStyle
    connection: ConnectionType
    output_format: OutputFormat
    
    def to_string(self) -> str:
        """Convert to human-readable string"""
        return (f"{self.operation.value}_{self.focus.value}_"
                f"{self.style.value}_{self.connection.value}_"
                f"{self.output_format.value}")
    
    def __hash__(self):
        return hash((self.operation, self.focus, self.style, 
                    self.connection, self.output_format))


class ActionComposer:
    """Composes actions based on state and constraints"""
    
    # Define semantic compatibility rules (Bayesian network structure)
    COMPATIBILITY_RULES = {
        # Some operations work better with certain focus aspects
        CognitiveOperation.DECOMPOSE: {
            FocusAspect.STRUCTURE, FocusAspect.GOAL, FocusAspect.CONSTRAINTS
        },
        CognitiveOperation.ANALYZE: {
            FocusAspect.DETAILS, FocusAspect.ASSUMPTIONS, FocusAspect.ERRORS
        },
        CognitiveOperation.SYNTHESIZE: {
            FocusAspect.STRUCTURE, FocusAspect.PROGRESS, FocusAspect.ALTERNATIVES
        },
        CognitiveOperation.VERIFY: {
            FocusAspect.ERRORS, FocusAspect.ASSUMPTIONS, FocusAspect.CONSTRAINTS
        },
    }
    
    # Some styles work better with certain operations
    STYLE_COMPATIBILITY = {
        CognitiveOperation.DECOMPOSE: {
            ReasoningStyle.SYSTEMATIC, ReasoningStyle.FORMAL
        },
        CognitiveOperation.GENERATE: {
            ReasoningStyle.CREATIVE, ReasoningStyle.EXPLORATORY
        },
        CognitiveOperation.VERIFY: {
            ReasoningStyle.CRITICAL, ReasoningStyle.FORMAL
        },
    }
    
    @staticmethod
    def get_valid_actions(
        context: str,
        previous_action: Optional[ComposedAction] = None,
        constraints: Optional[Dict[str, Any]] = None
    ) -> List[ComposedAction]:
        """
        Generate valid compositional actions based on context and constraints.
        
        This implements the Bayesian network structure from the paper,
        where certain combinations are more likely/valid than others.
        """
        valid_actions = []
        
        # Generate all possible combinations
        for op in CognitiveOperation:
            # Get compatible focus aspects
            compatible_focuses = ActionComposer.COMPATIBILITY_RULES.get(
                op, set(FocusAspect)
            )
            
            # Get compatible styles
            compatible_styles = ActionComposer.STYLE_COMPATIBILITY.get(
                op, set(ReasoningStyle)
            )
            
            for focus in compatible_focuses:
                for style in compatible_styles:
                    # Connection type depends on previous action
                    if previous_action is None:
                        valid_connections = [ConnectionType.CONTINUE]
                    else:
                        valid_connections = ActionComposer._get_valid_connections(
                            op, previous_action
                        )
                    
                    for connection in valid_connections:
                        # Output format depends on operation and focus
                        valid_formats = ActionComposer._get_valid_formats(op, focus)
                        
                        for fmt in valid_formats:
                            action = ComposedAction(
                                operation=op,
                                focus=focus,
                                style=style,
                                connection=connection,
                                output_format=fmt
                            )
                            valid_actions.append(action)
        
        # Apply any additional constraints
        if constraints:
            valid_actions = ActionComposer._apply_constraints(
                valid_actions, constraints
            )
        
        return valid_actions
    
    @staticmethod
    def _get_valid_connections(
        current_op: CognitiveOperation,
        previous_action: ComposedAction
    ) -> List[ConnectionType]:
        """Determine valid connection types based on previous action"""
        # If previous was decompose, we can elaborate or continue
        if previous_action.operation == CognitiveOperation.DECOMPOSE:
            return [ConnectionType.ELABORATE, ConnectionType.CONTINUE]
        # If previous was analyze, we can verify or pivot
        elif previous_action.operation == CognitiveOperation.ANALYZE:
            return [ConnectionType.VERIFY, ConnectionType.PIVOT, ConnectionType.CONTINUE]
        # If we're synthesizing, we might be concluding
        elif current_op == CognitiveOperation.SYNTHESIZE:
            return [ConnectionType.SUMMARIZE, ConnectionType.CONCLUDE]
        else:
            return list(ConnectionType)
    
    @staticmethod
    def _get_valid_formats(
        op: CognitiveOperation,
        focus: FocusAspect
    ) -> List[OutputFormat]:
        """Determine valid output formats based on operation and focus"""
        if op == CognitiveOperation.DECOMPOSE:
            return [OutputFormat.LIST, OutputFormat.STEPS]
        elif op == CognitiveOperation.COMPARE:
            return [OutputFormat.COMPARISON, OutputFormat.LIST]
        elif focus == FocusAspect.DETAILS:
            return [OutputFormat.EXPLANATION, OutputFormat.STEPS]
        elif op == CognitiveOperation.GENERATE and focus == FocusAspect.ALTERNATIVES:
            return [OutputFormat.LIST, OutputFormat.COMPARISON]
        else:
            return list(OutputFormat)
    
    @staticmethod
    def _apply_constraints(
        actions: List[ComposedAction],
        constraints: Dict[str, Any]
    ) -> List[ComposedAction]:
        """Apply additional constraints to filter actions"""
        filtered = actions
        
        # Example: If we want only systematic reasoning
        if constraints.get("require_systematic"):
            filtered = [a for a in filtered if a.style == ReasoningStyle.SYSTEMATIC]
        
        # Example: If we need a specific output format
        if "output_format" in constraints:
            required_format = constraints["output_format"]
            filtered = [a for a in filtered if a.output_format == required_format]
        
        return filtered


class CompositionalPromptBuilder:
    """Builds actual prompts from compositional actions"""
    
    @staticmethod
    def build_prompt(
        action: ComposedAction,
        context: str,
        original_question: str,
        previous_response: Optional[str] = None
    ) -> str:
        """
        Construct the actual prompt from a compositional action.
        
        This implements the g(a, c) function from the paper.
        """
        parts = []
        
        # 1. Connection phrase (if there's previous context)
        if previous_response:
            connection_phrase = CompositionalPromptBuilder._get_connection_phrase(
                action.connection, previous_response
            )
            parts.append(connection_phrase)
        
        # 2. Main directive based on operation, focus, and style
        directive = CompositionalPromptBuilder._get_main_directive(
            action.operation, action.focus, action.style,
            context, original_question
        )
        parts.append(directive)
        
        # 3. Output format specification
        format_spec = CompositionalPromptBuilder._get_format_specification(
            action.output_format
        )
        parts.append(format_spec)
        
        return "\n\n".join(parts)
    
    @staticmethod
    def _get_connection_phrase(
        connection: ConnectionType,
        previous: str
    ) -> str:
        """Generate connection phrase based on connection type"""
        phrases = {
            ConnectionType.CONTINUE: f"Building on the previous analysis:\n{previous[:200]}...\n\nLet's continue by",
            ConnectionType.CONTRAST: f"In contrast to what we just established:\n{previous[:200]}...\n\nLet's consider",
            ConnectionType.ELABORATE: f"To elaborate further on:\n{previous[:200]}...\n\nWe need to",
            ConnectionType.SUMMARIZE: f"Given everything so far:\n{previous[:200]}...\n\nLet me summarize by",
            ConnectionType.PIVOT: f"While we've explored:\n{previous[:200]}...\n\nLet's pivot to",
            ConnectionType.VERIFY: f"To verify our reasoning:\n{previous[:200]}...\n\nLet's check by",
            ConnectionType.CONCLUDE: f"Based on all our analysis:\n{previous[:200]}...\n\nWe can conclude by",
            ConnectionType.QUESTION: f"This raises questions about:\n{previous[:200]}...\n\nLet's explore",
        }
        return phrases.get(connection, "Continuing from the previous step,")
    
    @staticmethod
    def _get_main_directive(
        operation: CognitiveOperation,
        focus: FocusAspect,
        style: ReasoningStyle,
        context: str,
        question: str
    ) -> str:
        """Generate the main directive combining operation, focus, and style"""
        
        # Start with the operation
        op_phrases = {
            CognitiveOperation.DECOMPOSE: "breaking down",
            CognitiveOperation.ANALYZE: "analyzing",
            CognitiveOperation.SYNTHESIZE: "synthesizing",
            CognitiveOperation.VERIFY: "verifying",
            CognitiveOperation.ABSTRACT: "abstracting",
            CognitiveOperation.CONCRETIZE: "making concrete",
            CognitiveOperation.COMPARE: "comparing",
            CognitiveOperation.EVALUATE: "evaluating",
            CognitiveOperation.GENERATE: "generating",
            CognitiveOperation.REFINE: "refining",
        }
        
        # Add focus
        focus_phrases = {
            FocusAspect.STRUCTURE: "the overall structure",
            FocusAspect.DETAILS: "the specific details",
            FocusAspect.ASSUMPTIONS: "the underlying assumptions",
            FocusAspect.CONSTRAINTS: "the constraints and limitations",
            FocusAspect.GOAL: "the end goal",
            FocusAspect.PROGRESS: "our current progress",
            FocusAspect.ERRORS: "potential errors or issues",
            FocusAspect.ALTERNATIVES: "alternative approaches",
        }
        
        # Add style modifier
        style_modifiers = {
            ReasoningStyle.SYSTEMATIC: "systematically and methodically",
            ReasoningStyle.INTUITIVE: "using intuition and pattern recognition",
            ReasoningStyle.FORMAL: "with formal logical rigor",
            ReasoningStyle.EXPLORATORY: "through open exploration",
            ReasoningStyle.CRITICAL: "with critical analysis",
            ReasoningStyle.CREATIVE: "with creative thinking",
        }
        
        directive = f"""
Please help by {op_phrases[operation]} {focus_phrases[focus]} {style_modifiers[style]}.

Context: {context}

Original Question: {question}

Focus specifically on {focus_phrases[focus]} while {op_phrases[operation]} the problem.
"""
        return directive
    
    @staticmethod
    def _get_format_specification(output_format: OutputFormat) -> str:
        """Generate output format specification"""
        format_specs = {
            OutputFormat.LIST: "Please provide your response as a clear bullet-point list.",
            OutputFormat.STEPS: "Please provide your response as numbered steps.",
            OutputFormat.COMPARISON: "Please provide a side-by-side comparison.",
            OutputFormat.EXPLANATION: "Please provide a detailed narrative explanation.",
            OutputFormat.SOLUTION: "Please provide a direct, concise solution.",
            OutputFormat.CODE: "Please provide your response with code examples.",
            OutputFormat.MATHEMATICAL: "Please use mathematical notation where appropriate.",
            OutputFormat.FREEFORM: "Feel free to structure your response as you see fit.",
        }
        return format_specs[output_format]


class CompositionalAction(BaseAction):
    """
    A compositional action that combines multiple linguistic components.
    
    This replaces the simple atomic actions with rich compositional ones.
    """
    
    def __init__(self, composed_action: ComposedAction):
        self.composed_action = composed_action
        super().__init__(
            action_type=composed_action.to_string(),
            description=f"Compositional action: {composed_action.operation.value}"
        )
    
    def execute(
        self,
        context: str,
        original_question: str,
        llm_manager: LLMManager,
        previous_response: Optional[str] = None,
        **kwargs
    ) -> ActionResult:
        """Execute the compositional action"""
        
        # Build the prompt from the compositional action
        prompt = CompositionalPromptBuilder.build_prompt(
            self.composed_action,
            context,
            original_question,
            previous_response
        )
        
        # Execute with LLM
        response = llm_manager.reason(prompt, temperature=0.7)
        
        # Build new context based on operation type
        new_context = self._build_new_context(
            context, response.text, self.composed_action.operation
        )
        
        return ActionResult(
            response=response.text,
            new_context=new_context,
            metadata={
                "action_components": {
                    "operation": self.composed_action.operation.value,
                    "focus": self.composed_action.focus.value,
                    "style": self.composed_action.style.value,
                    "connection": self.composed_action.connection.value,
                    "format": self.composed_action.output_format.value,
                }
            }
        )
    
    def _build_new_context(
        self,
        old_context: str,
        response: str,
        operation: CognitiveOperation
    ) -> str:
        """
        Build new context based on operation type.
        
        This implements the context management function ψ from the paper.
        """
        if operation == CognitiveOperation.DECOMPOSE:
            # Replace with decomposition
            return f"Problem decomposition:\n{response}"
        elif operation == CognitiveOperation.SYNTHESIZE:
            # Compress previous reasoning
            return f"Synthesis of reasoning:\n{response}"
        elif operation == CognitiveOperation.REFINE:
            # Replace with refined version
            return f"Refined approach:\n{response}"
        elif operation in [CognitiveOperation.ANALYZE, CognitiveOperation.EVALUATE]:
            # Append analysis
            return f"{old_context}\n\nAnalysis:\n{response}"
        else:
            # Default: append
            return f"{old_context}\n\n{response}"


class CompositionalActionSelector:
    """
    Selects compositional actions using MCTS-style exploration.
    
    This replaces the Q-learning policy with MCTS exploration.
    """
    
    def __init__(self, exploration_constant: float = 1.414):
        self.exploration_constant = exploration_constant
        self.action_visit_counts: Dict[ComposedAction, int] = {}
        self.action_values: Dict[ComposedAction, float] = {}
    
    def select_action(
        self,
        valid_actions: List[ComposedAction],
        parent_visit_count: int,
        temperature: float = 1.0
    ) -> ComposedAction:
        """
        Select action using UCB1 formula for MCTS.
        
        Instead of Q-values, we use empirical action values from tree exploration.
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
                    np.log(parent_visit_count) / self.action_visit_counts[action]
                )
                ucb = exploitation + exploration * temperature
            else:
                ucb = float('inf')  # Unexplored action
            ucb_scores.append(ucb)
        
        # Select action with highest UCB score
        best_idx = np.argmax(ucb_scores)
        return valid_actions[best_idx]
    
    def update_action_value(
        self,
        action: ComposedAction,
        value: float
    ):
        """Update action value based on tree exploration results"""
        if action not in self.action_visit_counts:
            self.action_visit_counts[action] = 0
            self.action_values[action] = 0
        
        # Incremental update
        n = self.action_visit_counts[action]
        self.action_values[action] = (
            (self.action_values[action] * n + value) / (n + 1)
        )
        self.action_visit_counts[action] += 1