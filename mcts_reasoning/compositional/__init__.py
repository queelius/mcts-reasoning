"""
Compositional Prompting for MCTS Reasoning

A fluid API framework for building sophisticated LLM prompts through compositional actions,
integrated with MCTS for systematic reasoning exploration.

This module provides:
- Compositional action space (ω, φ, σ, κ, τ)
- Fluid API for prompt construction
- Multi-provider LLM support
- Parallel execution capabilities
- LLM meta-reasoning (augmentation, coherence checking)
- Weighted action sampling
- Smart termination detection
"""

from typing import List, Optional, Dict, Any, Union
from enum import Enum
from dataclasses import dataclass, field
import logging
from abc import ABC, abstractmethod
import concurrent.futures
import random
import re

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ========== Core Compositional Action Components ==========

class CognitiveOperation(Enum):
    """ω: High-level reasoning operations"""
    DECOMPOSE = "decompose"          # Break into parts
    ANALYZE = "analyze"              # Examine in detail
    SYNTHESIZE = "synthesize"        # Combine elements
    VERIFY = "verify"                # Check correctness
    ABSTRACT = "abstract"            # Extract general principles
    CONCRETIZE = "concretize"        # Make specific/concrete
    COMPARE = "compare"              # Find similarities/differences
    EVALUATE = "evaluate"            # Assess quality/value
    GENERATE = "generate"            # Create new content
    REFINE = "refine"                # Improve existing content
    FINALIZE = "finalize"            # Create polished final answer (terminal)


class FocusAspect(Enum):
    """φ: What aspect to focus on"""
    STRUCTURE = "structure"          # Overall organization
    DETAILS = "details"              # Specific elements
    ASSUMPTIONS = "assumptions"      # Underlying assumptions
    CONSTRAINTS = "constraints"      # Limitations and requirements
    GOAL = "goal"                    # End objective
    PROGRESS = "progress"            # Current state of solution
    ERRORS = "errors"                # Mistakes or issues
    ALTERNATIVES = "alternatives"    # Other possibilities
    PATTERNS = "patterns"            # Recurring patterns
    SOLUTION = "solution"            # Direct solution focus
    CORRECTNESS = "correctness"      # Correctness verification
    EFFICIENCY = "efficiency"        # Efficiency considerations
    EXAMPLES = "examples"            # Example-based focus
    RELATIONSHIPS = "relationships"  # Relationships between elements


class ReasoningStyle(Enum):
    """σ: How to approach the reasoning"""
    SYSTEMATIC = "systematic"        # Step-by-step, methodical
    INTUITIVE = "intuitive"          # Pattern recognition, insight
    FORMAL = "formal"                # Mathematical/logical rigor
    EXPLORATORY = "exploratory"      # Open-ended investigation
    CRITICAL = "critical"            # Questioning and skeptical
    CREATIVE = "creative"            # Novel and imaginative


class ConnectionType(Enum):
    """κ: How to connect to previous reasoning"""
    CONTINUE = "continue"            # Build on previous
    CONTRAST = "contrast"            # Show differences
    ELABORATE = "elaborate"          # Add more detail
    SUMMARIZE = "summarize"          # Condense previous
    PIVOT = "pivot"                  # Change direction
    VERIFY = "verify"                # Check previous work
    CONCLUDE = "conclude"            # Draw final conclusion
    QUESTION = "question"            # Raise new questions
    THEREFORE = "therefore"          # Logical consequence
    HOWEVER = "however"              # Contrast/exception
    BUILDING_ON = "building_on"      # Building on previous
    ALTERNATIVELY = "alternatively"  # Alternative approach


class OutputFormat(Enum):
    """τ: How to structure the output"""
    LIST = "list"                    # Bullet points
    STEPS = "steps"                  # Numbered sequence
    COMPARISON = "comparison"        # Side-by-side comparison
    EXPLANATION = "explanation"      # Narrative explanation
    SOLUTION = "solution"            # Direct answer
    CODE = "code"                    # Programming code
    MATHEMATICAL = "mathematical"    # Mathematical notation
    FREEFORM = "free-form"           # Unstructured
    NARRATIVE = "narrative"          # Story-like narrative
    TABLE = "table"                  # Tabular format


@dataclass
class ComposingPrompt:
    """
    Fluid API for building compositional prompts.

    Usage:
        prompt = (ComposingPrompt()
                 .cognitive_op(CognitiveOperation.DECOMPOSE)
                 .focus(FocusAspect.STRUCTURE)
                 .style(ReasoningStyle.SYSTEMATIC)
                 .connect(ConnectionType.THEREFORE)
                 .format(OutputFormat.STEPS)
                 .problem_context("Solve x^2 + 5x + 6 = 0")
                 .llm_augment("consider multiple solution methods")
                 .build())
    """

    # Core compositional components
    _cognitive_op: Optional[CognitiveOperation] = None
    _focus: Optional[FocusAspect] = None
    _style: Optional[ReasoningStyle] = None
    _connection: Optional[ConnectionType] = None
    _output_format: Optional[OutputFormat] = None

    # LLM enhancement chain
    _llm_augmentations: List[Dict[str, Any]] = field(default_factory=list)
    _examples: List[str] = field(default_factory=list)
    _coherence_checks: bool = False
    _context_additions: List[str] = field(default_factory=list)

    # Base prompt content
    _base_prompt: str = ""
    _problem_context: str = ""

    # Core compositional action methods
    def cognitive_op(self, operation: Union[CognitiveOperation, str]) -> 'ComposingPrompt':
        """Set the cognitive operation (ω)"""
        if isinstance(operation, str):
            operation = CognitiveOperation(operation)
        self._cognitive_op = operation
        return self

    def focus(self, aspect: Union[FocusAspect, str]) -> 'ComposingPrompt':
        """Set the focus aspect (φ)"""
        if isinstance(aspect, str):
            aspect = FocusAspect(aspect)
        self._focus = aspect
        return self

    def style(self, reasoning_style: Union[ReasoningStyle, str]) -> 'ComposingPrompt':
        """Set the reasoning style (σ)"""
        if isinstance(reasoning_style, str):
            reasoning_style = ReasoningStyle(reasoning_style)
        self._style = reasoning_style
        return self

    def connect(self, connection_type: Union[ConnectionType, str]) -> 'ComposingPrompt':
        """Set the connection type (κ)"""
        if isinstance(connection_type, str):
            connection_type = ConnectionType(connection_type)
        self._connection = connection_type
        return self

    def format(self, output_format: Union[OutputFormat, str]) -> 'ComposingPrompt':
        """Set the output format (τ)"""
        if isinstance(output_format, str):
            output_format = OutputFormat(output_format)
        self._output_format = output_format
        return self

    def problem_context(self, context: str) -> 'ComposingPrompt':
        """Set the problem context"""
        self._problem_context = context
        return self

    def base_prompt(self, prompt: str) -> 'ComposingPrompt':
        """Set the base prompt content"""
        self._base_prompt = prompt
        return self

    # LLM meta-reasoning enhancement methods
    def llm_augment(self, instruction: str) -> 'ComposingPrompt':
        """Add LLM-based augmentation instruction"""
        self._llm_augmentations.append({
            'type': 'augment',
            'instruction': instruction
        })
        return self

    def llm_coherence_check(self) -> 'ComposingPrompt':
        """Enable LLM-based coherence checking"""
        self._coherence_checks = True
        return self

    def llm_add_examples(self, n: int = 2, domain: str = None, parallel: bool = True) -> 'ComposingPrompt':
        """Request LLM to generate relevant examples"""
        if domain:
            instruction = f"Generate {n} relevant examples from the {domain} domain"
        else:
            instruction = f"Generate {n} relevant examples"

        self._llm_augmentations.append({
            'type': 'examples',
            'instruction': instruction,
            'n': n,
            'domain': domain,
            'parallel': parallel
        })
        return self

    def add_context(self, context: str) -> 'ComposingPrompt':
        """Add additional context"""
        self._context_additions.append(context)
        return self

    def with_examples(self, examples: List[Any],
                     include_steps: bool = True) -> 'ComposingPrompt':
        """
        Add examples for few-shot learning.

        Args:
            examples: List of Example objects or formatted example strings
            include_steps: Whether to include reasoning steps in examples

        Returns:
            self (for chaining)
        """
        from .examples import Example

        for example in examples:
            if isinstance(example, Example):
                self._examples.append(
                    example.to_prompt_string(include_steps=include_steps)
                )
            elif isinstance(example, str):
                self._examples.append(example)
            else:
                raise TypeError(f"Examples must be Example objects or strings, got {type(example)}")

        return self

    def with_rag_examples(self, rag_store, n: int = 3,
                         include_steps: bool = True) -> 'ComposingPrompt':
        """
        Add examples retrieved from a RAG store.

        Args:
            rag_store: SolutionRAGStore to retrieve examples from
            n: Number of examples to retrieve
            include_steps: Whether to include reasoning steps

        Returns:
            self (for chaining)
        """
        from .rag import SolutionRAGStore

        if not isinstance(rag_store, SolutionRAGStore):
            raise TypeError("rag_store must be a SolutionRAGStore")

        if self._problem_context:
            # Retrieve similar examples
            examples = rag_store.retrieve(self._problem_context, k=n)
            self.with_examples(examples, include_steps=include_steps)

        return self

    def with_rag_guidance(self, rag_store) -> 'ComposingPrompt':
        """
        Apply compositional guidance from a RAG store.

        This uses CompositionalRAGStore to set the compositional dimensions
        based on the problem type.

        Args:
            rag_store: CompositionalRAGStore to get guidance from

        Returns:
            self (for chaining)
        """
        from .rag import CompositionalRAGStore

        if not isinstance(rag_store, CompositionalRAGStore):
            raise TypeError("rag_store must be a CompositionalRAGStore")

        if self._problem_context:
            # Get recommended weights and sample with them
            weights = rag_store.get_recommended_weights(self._problem_context)
            if weights:
                self.sample_weighted(weights)

        return self

    def sample_weighted(self, weights: Optional[Dict[str, Dict[Any, float]]] = None) -> 'ComposingPrompt':
        """Sample action components using optional weights for biased exploration"""

        def _sample_weighted_component(options: List[Any], component_weights: Dict[Any, float]) -> Any:
            if not component_weights:
                return random.choice(options)

            # Normalize weights
            total_weight = sum(component_weights.get(opt, 1.0) for opt in options)
            probs = [component_weights.get(opt, 1.0) / total_weight for opt in options]

            return random.choices(options, weights=probs)[0]

        if weights:
            # Sample each component with weights
            if 'cognitive_op' in weights:
                self._cognitive_op = _sample_weighted_component(
                    list(CognitiveOperation), weights['cognitive_op']
                )

            if 'focus' in weights:
                self._focus = _sample_weighted_component(
                    list(FocusAspect), weights['focus']
                )

            if 'style' in weights:
                self._style = _sample_weighted_component(
                    list(ReasoningStyle), weights['style']
                )

            if 'connection' in weights:
                self._connection = _sample_weighted_component(
                    list(ConnectionType), weights['connection']
                )

            if 'output_format' in weights:
                self._output_format = _sample_weighted_component(
                    list(OutputFormat), weights['output_format']
                )
        else:
            # Uniform random sampling
            self._cognitive_op = random.choice(list(CognitiveOperation))
            self._focus = random.choice(list(FocusAspect))
            self._style = random.choice(list(ReasoningStyle))
            self._connection = random.choice(list(ConnectionType))
            self._output_format = random.choice(list(OutputFormat))

        return self

    @classmethod
    def sample_action(cls, weights: Optional[Dict[str, Dict[Any, float]]] = None) -> 'ComposingPrompt':
        """Factory method to create a randomly sampled compositional action"""
        return cls().sample_weighted(weights)

    # Build methods
    def _build_compositional_core(self) -> str:
        """Build the core compositional prompt from (ω,φ,σ,κ,τ)"""
        parts = []

        if self._cognitive_op:
            if self._cognitive_op == CognitiveOperation.DECOMPOSE:
                parts.append("Let me break this problem down systematically.")
            elif self._cognitive_op == CognitiveOperation.ANALYZE:
                parts.append("Let me analyze this problem carefully.")
            elif self._cognitive_op == CognitiveOperation.GENERATE:
                parts.append("Let me generate a solution approach.")
            elif self._cognitive_op == CognitiveOperation.VERIFY:
                parts.append("Let me verify this reasoning step by step.")
            elif self._cognitive_op == CognitiveOperation.SYNTHESIZE:
                parts.append("Let me synthesize the key insights.")
            elif self._cognitive_op == CognitiveOperation.ABSTRACT:
                parts.append("Let me abstract the essential patterns.")
            elif self._cognitive_op == CognitiveOperation.CONCRETIZE:
                parts.append("Let me make this more concrete with specific examples.")
            elif self._cognitive_op == CognitiveOperation.COMPARE:
                parts.append("Let me compare different approaches.")
            elif self._cognitive_op == CognitiveOperation.EVALUATE:
                parts.append("Let me evaluate the quality of this approach.")
            elif self._cognitive_op == CognitiveOperation.REFINE:
                parts.append("Let me refine this solution.")

        if self._focus:
            if self._focus == FocusAspect.STRUCTURE:
                parts.append("I'll focus on the structural relationships and organization.")
            elif self._focus == FocusAspect.CONSTRAINTS:
                parts.append("I'll focus on the constraints and limitations.")
            elif self._focus == FocusAspect.PATTERNS:
                parts.append("I'll focus on identifying key patterns.")
            elif self._focus == FocusAspect.SOLUTION:
                parts.append("I'll focus on developing a clear solution.")
            elif self._focus == FocusAspect.CORRECTNESS:
                parts.append("I'll focus on ensuring correctness.")
            elif self._focus == FocusAspect.EFFICIENCY:
                parts.append("I'll focus on efficiency and optimization.")
            elif self._focus == FocusAspect.DETAILS:
                parts.append("I'll focus on the specific details.")
            elif self._focus == FocusAspect.ASSUMPTIONS:
                parts.append("I'll focus on the underlying assumptions.")
            elif self._focus == FocusAspect.GOAL:
                parts.append("I'll focus on the end goal.")
            elif self._focus == FocusAspect.PROGRESS:
                parts.append("I'll focus on our current progress.")
            elif self._focus == FocusAspect.ERRORS:
                parts.append("I'll focus on potential errors or issues.")
            elif self._focus == FocusAspect.ALTERNATIVES:
                parts.append("I'll focus on alternative approaches.")

        if self._style:
            if self._style == ReasoningStyle.SYSTEMATIC:
                parts.append("I'll approach this systematically and methodically.")
            elif self._style == ReasoningStyle.CREATIVE:
                parts.append("I'll approach this with creative thinking.")
            elif self._style == ReasoningStyle.CRITICAL:
                parts.append("I'll approach this with critical analysis.")
            elif self._style == ReasoningStyle.FORMAL:
                parts.append("I'll approach this with formal rigor.")
            elif self._style == ReasoningStyle.INTUITIVE:
                parts.append("I'll approach this with intuitive reasoning.")
            elif self._style == ReasoningStyle.EXPLORATORY:
                parts.append("I'll approach this through open exploration.")

        if self._connection:
            if self._connection == ConnectionType.THEREFORE:
                parts.append("Therefore,")
            elif self._connection == ConnectionType.HOWEVER:
                parts.append("However,")
            elif self._connection == ConnectionType.BUILDING_ON:
                parts.append("Building on the previous analysis,")
            elif self._connection == ConnectionType.ALTERNATIVELY:
                parts.append("Alternatively,")
            elif self._connection == ConnectionType.VERIFY:
                parts.append("To verify this,")
            elif self._connection == ConnectionType.CONTINUE:
                parts.append("Continuing from where we left off,")
            elif self._connection == ConnectionType.CONTRAST:
                parts.append("In contrast,")
            elif self._connection == ConnectionType.ELABORATE:
                parts.append("To elaborate further,")
            elif self._connection == ConnectionType.SUMMARIZE:
                parts.append("To summarize,")
            elif self._connection == ConnectionType.PIVOT:
                parts.append("Let's pivot to")
            elif self._connection == ConnectionType.CONCLUDE:
                parts.append("In conclusion,")
            elif self._connection == ConnectionType.QUESTION:
                parts.append("This raises the question:")

        if self._output_format:
            if self._output_format == OutputFormat.STEPS:
                parts.append("I'll present this as clear steps:")
            elif self._output_format == OutputFormat.LIST:
                parts.append("I'll present this as a structured list:")
            elif self._output_format == OutputFormat.MATHEMATICAL:
                parts.append("I'll present this with mathematical notation:")
            elif self._output_format == OutputFormat.NARRATIVE:
                parts.append("I'll present this as a clear narrative:")
            elif self._output_format == OutputFormat.CODE:
                parts.append("I'll present this as code:")
            elif self._output_format == OutputFormat.SOLUTION:
                parts.append("Here's the solution:")
            elif self._output_format == OutputFormat.EXPLANATION:
                parts.append("Here's a detailed explanation:")
            elif self._output_format == OutputFormat.COMPARISON:
                parts.append("Here's a comparison:")
            elif self._output_format == OutputFormat.TABLE:
                parts.append("Here's a table:")
            elif self._output_format == OutputFormat.FREEFORM:
                parts.append("")  # No format constraint

        return " ".join(parts)

    def build(self) -> str:
        """Build the final prompt"""
        components = []

        # Add base prompt if provided
        if self._base_prompt:
            components.append(self._base_prompt)

        # Add problem context if provided
        if self._problem_context:
            components.append(f"Problem: {self._problem_context}")

        # Add compositional core
        core = self._build_compositional_core()
        if core:
            components.append(core)

        # Add context additions
        if self._context_additions:
            components.extend(self._context_additions)

        # Add examples if any
        if self._examples:
            components.append("Relevant examples:")
            components.extend(self._examples)

        # Note augmentations (for debugging/tracking)
        if self._llm_augmentations:
            aug_descriptions = [aug['instruction'] for aug in self._llm_augmentations]
            components.append(f"[Augmentations: {', '.join(aug_descriptions)}]")

        if self._coherence_checks:
            components.append("[Apply coherence checking]")

        return "\n\n".join(components)

    def get_action_vector(self) -> Dict[str, Any]:
        """Get the compositional action as a structured vector"""
        return {
            'omega': self._cognitive_op.value if self._cognitive_op else None,
            'phi': self._focus.value if self._focus else None,
            'sigma': self._style.value if self._style else None,
            'kappa': self._connection.value if self._connection else None,
            'tau': self._output_format.value if self._output_format else None,
            'llm_augmentations': len(self._llm_augmentations),
            'coherence_checks': self._coherence_checks,
            'context_additions': len(self._context_additions),
            'has_parallel_ops': any(aug.get('parallel', False) for aug in self._llm_augmentations)
        }

    def to_tuple(self) -> tuple:
        """Convert to (ω, φ, σ, κ, τ) tuple for hashing/comparison"""
        return (
            self._cognitive_op,
            self._focus,
            self._style,
            self._connection,
            self._output_format
        )

    def __hash__(self):
        """Make hashable for use in dictionaries"""
        return hash(self.to_tuple())

    def __eq__(self, other):
        """Equality based on compositional components"""
        if not isinstance(other, ComposingPrompt):
            return False
        return self.to_tuple() == other.to_tuple()

    def __repr__(self):
        """String representation"""
        return f"ComposingPrompt({self._cognitive_op}, {self._focus}, {self._style})"


def smart_termination(state: str, llm_provider=None, pattern_only: bool = False) -> bool:
    """
    Smart termination detection combining pattern matching and LLM reasoning.

    Args:
        state: Current reasoning state
        llm_provider: Optional LLM provider for semantic checking
        pattern_only: If True, only use pattern matching

    Returns:
        True if state represents a complete/terminal solution
    """
    # Classical termination patterns
    termination_patterns = [
        r'\bfinal answer:?\s*(.+)', r'\bconclusion:?\s*(.+)',
        r'\btherefore,?\s+the answer is\s+(.+)', r'\bso,?\s+the solution is\s+(.+)',
        r'\bhence,?\s+(.+)', r'\bin conclusion,?\s+(.+)',
        r'\bthe result is\s+(.+)', r'\bthe answer is\s+(.+)',
        r'\bwe conclude that\s+(.+)', r'\bthus,?\s+(.+)',
        r'\bQED\b', r'\bproven\b', r'\bsolved\b',
        r'##\s*(answer|solution|conclusion)'
    ]

    for pattern in termination_patterns:
        if re.search(pattern, state.lower(), re.IGNORECASE):
            return True

    if pattern_only or not llm_provider:
        return False

    # LLM-based check for subtle completeness
    termination_prompt = f"""
Analyze this reasoning output and determine if it represents a complete answer.

Consider:
- Does it provide a definitive answer?
- Is the reasoning chain complete?
- Would a human consider this final?

Respond with only "TERMINAL" or "CONTINUE".

Output to analyze:
{state[-500:]}

Response:"""

    try:
        response = llm_provider.generate(termination_prompt, max_tokens=10, temperature=0.1)
        return "TERMINAL" in response.upper()
    except Exception:
        pass

    return len(state) > 2000  # Fallback to length check


# MCP imports (optional, only if needed)
try:
    from .mcp import (
        MCPToolType,
        MCPTool,
        MCPToolCall,
        MCPToolResult,
        MCPClient,
        MCPLLMProvider,
        create_mcp_client,
        create_mcp_provider,
    )
    from .mcp_actions import (
        MCPActionIntent,
        MCPCompositionalAction,
        MCPActionSelector,
        create_mcp_action,
        create_code_execution_action,
        create_research_action,
    )
    _has_mcp = True
except ImportError:
    _has_mcp = False


# Export all public components
__all__ = [
    # Enums
    'CognitiveOperation',
    'FocusAspect',
    'ReasoningStyle',
    'ConnectionType',
    'OutputFormat',

    # Core classes
    'ComposingPrompt',

    # Utility functions
    'smart_termination',

    # Examples and RAG (lazy imports to avoid circular dependencies)
    # from .examples import Example, ExampleSet, get_math_examples, etc.
    # from .rag import CompositionalRAGStore, SolutionRAGStore, etc.
]

# Add MCP exports if available
if _has_mcp:
    __all__.extend([
        # MCP Enums
        'MCPToolType',
        'MCPActionIntent',

        # MCP Core
        'MCPTool',
        'MCPToolCall',
        'MCPToolResult',
        'MCPClient',
        'MCPLLMProvider',

        # MCP Actions
        'MCPCompositionalAction',
        'MCPActionSelector',

        # MCP Utilities
        'create_mcp_client',
        'create_mcp_provider',
        'create_mcp_action',
        'create_code_execution_action',
        'create_research_action',
    ])
