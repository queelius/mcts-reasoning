"""
MCP-Aware Compositional Actions

Extends the compositional action system with MCP tool awareness.
These actions explicitly guide the LLM to use specific MCP tools.

For example:
- "Analyze with Python execution" → Encourages using execute_python tool
- "Verify with code" → Suggests testing the solution with code
- "Research with web search" → Prompts using web search tool
"""

from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from enum import Enum
import random

from . import (
    CognitiveOperation,
    FocusAspect,
    ReasoningStyle,
    ConnectionType,
    OutputFormat,
    ComposingPrompt
)
from .actions import CompositionalAction, ActionSelector
from .mcp import MCPToolType, MCPClient
from .providers import LLMProvider


class MCPActionIntent(Enum):
    """Specific intents for MCP tool usage."""
    EXECUTE_CODE = "execute_code"
    TEST_SOLUTION = "test_solution"
    CALCULATE = "calculate"
    RESEARCH = "research"
    VERIFY_FACTS = "verify_facts"
    READ_DATA = "read_data"
    WRITE_RESULTS = "write_results"
    NONE = "none"  # No specific MCP intent


@dataclass
class MCPCompositionalAction(CompositionalAction):
    """
    Compositional action with MCP tool awareness.

    Extends CompositionalAction to explicitly encourage MCP tool usage
    in the prompt.
    """

    mcp_intent: MCPActionIntent = MCPActionIntent.NONE
    suggested_tools: List[str] = None

    def __post_init__(self):
        """Initialize suggested tools list."""
        if self.suggested_tools is None:
            self.suggested_tools = []

    def to_prompt(self, current_state: str, original_question: str,
                  previous_response: Optional[str] = None,
                  available_tools: Optional[List[str]] = None) -> str:
        """
        Build prompt with MCP tool encouragement.

        Args:
            current_state: Current reasoning state
            original_question: Original question
            previous_response: Previous reasoning step
            available_tools: List of available tool names

        Returns:
            Enhanced prompt with tool suggestions
        """
        # Get base prompt from parent
        base_prompt = super().to_prompt(current_state, original_question, previous_response)

        # Add MCP tool guidance if intent is set
        if self.mcp_intent != MCPActionIntent.NONE:
            tool_guidance = self._get_tool_guidance(available_tools)
            if tool_guidance:
                base_prompt += f"\n\n{tool_guidance}"

        return base_prompt

    def _get_tool_guidance(self, available_tools: Optional[List[str]] = None) -> str:
        """Generate tool usage guidance based on intent."""
        guidance_map = {
            MCPActionIntent.EXECUTE_CODE: """
Consider using the execute_python tool to test your ideas with code.
You can write and run Python to:
- Verify calculations
- Test algorithms
- Generate examples
- Validate logic
""",
            MCPActionIntent.TEST_SOLUTION: """
Test your solution by writing Python code with execute_python.
Create test cases to verify correctness:
- Edge cases
- Normal cases
- Error conditions
""",
            MCPActionIntent.CALCULATE: """
For numerical calculations, use the execute_python tool.
Write Python code to compute the result accurately.
""",
            MCPActionIntent.RESEARCH: """
If you need external information, use the search_web tool.
Search for:
- Facts and definitions
- Examples and use cases
- Related concepts
""",
            MCPActionIntent.VERIFY_FACTS: """
Verify factual claims using the search_web tool.
Look up information to confirm:
- Historical facts
- Scientific data
- Technical specifications
""",
            MCPActionIntent.READ_DATA: """
If relevant data files exist, use the read_file tool.
Read files to:
- Load datasets
- Access configurations
- Review documentation
""",
            MCPActionIntent.WRITE_RESULTS: """
Consider writing results to a file with write_file.
Save:
- Final solutions
- Intermediate results
- Generated code or data
"""
        }

        guidance = guidance_map.get(self.mcp_intent, "")

        # Add specific tool suggestions if provided
        if self.suggested_tools:
            tools_list = ", ".join(self.suggested_tools)
            guidance += f"\n\nSuggested tools: {tools_list}"

        # Filter by available tools if provided
        if available_tools and self.suggested_tools:
            available_suggested = [t for t in self.suggested_tools if t in available_tools]
            if available_suggested and available_suggested != self.suggested_tools:
                tools_list = ", ".join(available_suggested)
                guidance += f"\n(Available: {tools_list})"

        return guidance.strip()


class MCPActionSelector(ActionSelector):
    """
    Action selector with MCP awareness.

    Extends ActionSelector to generate MCP-aware actions based on
    available tools and reasoning context.
    """

    def __init__(self, exploration_constant: float = 1.414,
                 use_compatibility_rules: bool = True,
                 mcp_client: Optional[MCPClient] = None):
        """
        Initialize MCP-aware action selector.

        Args:
            exploration_constant: UCB1 exploration parameter
            use_compatibility_rules: Whether to enforce compatibility rules
            mcp_client: MCP client for tool availability info
        """
        super().__init__(exploration_constant, use_compatibility_rules)
        self.mcp_client = mcp_client

    def get_mcp_actions(self, current_state: str,
                       previous_action: Optional[CompositionalAction] = None,
                       n_samples: int = 10) -> List[MCPCompositionalAction]:
        """
        Get MCP-aware compositional actions.

        Args:
            current_state: Current reasoning state
            previous_action: Previous action taken
            n_samples: Number of actions to generate

        Returns:
            List of MCPCompositionalAction objects
        """
        actions = []

        # Get available tools if MCP client is connected
        available_tools = []
        if self.mcp_client:
            available_tools = list(self.mcp_client.tools.keys())

        # Generate a mix of MCP-aware and regular actions
        mcp_action_ratio = 0.4  # 40% of actions have MCP intent

        for i in range(n_samples):
            # Decide if this should be an MCP-aware action
            use_mcp = (i < n_samples * mcp_action_ratio) and available_tools

            if use_mcp:
                action = self._generate_mcp_action(current_state, previous_action, available_tools)
            else:
                # Generate regular compositional action
                base_action = self.get_valid_actions(
                    current_state, previous_action, n_samples=1
                )[0]

                # Wrap in MCP action with no intent
                action = MCPCompositionalAction(
                    operation=base_action.operation,
                    focus=base_action.focus,
                    style=base_action.style,
                    connection=base_action.connection,
                    output_format=base_action.output_format,
                    mcp_intent=MCPActionIntent.NONE
                )

            actions.append(action)

        return actions

    def _generate_mcp_action(self, current_state: str,
                            previous_action: Optional[CompositionalAction],
                            available_tools: List[str]) -> MCPCompositionalAction:
        """Generate a single MCP-aware action."""
        # Map operations to likely MCP intents
        operation_intent_map = {
            CognitiveOperation.VERIFY: [MCPActionIntent.TEST_SOLUTION, MCPActionIntent.VERIFY_FACTS],
            CognitiveOperation.EVALUATE: [MCPActionIntent.EXECUTE_CODE, MCPActionIntent.TEST_SOLUTION],
            CognitiveOperation.GENERATE: [MCPActionIntent.EXECUTE_CODE],
            CognitiveOperation.ANALYZE: [MCPActionIntent.CALCULATE, MCPActionIntent.READ_DATA],
            CognitiveOperation.ABSTRACT: [MCPActionIntent.RESEARCH],
            CognitiveOperation.COMPARE: [MCPActionIntent.RESEARCH, MCPActionIntent.VERIFY_FACTS],
        }

        # Choose operation
        operation = random.choice(list(CognitiveOperation))

        # Get compatible MCP intent
        possible_intents = operation_intent_map.get(
            operation,
            [MCPActionIntent.EXECUTE_CODE, MCPActionIntent.RESEARCH]
        )
        mcp_intent = random.choice(possible_intents)

        # Map intent to tools
        intent_tools_map = {
            MCPActionIntent.EXECUTE_CODE: ["execute_python"],
            MCPActionIntent.TEST_SOLUTION: ["execute_python"],
            MCPActionIntent.CALCULATE: ["execute_python"],
            MCPActionIntent.RESEARCH: ["search_web"],
            MCPActionIntent.VERIFY_FACTS: ["search_web"],
            MCPActionIntent.READ_DATA: ["read_file"],
            MCPActionIntent.WRITE_RESULTS: ["write_file"],
        }

        suggested_tools = intent_tools_map.get(mcp_intent, [])

        # Filter to available tools
        suggested_tools = [t for t in suggested_tools if t in available_tools]

        # Generate other components using base logic
        focus = random.choice(list(FocusAspect))
        style = random.choice(list(ReasoningStyle))
        connection = self._get_connection_type(operation, previous_action)
        output_format = self._get_output_format(operation, focus)

        return MCPCompositionalAction(
            operation=operation,
            focus=focus,
            style=style,
            connection=connection,
            output_format=output_format,
            mcp_intent=mcp_intent,
            suggested_tools=suggested_tools
        )


# ========== Factory Functions ==========

def create_mcp_action(operation: CognitiveOperation,
                     focus: FocusAspect,
                     style: ReasoningStyle,
                     mcp_intent: MCPActionIntent,
                     suggested_tools: Optional[List[str]] = None,
                     **kwargs) -> MCPCompositionalAction:
    """
    Factory function to create MCP-aware actions.

    Example:
        action = create_mcp_action(
            operation=CognitiveOperation.VERIFY,
            focus=FocusAspect.CORRECTNESS,
            style=ReasoningStyle.SYSTEMATIC,
            mcp_intent=MCPActionIntent.TEST_SOLUTION,
            suggested_tools=["execute_python"]
        )
    """
    return MCPCompositionalAction(
        operation=operation,
        focus=focus,
        style=style,
        connection=kwargs.get('connection', ConnectionType.CONTINUE),
        output_format=kwargs.get('output_format', OutputFormat.STEPS),
        mcp_intent=mcp_intent,
        suggested_tools=suggested_tools or []
    )


def create_code_execution_action(operation: CognitiveOperation = CognitiveOperation.VERIFY,
                                focus: FocusAspect = FocusAspect.CORRECTNESS,
                                **kwargs) -> MCPCompositionalAction:
    """
    Create an action that encourages Python code execution.

    Example:
        action = create_code_execution_action()
    """
    return create_mcp_action(
        operation=operation,
        focus=focus,
        style=kwargs.get('style', ReasoningStyle.SYSTEMATIC),
        mcp_intent=MCPActionIntent.EXECUTE_CODE,
        suggested_tools=["execute_python"],
        **kwargs
    )


def create_research_action(operation: CognitiveOperation = CognitiveOperation.ANALYZE,
                          focus: FocusAspect = FocusAspect.PATTERNS,
                          **kwargs) -> MCPCompositionalAction:
    """
    Create an action that encourages web research.

    Example:
        action = create_research_action()
    """
    return create_mcp_action(
        operation=operation,
        focus=focus,
        style=kwargs.get('style', ReasoningStyle.EXPLORATORY),
        mcp_intent=MCPActionIntent.RESEARCH,
        suggested_tools=["search_web"],
        **kwargs
    )


# ========== Export All ==========

__all__ = [
    # Enums
    'MCPActionIntent',

    # Classes
    'MCPCompositionalAction',
    'MCPActionSelector',

    # Factory functions
    'create_mcp_action',
    'create_code_execution_action',
    'create_research_action',
]
