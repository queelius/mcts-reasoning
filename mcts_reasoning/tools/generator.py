"""
ToolAwareGenerator: Wraps a Generator to add tool support.

Intercepts generation to inject tool definitions and process tool calls.
Supports both parsing-based (XML/JSON) and native function calling.
"""

import logging
from typing import List, Optional, Any

from ..generator import Generator, Continuation
from ..terminal import TerminalDetector
from .context import ToolContext
from .formats import ToolCall, ToolResult

logger = logging.getLogger(__name__)


class ToolAwareGenerator(Generator):
    """
    Generator wrapper that adds tool support.

    Wraps any Generator and:
    1. Injects tool definitions into prompts (or uses native API)
    2. Parses tool calls from LLM output (or receives them natively)
    3. Executes tool calls and augments state with results
    4. Iterates until no more tool calls or max iterations reached

    Supports two modes:
    - Parsing mode: Injects XML/JSON tool definitions, parses tool calls from text
    - Native mode: Uses OpenAI/Anthropic native function calling APIs
    """

    def __init__(
        self,
        base_generator: Generator,
        tool_context: Optional[ToolContext] = None,
        terminal_detector: Optional[TerminalDetector] = None,
        native_provider: Optional[Any] = None,
        use_native: bool = True,
    ):
        """
        Initialize tool-aware generator.

        Args:
            base_generator: The underlying generator to wrap
            tool_context: Optional tool context (enables tools when provided)
            terminal_detector: Optional terminal detector (defaults to base generator's)
            native_provider: Optional NativeFunctionCallProvider for native tool calling
            use_native: Whether to use native function calling when available (default: True)
        """
        # Use base generator's terminal detector if not provided
        detector = terminal_detector or base_generator.terminal_detector
        super().__init__(detector)

        self.base_generator = base_generator
        self.tool_context = tool_context
        self.native_provider = native_provider
        self.use_native = use_native

        # Check if native provider supports tools
        self._native_available = False
        if native_provider is not None and use_native:
            try:
                from .native import NativeFunctionCallProvider
                if hasattr(native_provider, 'supports_native_tools'):
                    self._native_available = native_provider.supports_native_tools()
            except ImportError:
                pass

    def generate(
        self,
        question: str,
        state: str,
        n: int = 1,
    ) -> List[Continuation]:
        """
        Generate continuations with tool support.

        If tool_context is configured:
        1. Generate continuation from base generator (or native provider)
        2. Check for tool calls in the output (or receive them natively)
        3. Execute tool calls and augment state
        4. Repeat until no tool calls or terminal

        Args:
            question: The original question
            state: Current reasoning state
            n: Number of continuations to generate

        Returns:
            List of Continuation objects
        """
        # If no tool context, delegate directly to base generator
        if self.tool_context is None or not self.tool_context.has_tools:
            return self.base_generator.generate(question, state, n)

        # Generate with tool support
        continuations = []
        for _ in range(n):
            if self._native_available:
                continuation = self._generate_with_native_tools(question, state)
            else:
                continuation = self._generate_with_tools(question, state)
            continuations.append(continuation)

        return continuations

    def _generate_with_tools(
        self,
        question: str,
        state: str,
    ) -> Continuation:
        """Generate a single continuation with tool call handling (parsing mode)."""
        current_state = state
        iterations = 0
        max_iterations = self.tool_context.max_tool_iterations

        while iterations < max_iterations:
            iterations += 1

            # Generate from base generator
            continuations = self.base_generator.generate(question, current_state, n=1)
            continuation = continuations[0]

            # Check for tool calls in the new content
            # Extract just the new content (what was added to the state)
            new_content = continuation.text[len(current_state):].strip()

            # Process any tool calls
            result = self.tool_context.process_response(new_content)

            if not result.has_calls:
                # No tool calls - return the continuation as-is
                return continuation

            # Tool calls found - execute and augment state
            logger.debug(
                f"Iteration {iterations}: {len(result.calls)} tool calls (parsing mode)"
            )

            # Build augmented state with tool results
            augmented_state = self.tool_context.augment_state_with_results(
                continuation.text,
                result,
            )

            # Check if we should continue (not terminal and more iterations allowed)
            if continuation.is_terminal:
                # Terminal state reached - return with augmented state
                return Continuation(
                    text=augmented_state,
                    is_terminal=True,
                    answer=continuation.answer,
                )

            # Continue with augmented state for next iteration
            current_state = augmented_state

        # Max iterations reached - return current state
        return Continuation(
            text=current_state,
            is_terminal=self.is_terminal(current_state),
            answer=self.extract_answer(current_state),
        )

    def _generate_with_native_tools(
        self,
        question: str,
        state: str,
    ) -> Continuation:
        """Generate a single continuation with native function calling."""
        from .execution import ToolExecutionResult

        current_state = state
        iterations = 0
        max_iterations = self.tool_context.max_tool_iterations
        tools = self.tool_context.get_available_tools()

        while iterations < max_iterations:
            iterations += 1

            # Generate with native function calling
            prompt = f"{current_state}\n\nContinue reasoning about: {question}"
            text, tool_calls = self.native_provider.generate_with_tools(
                prompt=prompt,
                tools=tools,
            )

            # If no tool calls, build continuation from text
            if not tool_calls:
                new_state = f"{current_state}\n\n{text}" if text else current_state
                return Continuation(
                    text=new_state,
                    is_terminal=self.is_terminal(new_state),
                    answer=self.extract_answer(new_state),
                )

            # Execute tool calls
            logger.debug(
                f"Iteration {iterations}: {len(tool_calls)} tool calls (native mode)"
            )

            results = []
            for call in tool_calls:
                try:
                    result_value = self.tool_context._execute_tool(call.name, call.arguments)
                    results.append(ToolResult(
                        call_id=call.call_id,
                        name=call.name,
                        result=result_value,
                    ))
                except Exception as e:
                    results.append(ToolResult(
                        call_id=call.call_id,
                        name=call.name,
                        result=None,
                        error=str(e),
                    ))

            # Build augmented state
            exec_result = ToolExecutionResult(
                calls=tool_calls,
                results=results,
                remaining_text=text,
            )

            # Include text and tool results in state
            new_state = current_state
            if text:
                new_state = f"{new_state}\n\n{text}"

            formatted_results = exec_result.format_results(self.tool_context.fmt)
            if formatted_results:
                new_state = f"{new_state}\n\n[Tool Results]\n{formatted_results}"

            # Check if terminal
            if self.is_terminal(new_state):
                return Continuation(
                    text=new_state,
                    is_terminal=True,
                    answer=self.extract_answer(new_state),
                )

            current_state = new_state

        # Max iterations reached
        return Continuation(
            text=current_state,
            is_terminal=self.is_terminal(current_state),
            answer=self.extract_answer(current_state),
        )

    def inject_tools(self, prompt: str) -> str:
        """
        Inject tool definitions into a prompt.

        Args:
            prompt: Base prompt

        Returns:
            Prompt with tools injected (if tool_context configured)
        """
        if self.tool_context is None:
            return prompt
        return self.tool_context.inject_tools_into_prompt(prompt)


def wrap_generator_with_tools(
    generator: Generator,
    tool_context: ToolContext,
    native_provider: Optional[Any] = None,
    use_native: bool = True,
) -> ToolAwareGenerator:
    """
    Convenience function to wrap a generator with tool support.

    Args:
        generator: Generator to wrap
        tool_context: Tool context
        native_provider: Optional NativeFunctionCallProvider for native tool calling
        use_native: Whether to use native function calling when available

    Returns:
        ToolAwareGenerator wrapping the base generator
    """
    return ToolAwareGenerator(
        base_generator=generator,
        tool_context=tool_context,
        native_provider=native_provider,
        use_native=use_native,
    )


def create_native_tool_generator(
    generator: Generator,
    tool_context: ToolContext,
    llm_provider,
) -> ToolAwareGenerator:
    """
    Create a ToolAwareGenerator with native function calling support.

    Automatically wraps the LLM provider with native function calling
    if the provider supports it (OpenAI, Anthropic).

    Args:
        generator: Base generator to wrap
        tool_context: Tool context with available tools
        llm_provider: LLMProvider instance (OpenAIProvider or AnthropicProvider)

    Returns:
        ToolAwareGenerator with native function calling if supported

    Example:
        from mcts_reasoning import LLMGenerator, get_llm
        from mcts_reasoning.tools import ToolContext, create_native_tool_generator

        llm = get_llm("openai", model="gpt-4")
        generator = LLMGenerator(llm=llm)
        context = ToolContext.mock({"calc": {"response": "42"}})

        tool_gen = create_native_tool_generator(generator, context, llm)
    """
    from .native import wrap_provider_for_native_tools

    native_provider = wrap_provider_for_native_tools(llm_provider)

    return ToolAwareGenerator(
        base_generator=generator,
        tool_context=tool_context,
        native_provider=native_provider,
        use_native=native_provider is not None,
    )
