"""Tests for the actions module."""

import pytest
from mcts_reasoning.actions import (
    Action,
    ActionResult,
    ActionSpace,
    ContinueAction,
    DefaultActionSpace,
    CompressAction,
    ExtendedActionSpace,
)
from mcts_reasoning.generator import MockGenerator


class MockLLM:
    """Mock LLM for testing actions."""

    def __init__(self, response: str = "Next step: analyzing..."):
        self.response = response
        self.calls = []

    def generate(self, prompt: str, **kwargs) -> str:
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        return self.response


class TestActionResult:
    """Tests for ActionResult dataclass."""

    def test_create_basic_result(self):
        result = ActionResult(new_state="state", is_terminal=False)
        assert result.new_state == "state"
        assert result.is_terminal is False
        assert result.answer is None

    def test_create_terminal_result(self):
        result = ActionResult(new_state="state", is_terminal=True, answer="42")
        assert result.is_terminal is True
        assert result.answer == "42"


class TestContinueAction:
    """Tests for ContinueAction."""

    def test_action_name(self):
        action = ContinueAction(llm=MockLLM())
        assert action.name == "CONTINUE"

    def test_apply_non_terminal_with_llm(self):
        """Test ContinueAction with raw LLM."""
        llm = MockLLM(response="Step 2: Let me think...")
        action = ContinueAction(llm=llm)

        result = action.apply("What is 2+2?", "Step 1: This is addition.")

        assert "Step 1: This is addition." in result.new_state
        assert "Step 2: Let me think..." in result.new_state
        assert result.is_terminal is False
        assert result.answer is None

    def test_apply_terminal_with_llm(self):
        """Test terminal detection with raw LLM."""
        llm = MockLLM(response="Therefore, ANSWER: 4")
        action = ContinueAction(llm=llm)

        result = action.apply("What is 2+2?", "Step 1: This is addition.")

        assert result.is_terminal is True
        assert result.answer == "4"

    def test_apply_with_generator(self):
        """Test ContinueAction with Generator (preferred path)."""
        generator = MockGenerator(responses=["Step 2: Working on it..."])
        action = ContinueAction(generator=generator)

        result = action.apply("What is 2+2?", "Step 1: Start")

        assert "Step 2: Working on it..." in result.new_state
        assert result.is_terminal is False

    def test_apply_uses_llm(self):
        llm = MockLLM()
        action = ContinueAction(llm=llm)

        action.apply("question", "state")

        assert len(llm.calls) == 1
        assert "question" in llm.calls[0]["prompt"]
        assert "state" in llm.calls[0]["prompt"]

    def test_custom_prompt_template(self):
        llm = MockLLM()
        action = ContinueAction(llm=llm, prompt_template="Q: {question}\nS: {state}\nNext:")

        action.apply("What?", "Thinking...")

        assert "Q: What?" in llm.calls[0]["prompt"]
        assert "S: Thinking..." in llm.calls[0]["prompt"]

    def test_temperature_and_max_tokens(self):
        llm = MockLLM()
        action = ContinueAction(llm=llm, temperature=0.5, max_tokens=100)

        action.apply("q", "s")

        assert llm.calls[0]["kwargs"]["temperature"] == 0.5
        assert llm.calls[0]["kwargs"]["max_tokens"] == 100

    def test_extract_answer_various_formats(self):
        action = ContinueAction(llm=MockLLM())

        # Standard format
        assert action._extract_answer("blah ANSWER: 42") == "42"

        # With extra whitespace
        assert action._extract_answer("blah ANSWER:   42  ") == "42"

        # With newline after
        assert action._extract_answer("blah ANSWER: 42\n\nMore text") == "42"

        # No answer
        assert action._extract_answer("no answer here") is None

    def test_requires_generator_or_llm(self):
        """Test that ContinueAction raises if neither generator nor llm provided."""
        action = ContinueAction()  # No generator or llm

        with pytest.raises(ValueError, match="requires either a generator or llm"):
            action.apply("q", "s")


class TestDefaultActionSpace:
    """Tests for DefaultActionSpace."""

    def test_non_terminal_returns_continue(self):
        generator = MockGenerator()
        space = DefaultActionSpace(generator=generator)
        actions = space.get_actions("some state", is_terminal=False)

        assert len(actions) == 1
        assert actions[0].name == "CONTINUE"

    def test_terminal_returns_empty(self):
        generator = MockGenerator()
        space = DefaultActionSpace(generator=generator)
        actions = space.get_actions("final state", is_terminal=True)

        assert len(actions) == 0

    def test_custom_continue_action(self):
        custom_continue = ContinueAction(llm=MockLLM(), temperature=0.9)
        space = DefaultActionSpace(continue_action=custom_continue)

        actions = space.get_actions("state", is_terminal=False)
        assert actions[0].temperature == 0.9


class TestCompressAction:
    """Tests for CompressAction (extension)."""

    def test_action_name(self):
        llm = MockLLM()
        action = CompressAction(llm=llm)
        assert action.name == "COMPRESS"

    def test_is_available_short_trace(self):
        llm = MockLLM()
        action = CompressAction(llm=llm, threshold=100)
        assert action.is_available("short") is False

    def test_is_available_long_trace(self):
        llm = MockLLM()
        action = CompressAction(llm=llm, threshold=100)
        long_trace = "x" * 200
        assert action.is_available(long_trace) is True

    def test_apply_creates_compressed_state(self):
        llm = MockLLM(response="Summary: key insights here")
        action = CompressAction(llm=llm)

        result = action.apply("question", "long trace...")

        assert "[COMPRESSED REASONING]" in result.new_state
        assert "Summary: key insights here" in result.new_state
        assert "[/COMPRESSED]" in result.new_state
        assert result.is_terminal is False  # Compression never terminates


class TestExtendedActionSpace:
    """Tests for ExtendedActionSpace (extension)."""

    def test_short_trace_only_continue(self):
        llm = MockLLM()
        space = ExtendedActionSpace(llm=llm)
        actions = space.get_actions("short trace", is_terminal=False)

        assert len(actions) == 1
        assert actions[0].name == "CONTINUE"

    def test_long_trace_has_compress(self):
        llm = MockLLM()
        space = ExtendedActionSpace(llm=llm, compress_threshold=10)

        long_trace = "x" * 100
        actions = space.get_actions(long_trace, is_terminal=False)

        assert len(actions) == 2
        action_names = [a.name for a in actions]
        assert "CONTINUE" in action_names
        assert "COMPRESS" in action_names

    def test_terminal_returns_empty(self):
        llm = MockLLM()
        space = ExtendedActionSpace(llm=llm)
        actions = space.get_actions("state", is_terminal=True)

        assert len(actions) == 0

    def test_with_generator(self):
        """Test ExtendedActionSpace with Generator."""
        generator = MockGenerator()
        llm = MockLLM()
        space = ExtendedActionSpace(generator=generator, llm=llm)

        actions = space.get_actions("state", is_terminal=False)
        assert len(actions) == 1
        assert actions[0].name == "CONTINUE"

    def test_requires_generator_or_llm(self):
        """Test that ExtendedActionSpace raises without generator or llm."""
        with pytest.raises(ValueError, match="requires generator or llm"):
            ExtendedActionSpace()


class TestActionProtocol:
    """Tests that custom actions work with the protocol."""

    def test_custom_action_satisfies_protocol(self):
        class VerifyAction:
            @property
            def name(self) -> str:
                return "VERIFY"

            def apply(self, question: str, state: str) -> ActionResult:
                return ActionResult(
                    new_state=f"{state}\n[Verified]",
                    is_terminal=False,
                )

        action = VerifyAction()
        assert isinstance(action, Action)

        result = action.apply("q", "s")
        assert "[Verified]" in result.new_state
