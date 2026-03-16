"""Tests for the prompt strategy module."""

import pytest
from mcts_reasoning.prompt import (
    PromptStrategy,
    StepByStepPrompt,
    Example,
    ExampleSource,
    StaticExampleSource,
    FewShotPrompt,
)
from mcts_reasoning.terminal import MarkerTerminalDetector, BoxedTerminalDetector
from mcts_reasoning.types import State


class TestStepByStepPrompt:
    """Tests for StepByStepPrompt."""

    def test_format_returns_list_of_messages(self):
        detector = MarkerTerminalDetector()
        prompt = StepByStepPrompt(terminal_detector=detector)

        messages = prompt.format("What is 2+2?", State("Initial state"))

        assert isinstance(messages, list)
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_format_single_includes_question_and_state(self):
        detector = MarkerTerminalDetector()
        prompt = StepByStepPrompt(terminal_detector=detector)

        messages = prompt.format("What is 2+2?", State("Let me think..."))

        user_content = messages[1]["content"]
        assert "What is 2+2?" in user_content
        assert "Let me think..." in user_content
        assert "ONE clear next step" in user_content

    def test_format_single_includes_terminal_instruction(self):
        detector = MarkerTerminalDetector(marker="ANSWER:")
        prompt = StepByStepPrompt(terminal_detector=detector)

        messages = prompt.format("q", State("s"))

        user_content = messages[1]["content"]
        assert "ANSWER:" in user_content

    def test_format_diverse_includes_continuation_markers(self):
        detector = MarkerTerminalDetector()
        prompt = StepByStepPrompt(terminal_detector=detector)

        messages = prompt.format("q", State("s"), n=3)

        user_content = messages[1]["content"]
        assert "3 DIFFERENT" in user_content
        assert "CONTINUATION 1" in user_content
        assert "CONTINUATION 2" in user_content

    def test_format_with_boxed_detector(self):
        detector = BoxedTerminalDetector()
        prompt = StepByStepPrompt(terminal_detector=detector)

        messages = prompt.format("q", State("s"))

        user_content = messages[1]["content"]
        assert "\\boxed" in user_content

    def test_parse_single_returns_response(self):
        detector = MarkerTerminalDetector()
        prompt = StepByStepPrompt(terminal_detector=detector)

        result = prompt.parse("Step 1: Let me calculate...", n=1)

        assert result == ["Step 1: Let me calculate..."]

    def test_parse_diverse_splits_on_markers(self):
        detector = MarkerTerminalDetector()
        prompt = StepByStepPrompt(terminal_detector=detector)

        response = (
            "--- CONTINUATION 1 ---\n"
            "First approach: add numbers\n\n"
            "--- CONTINUATION 2 ---\n"
            "Second approach: use counting\n\n"
            "--- CONTINUATION 3 ---\n"
            "Third approach: use algebra"
        )

        result = prompt.parse(response, n=3)

        assert len(result) == 3
        assert "First approach" in result[0]
        assert "Second approach" in result[1]
        assert "Third approach" in result[2]

    def test_parse_diverse_falls_back_if_no_markers(self):
        detector = MarkerTerminalDetector()
        prompt = StepByStepPrompt(terminal_detector=detector)

        response = "Just one continuation without markers"

        result = prompt.parse(response, n=3)

        assert result == ["Just one continuation without markers"]

    def test_parse_diverse_handles_extra_whitespace(self):
        detector = MarkerTerminalDetector()
        prompt = StepByStepPrompt(terminal_detector=detector)

        response = (
            "--- CONTINUATION 1 ---\n"
            "  First  \n\n"
            "---  CONTINUATION  2  ---\n"
            "  Second  "
        )

        result = prompt.parse(response, n=2)

        assert len(result) == 2
        assert result[0] == "First"
        assert result[1] == "Second"


class TestExample:
    """Tests for Example dataclass."""

    def test_example_basic(self):
        ex = Example(problem="What is 1+1?", solution="2")
        assert ex.problem == "What is 1+1?"
        assert ex.solution == "2"
        assert ex.reasoning is None

    def test_example_with_reasoning(self):
        ex = Example(
            problem="What is 1+1?",
            solution="2",
            reasoning="1 + 1 = 2",
        )
        assert ex.reasoning == "1 + 1 = 2"


class TestStaticExampleSource:
    """Tests for StaticExampleSource."""

    def test_returns_first_k(self):
        examples = [
            Example(problem=f"Q{i}", solution=f"A{i}") for i in range(5)
        ]
        source = StaticExampleSource(examples)

        result = source.find_similar("any question", k=3)

        assert len(result) == 3
        assert result[0].problem == "Q0"
        assert result[2].problem == "Q2"

    def test_returns_all_if_k_exceeds_length(self):
        examples = [Example(problem="Q1", solution="A1")]
        source = StaticExampleSource(examples)

        result = source.find_similar("any", k=5)

        assert len(result) == 1

    def test_returns_empty_if_no_examples(self):
        source = StaticExampleSource([])

        result = source.find_similar("any", k=3)

        assert result == []

    def test_satisfies_example_source_protocol(self):
        source = StaticExampleSource([])
        assert isinstance(source, ExampleSource)


class TestFewShotPrompt:
    """Tests for FewShotPrompt decorator."""

    def test_prepends_examples_as_pairs(self):
        detector = MarkerTerminalDetector()
        base = StepByStepPrompt(terminal_detector=detector)
        examples = StaticExampleSource([
            Example(problem="What is 1+1?", solution="2"),
            Example(problem="What is 3+3?", solution="6"),
        ])

        prompt = FewShotPrompt(base=base, examples=examples, k=2)
        messages = prompt.format("What is 2+2?", State("s"))

        # 2 examples * 2 messages each + 2 base messages = 6
        assert len(messages) == 6

        # First pair: example 1
        assert messages[0]["role"] == "user"
        assert "What is 1+1?" in messages[0]["content"]
        assert messages[1]["role"] == "assistant"
        assert "2" in messages[1]["content"]

        # Second pair: example 2
        assert messages[2]["role"] == "user"
        assert "What is 3+3?" in messages[2]["content"]
        assert messages[3]["role"] == "assistant"
        assert "6" in messages[3]["content"]

        # Base messages at end
        assert messages[4]["role"] == "system"
        assert messages[5]["role"] == "user"

    def test_includes_reasoning_in_example(self):
        detector = MarkerTerminalDetector()
        base = StepByStepPrompt(terminal_detector=detector)
        examples = StaticExampleSource([
            Example(
                problem="What is 1+1?",
                solution="ANSWER: 2",
                reasoning="1 + 1 = 2",
            ),
        ])

        prompt = FewShotPrompt(base=base, examples=examples, k=1)
        messages = prompt.format("q", State("s"))

        assistant_msg = messages[1]["content"]
        assert "1 + 1 = 2" in assistant_msg
        assert "ANSWER: 2" in assistant_msg

    def test_parse_delegates_to_base(self):
        detector = MarkerTerminalDetector()
        base = StepByStepPrompt(terminal_detector=detector)
        examples = StaticExampleSource([])

        prompt = FewShotPrompt(base=base, examples=examples, k=0)

        result = prompt.parse("some response", n=1)
        assert result == ["some response"]

    def test_respects_k_limit(self):
        detector = MarkerTerminalDetector()
        base = StepByStepPrompt(terminal_detector=detector)
        examples = StaticExampleSource([
            Example(problem=f"Q{i}", solution=f"A{i}") for i in range(10)
        ])

        prompt = FewShotPrompt(base=base, examples=examples, k=2)
        messages = prompt.format("q", State("s"))

        # Only 2 examples * 2 + 2 base = 6
        assert len(messages) == 6

    def test_no_examples_returns_base_messages_only(self):
        detector = MarkerTerminalDetector()
        base = StepByStepPrompt(terminal_detector=detector)
        examples = StaticExampleSource([])

        prompt = FewShotPrompt(base=base, examples=examples, k=3)
        messages = prompt.format("q", State("s"))

        # Just 2 base messages
        assert len(messages) == 2
