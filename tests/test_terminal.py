"""Tests for the terminal detection module."""

import pytest
from mcts_reasoning.terminal import (
    TerminalDetector,
    TerminalCheck,
    MarkerTerminalDetector,
    BoxedTerminalDetector,
    MultiMarkerTerminalDetector,
)


class TestTerminalCheck:
    """Tests for TerminalCheck dataclass."""

    def test_create_non_terminal(self):
        check = TerminalCheck(is_terminal=False)
        assert check.is_terminal is False
        assert check.answer is None
        assert check.confidence == 1.0

    def test_create_terminal(self):
        check = TerminalCheck(is_terminal=True, answer="42", confidence=0.9)
        assert check.is_terminal is True
        assert check.answer == "42"
        assert check.confidence == 0.9


class TestMarkerTerminalDetector:
    """Tests for MarkerTerminalDetector."""

    def test_default_marker(self):
        detector = MarkerTerminalDetector()
        assert detector.marker == "ANSWER:"

    def test_custom_marker(self):
        detector = MarkerTerminalDetector(marker="RESULT:")
        assert detector.marker == "RESULT:"

    def test_check_non_terminal(self):
        detector = MarkerTerminalDetector()
        result = detector.check("Still thinking about the problem...")

        assert result.is_terminal is False
        assert result.answer is None

    def test_check_terminal_basic(self):
        detector = MarkerTerminalDetector()
        result = detector.check("After calculation, ANSWER: 42")

        assert result.is_terminal is True
        assert result.answer == "42"
        assert result.confidence == 1.0

    def test_check_terminal_with_whitespace(self):
        detector = MarkerTerminalDetector()
        result = detector.check("ANSWER:   the answer is 7  ")

        assert result.is_terminal is True
        assert result.answer == "the answer is 7"

    def test_check_terminal_multiline(self):
        detector = MarkerTerminalDetector()
        state = """Step 1: Analysis
Step 2: Calculation
ANSWER: 42

Some extra text."""

        result = detector.check(state)
        assert result.is_terminal is True
        assert result.answer == "42"

    def test_check_terminal_at_end(self):
        detector = MarkerTerminalDetector()
        result = detector.check("Therefore ANSWER: final answer here")

        assert result.is_terminal is True
        assert result.answer == "final answer here"

    def test_format_instruction(self):
        detector = MarkerTerminalDetector()
        instruction = detector.format_instruction()

        assert "ANSWER:" in instruction
        assert "<your answer>" in instruction

    def test_custom_marker_format_instruction(self):
        detector = MarkerTerminalDetector(marker="SOLUTION:")
        instruction = detector.format_instruction()

        assert "SOLUTION:" in instruction


class TestBoxedTerminalDetector:
    """Tests for BoxedTerminalDetector."""

    def test_check_non_terminal(self):
        detector = BoxedTerminalDetector()
        result = detector.check("No boxed answer here")

        assert result.is_terminal is False

    def test_check_simple_boxed(self):
        detector = BoxedTerminalDetector()
        result = detector.check("The answer is \\boxed{42}")

        assert result.is_terminal is True
        assert result.answer == "42"

    def test_check_boxed_expression(self):
        detector = BoxedTerminalDetector()
        result = detector.check("Therefore \\boxed{x^2 + 2x + 1}")

        assert result.is_terminal is True
        assert result.answer == "x^2 + 2x + 1"

    def test_check_nested_braces(self):
        detector = BoxedTerminalDetector()
        result = detector.check("\\boxed{\\frac{1}{2}}")

        assert result.is_terminal is True
        assert result.answer == "\\frac{1}{2}"

    def test_format_instruction(self):
        detector = BoxedTerminalDetector()
        instruction = detector.format_instruction()

        assert "\\boxed" in instruction


class TestMultiMarkerTerminalDetector:
    """Tests for MultiMarkerTerminalDetector."""

    def test_default_markers(self):
        detector = MultiMarkerTerminalDetector()
        assert "ANSWER:" in detector.markers
        assert "FINAL ANSWER:" in detector.markers
        assert "\\boxed{" in detector.markers

    def test_custom_markers(self):
        detector = MultiMarkerTerminalDetector(markers=["RESULT:", "SOLUTION:"])
        assert detector.markers == ["RESULT:", "SOLUTION:"]

    def test_check_first_marker(self):
        detector = MultiMarkerTerminalDetector()
        result = detector.check("ANSWER: 42")

        assert result.is_terminal is True
        assert result.answer == "42"

    def test_check_second_marker(self):
        detector = MultiMarkerTerminalDetector()
        result = detector.check("FINAL ANSWER: 42")

        assert result.is_terminal is True
        assert result.answer == "42"

    def test_check_boxed_marker(self):
        detector = MultiMarkerTerminalDetector()
        result = detector.check("\\boxed{42}")

        assert result.is_terminal is True
        assert result.answer == "42"

    def test_check_non_terminal(self):
        detector = MultiMarkerTerminalDetector()
        result = detector.check("Still working on it...")

        assert result.is_terminal is False

    def test_format_instruction(self):
        detector = MultiMarkerTerminalDetector()
        instruction = detector.format_instruction()

        assert "ANSWER:" in instruction or "one of:" in instruction


class TestTerminalDetectorProtocol:
    """Tests that custom detectors work with the protocol."""

    def test_custom_detector_satisfies_protocol(self):
        class ConfidenceDetector:
            def check(self, state: str) -> TerminalCheck:
                if "confident" in state.lower():
                    return TerminalCheck(
                        is_terminal=True,
                        answer="found",
                        confidence=0.95,
                    )
                return TerminalCheck(is_terminal=False)

            def format_instruction(self) -> str:
                return "State 'confident' when you have an answer."

        detector = ConfidenceDetector()
        assert isinstance(detector, TerminalDetector)

        result = detector.check("I am confident the answer is X")
        assert result.is_terminal is True
        assert result.confidence == 0.95


class TestGeneratorIntegration:
    """Tests that Generator uses TerminalDetector correctly."""

    def test_generator_with_boxed_detector(self):
        from mcts_reasoning.generator import MockGenerator

        detector = BoxedTerminalDetector()
        generator = MockGenerator(
            responses=["Step 1: Working...", "\\boxed{42}"],
            terminal_detector=detector,
        )

        # First response - not terminal
        conts = generator.generate("q", "s", n=1)
        assert conts[0].is_terminal is False

        # Second response - terminal via boxed
        conts = generator.generate("q", "s", n=1)
        assert conts[0].is_terminal is True
        assert conts[0].answer == "42"

    def test_generator_extract_answer_uses_detector(self):
        from mcts_reasoning.generator import MockGenerator

        detector = BoxedTerminalDetector()
        generator = MockGenerator(terminal_detector=detector)

        answer = generator.extract_answer("The result is \\boxed{7}")
        assert answer == "7"

    def test_generator_is_terminal_uses_detector(self):
        from mcts_reasoning.generator import MockGenerator

        detector = MarkerTerminalDetector(marker="DONE:")
        generator = MockGenerator(terminal_detector=detector)

        assert generator.is_terminal("DONE: finished") is True
        assert generator.is_terminal("ANSWER: 42") is False  # Wrong marker
