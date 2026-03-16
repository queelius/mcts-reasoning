"""Tests for the generator module."""

import pytest
from mcts_reasoning.generator import Generator, LLMGenerator, MockGenerator, ANSWER_MARKER
from mcts_reasoning.prompt import StepByStepPrompt
from mcts_reasoning.terminal import MarkerTerminalDetector, BoxedTerminalDetector
from mcts_reasoning.testing import MockLLMProvider
from mcts_reasoning.types import Continuation


class TestGeneratorABC:
    """Tests for the Generator abstract base class."""

    def test_extract_answer_uses_marker(self):
        gen = MockGenerator()
        assert gen.extract_answer("ANSWER: 42") == "42"
        assert gen.extract_answer("No answer here") is None

    def test_is_terminal_uses_marker(self):
        gen = MockGenerator()
        assert gen.is_terminal("ANSWER: done") is True
        assert gen.is_terminal("Still working...") is False


class TestLLMGenerator:
    """Tests for LLMGenerator."""

    def test_basic_generation(self):
        provider = MockLLMProvider(responses=["Let me think about this..."])
        detector = MarkerTerminalDetector()
        prompt = StepByStepPrompt(terminal_detector=detector)

        gen = LLMGenerator(
            provider=provider,
            prompt_strategy=prompt,
            terminal_detector=detector,
        )

        results = gen.generate("What is 2+2?", "Initial state", n=1)

        assert len(results) == 1
        assert isinstance(results[0], Continuation)
        assert "Let me think about this..." in results[0].text
        assert results[0].is_terminal is False
        assert results[0].answer is None

    def test_terminal_detection(self):
        provider = MockLLMProvider(responses=["ANSWER: 42"])
        detector = MarkerTerminalDetector()
        prompt = StepByStepPrompt(terminal_detector=detector)

        gen = LLMGenerator(
            provider=provider,
            prompt_strategy=prompt,
            terminal_detector=detector,
        )

        results = gen.generate("What is 6*7?", "Let me calculate...", n=1)

        assert len(results) == 1
        assert results[0].is_terminal is True
        assert results[0].answer == "42"

    def test_diverse_generation(self):
        response = (
            "--- CONTINUATION 1 ---\n"
            "First approach: direct addition\n\n"
            "--- CONTINUATION 2 ---\n"
            "Second approach: use counting\n\n"
            "--- CONTINUATION 3 ---\n"
            "Third approach: ANSWER: 4"
        )
        provider = MockLLMProvider(responses=[response])
        detector = MarkerTerminalDetector()
        prompt = StepByStepPrompt(terminal_detector=detector)

        gen = LLMGenerator(
            provider=provider,
            prompt_strategy=prompt,
            terminal_detector=detector,
        )

        results = gen.generate("What is 2+2?", "state", n=3)

        assert len(results) == 3
        assert results[0].is_terminal is False
        assert results[2].is_terminal is True
        assert results[2].answer == "4"

    def test_state_extension(self):
        provider = MockLLMProvider(responses=["Next step..."])
        gen = LLMGenerator(provider=provider)

        results = gen.generate("q", "Previous reasoning", n=1)

        assert "Previous reasoning" in results[0].text
        assert "Next step..." in results[0].text

    def test_default_prompt_strategy(self):
        provider = MockLLMProvider(responses=["step"])
        gen = LLMGenerator(provider=provider)

        assert isinstance(gen.prompt_strategy, StepByStepPrompt)
        assert isinstance(gen.terminal_detector, MarkerTerminalDetector)

    def test_custom_terminal_detector(self):
        provider = MockLLMProvider(responses=["\\boxed{42}"])
        detector = BoxedTerminalDetector()
        prompt = StepByStepPrompt(terminal_detector=detector)

        gen = LLMGenerator(
            provider=provider,
            prompt_strategy=prompt,
            terminal_detector=detector,
        )

        results = gen.generate("q", "s", n=1)

        assert results[0].is_terminal is True
        assert results[0].answer == "42"

    def test_extract_answer_uses_detector(self):
        provider = MockLLMProvider()
        detector = BoxedTerminalDetector()

        gen = LLMGenerator(
            provider=provider,
            terminal_detector=detector,
        )

        assert gen.extract_answer("\\boxed{7}") == "7"
        assert gen.extract_answer("ANSWER: 7") is None  # Wrong marker

    def test_is_terminal_uses_detector(self):
        provider = MockLLMProvider()
        detector = MarkerTerminalDetector(marker="DONE:")

        gen = LLMGenerator(
            provider=provider,
            terminal_detector=detector,
        )

        assert gen.is_terminal("DONE: finished") is True
        assert gen.is_terminal("ANSWER: 42") is False

    def test_provider_receives_messages(self):
        provider = MockLLMProvider(responses=["response"])
        gen = LLMGenerator(provider=provider)

        gen.generate("What is 2+2?", "state", n=1)

        assert provider.call_count == 1
        messages = provider.calls[0]
        assert isinstance(messages, list)
        assert any("What is 2+2?" in m["content"] for m in messages)

    def test_legacy_llm_kwarg(self):
        """Test backward compat: LLMGenerator(llm=...) still works."""
        provider = MockLLMProvider(responses=["step"])
        gen = LLMGenerator(llm=provider, temperature=0.5)

        assert gen.provider is provider
        assert gen.temperature == 0.5

    def test_no_provider_raises(self):
        with pytest.raises(TypeError, match="requires a provider"):
            LLMGenerator(provider=None)


class TestMockGenerator:
    """Tests for MockGenerator (from testing module, re-exported)."""

    def test_basic_generation(self):
        gen = MockGenerator()
        results = gen.generate("q", "state", n=1)

        assert len(results) == 1
        assert isinstance(results[0], Continuation)

    def test_becomes_terminal(self):
        gen = MockGenerator(terminal_at=2)
        r1 = gen.generate("q", "s", n=1)
        r2 = gen.generate("q", "s", n=1)

        assert r1[0].is_terminal is False
        assert r2[0].is_terminal is True
        assert r2[0].answer == "42"

    def test_custom_responses(self):
        gen = MockGenerator(responses=["Step A", "Step B"])
        r1 = gen.generate("q", "s", n=1)
        r2 = gen.generate("q", "s", n=1)

        assert "Step A" in r1[0].text
        assert "Step B" in r2[0].text

    def test_multiple_continuations(self):
        gen = MockGenerator(terminal_at=5)
        results = gen.generate("q", "s", n=3)

        assert len(results) == 3

    def test_answer_marker_constant(self):
        assert ANSWER_MARKER == "ANSWER:"
