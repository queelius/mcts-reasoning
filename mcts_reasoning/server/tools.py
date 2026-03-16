"""
MCP Server tool implementations.

Pure functions that implement the business logic for each MCP tool.
These are testable without any MCP dependency -- the MCP layer in __init__
simply delegates to these functions.
"""

from __future__ import annotations

import json
from typing import Optional

from ..evaluator import ProcessEvaluator
from ..generator import LLMGenerator
from ..mcts import MCTS
from ..prompt import StepByStepPrompt
from ..terminal import MarkerTerminalDetector


def _resolve_provider(provider_name: str | None, model: str | None = None):
    """
    Resolve an LLM provider from a name or auto-detect.

    Args:
        provider_name: One of "openai", "anthropic", "ollama", "mock", "auto",
            or None (treated as "auto").
        model: Optional model name to pass to the provider.

    Returns:
        An LLMProvider instance.
    """
    from ..providers import detect_provider, get_provider

    kwargs: dict = {}
    if model:
        kwargs["model"] = model

    if provider_name == "auto" or provider_name is None:
        return detect_provider(**kwargs)
    if provider_name == "mock":
        from ..testing import MockLLMProvider

        return MockLLMProvider()
    return get_provider(provider_name, **kwargs)


def mcts_search_impl(
    question: str,
    provider_name: str = "auto",
    model: str | None = None,
    simulations: int = 10,
    exploration_constant: float = 1.414,
) -> dict:
    """
    Run MCTS search on a question.

    Returns a dict with answer, confidence, simulations, terminal_states,
    and tree_nodes -- or {"error": "..."} on failure.
    """
    try:
        provider = _resolve_provider(provider_name, model)
        detector = MarkerTerminalDetector()
        prompt = StepByStepPrompt(terminal_detector=detector)
        gen = LLMGenerator(
            provider=provider,
            prompt_strategy=prompt,
            terminal_detector=detector,
        )
        evaluator = ProcessEvaluator()
        mcts = MCTS(
            generator=gen,
            evaluator=evaluator,
            exploration_constant=exploration_constant,
        )
        state = mcts.search(question, simulations=simulations)

        best = (
            max(state.terminal_states, key=lambda t: t["score"])
            if state.terminal_states
            else None
        )
        return {
            "answer": best["answer"] if best else None,
            "confidence": best["score"] if best else 0.0,
            "simulations": state.simulations_run,
            "terminal_states": len(state.terminal_states),
            "tree_nodes": state.root.count_nodes(),
        }
    except Exception as e:
        return {"error": str(e)}


def mcts_explore_impl(
    question: str,
    provider_name: str = "auto",
    model: str | None = None,
    simulations: int = 10,
) -> dict:
    """
    Run MCTS and return the full reasoning tree for inspection.

    Returns a dict with question, simulations, tree (serialized), and
    terminal_states -- or {"error": "..."} on failure.
    """
    try:
        provider = _resolve_provider(provider_name, model)
        detector = MarkerTerminalDetector()
        prompt = StepByStepPrompt(terminal_detector=detector)
        gen = LLMGenerator(
            provider=provider,
            prompt_strategy=prompt,
            terminal_detector=detector,
        )
        evaluator = ProcessEvaluator()
        mcts = MCTS(generator=gen, evaluator=evaluator)
        state = mcts.search(question, simulations=simulations)
        return {
            "question": state.question,
            "simulations": state.simulations_run,
            "tree": state.root.to_dict(),
            "terminal_states": state.terminal_states,
        }
    except Exception as e:
        return {"error": str(e)}


def mcts_bench_impl(
    benchmark: str = "knights",
    provider_name: str = "auto",
    model: str | None = None,
    simulations: Optional[list[int]] = None,
) -> dict:
    """
    Run a benchmark: baseline vs MCTS at one or more simulation counts.

    Returns the report as a parsed JSON dict -- or {"error": "..."} on failure.
    """
    try:
        if simulations is None:
            simulations = [10]

        from ..bench.benchmarks import get_benchmark
        from ..bench.solver import BaselineSolver, MCTSSolver

        bench = get_benchmark(benchmark)
        provider = _resolve_provider(provider_name, model)
        detector = MarkerTerminalDetector()
        prompt = StepByStepPrompt(terminal_detector=detector)
        evaluator = ProcessEvaluator()

        solvers: list = [BaselineSolver(provider, prompt)]
        for s in simulations:
            solvers.append(
                MCTSSolver(
                    provider,
                    prompt,
                    evaluator,
                    detector,
                    simulations=s,
                )
            )

        from ..bench.runner import BenchRunner

        runner = BenchRunner()
        report = runner.run(bench, solvers)
        # to_json() returns a JSON string; parse it to a dict for the caller.
        return json.loads(report.to_json())
    except Exception as e:
        return {"error": str(e)}
