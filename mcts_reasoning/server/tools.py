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
from ..prompt import IncrementalReasoningPrompt
from ..terminal import MarkerTerminalDetector


def get_contracts_impl(contract_name: str | None = None) -> dict:
    """Return the ABC interface contracts so clients can implement them.

    Like querying a schema: returns method signatures, docstrings, and
    type annotations for each ABC. A client (like Claude) can read these
    and write a conforming implementation on the fly.

    Args:
        contract_name: Specific contract to return (e.g., "PromptStrategy").
            If None, returns all contracts.
    """
    import inspect
    from ..providers.base import LLMProvider
    from ..prompt import PromptStrategy, ExampleSource
    from ..generator import Generator
    from ..evaluator import Evaluator
    from ..terminal import MarkerTerminalDetector  # protocol, use impl for inspection
    from ..sampling import SamplingStrategy
    from ..consensus import ConsensusStrategy
    from ..bench.benchmark import Benchmark
    from ..bench.solver import Solver
    from ..bench.optimizer import PromptOptimizer

    abcs = {
        "LLMProvider": LLMProvider,
        "PromptStrategy": PromptStrategy,
        "Generator": Generator,
        "Evaluator": Evaluator,
        "SamplingStrategy": SamplingStrategy,
        "ConsensusStrategy": ConsensusStrategy,
        "Benchmark": Benchmark,
        "Solver": Solver,
        "PromptOptimizer": PromptOptimizer,
    }

    def extract_contract(cls):
        methods = {}
        for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
            if name.startswith("_") and name != "__init__":
                continue
            try:
                sig = inspect.signature(method)
                doc = inspect.getdoc(method) or ""
                methods[name] = {
                    "signature": str(sig),
                    "docstring": doc,
                }
            except (ValueError, TypeError):
                pass
        return {
            "class": cls.__name__,
            "module": cls.__module__,
            "docstring": inspect.getdoc(cls) or "",
            "methods": methods,
        }

    if contract_name:
        if contract_name not in abcs:
            return {"error": f"Unknown contract: {contract_name}. Available: {list(abcs.keys())}"}
        return extract_contract(abcs[contract_name])

    return {name: extract_contract(cls) for name, cls in abcs.items()}


def list_components_impl() -> dict:
    """List all available ABC implementations for each decision point.

    Returns a dict mapping each ABC name to its available implementations
    with descriptions, so MCP clients can discover what options exist.
    """
    return {
        "providers": {
            "openai": "OpenAI API (GPT-4, etc.). Requires OPENAI_API_KEY.",
            "anthropic": "Anthropic API (Claude). Requires ANTHROPIC_API_KEY.",
            "ollama": "Local Ollama instance. Requires Ollama running.",
        },
        "prompt_strategies": {
            "step-by-step": "Default. Asks the LLM to continue reasoning step by step.",
            "few-shot": "Wraps any base strategy with few-shot examples from an ExampleSource.",
        },
        "evaluators": {
            "ground-truth": "Exact string comparison with normalization.",
            "numeric": "Math with tolerance (fractions, scientific notation, partial credit).",
            "llm": "LLM-as-judge scoring (0-1).",
            "process": "Reasoning quality heuristics (steps, logic, verification).",
            "composite": "Weighted combination of other evaluators.",
        },
        "terminal_detectors": {
            "marker": "Looks for 'ANSWER:' marker (default).",
            "boxed": "Looks for \\boxed{} (math benchmarks).",
            "multi-marker": "Multiple completion markers.",
        },
        "sampling_strategies": {
            "value": "Highest average value paths.",
            "visits": "Most-explored paths (highest visit counts).",
            "diverse": "One path per unique answer, then fill by value.",
            "topk": "Top-k terminal nodes by score.",
        },
        "consensus_strategies": {
            "majority": "Simple majority vote (one vote per path).",
            "weighted": "Value-weighted vote (path value as weight).",
        },
        "benchmarks": {
            "knights": "Knights-and-knaves logic puzzles (15 problems, 3 difficulty levels).",
            "arithmetic": "Multi-step arithmetic chains (20 problems, 3 difficulty levels).",
        },
    }


def _resolve_provider(
    provider_name: str | None,
    model: str | None = None,
    base_url: str | None = None,
):
    """
    Resolve an LLM provider from a name or auto-detect.

    Args:
        provider_name: One of "openai", "anthropic", "ollama", "mock", "auto",
            or None (treated as "auto").
        model: Optional model name to pass to the provider.
        base_url: Optional base URL for the provider (e.g., remote Ollama).

    Returns:
        An LLMProvider instance.
    """
    from ..providers import detect_provider, get_provider

    kwargs: dict = {}
    if model:
        kwargs["model"] = model
    if base_url:
        kwargs["base_url"] = base_url

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
    base_url: str | None = None,
) -> dict:
    """
    Run MCTS search on a question.

    Returns a dict with answer, confidence, simulations, terminal_states,
    and tree_nodes -- or {"error": "..."} on failure.
    """
    try:
        provider = _resolve_provider(provider_name, model, base_url)
        detector = MarkerTerminalDetector()
        prompt = IncrementalReasoningPrompt(terminal_detector=detector)
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
    base_url: str | None = None,
) -> dict:
    """
    Run MCTS and return the full reasoning tree for inspection.

    Returns a dict with question, simulations, tree (serialized), and
    terminal_states -- or {"error": "..."} on failure.
    """
    try:
        provider = _resolve_provider(provider_name, model, base_url)
        detector = MarkerTerminalDetector()
        prompt = IncrementalReasoningPrompt(terminal_detector=detector)
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
    base_url: str | None = None,
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
        provider = _resolve_provider(provider_name, model, base_url)
        detector = MarkerTerminalDetector()
        prompt = IncrementalReasoningPrompt(terminal_detector=detector)
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
