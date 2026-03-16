"""
Benchmarking framework for MCTS-Reasoning.

Public API:
  Problem      – a single benchmark problem (question + ground truth)
  Benchmark    – abstract base for benchmark collections
  SolverResult – result of solving one problem
  Solver       – abstract base for problem solvers
  BaselineSolver  – single-pass LLM solver (no tree search)
  MCTSSolver      – MCTS-based solver
  BenchRunner  – runs benchmarks against solver lists
  BenchReport  – holds results; provides accuracy, lift, table/JSON/CSV output
  PromptOptimizer – abstract base for prompt optimization (v0.7)

Built-in benchmarks are in mcts_reasoning.bench.benchmarks.
"""

from .benchmark import Benchmark, Problem
from .solver import Solver, SolverResult, BaselineSolver, MCTSSolver
from .runner import BenchRunner
from .report import BenchReport
from .optimizer import PromptOptimizer

__all__ = [
    "Benchmark",
    "Problem",
    "Solver",
    "SolverResult",
    "BaselineSolver",
    "MCTSSolver",
    "BenchRunner",
    "BenchReport",
    "PromptOptimizer",
]
