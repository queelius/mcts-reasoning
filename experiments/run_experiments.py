#!/usr/bin/env python3
"""
MCTS Prompt Strategy Experiments

Systematically varies prompting strategies, parameters, and models
to find what makes MCTS-guided reasoning work well.

Results are appended to experiments/results.jsonl
"""
import json
import time
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcts_reasoning.types import State, Message
from mcts_reasoning.mcts import MCTS
from mcts_reasoning.generator import LLMGenerator
from mcts_reasoning.evaluator import ProcessEvaluator, GroundTruthEvaluator
from mcts_reasoning.terminal import MarkerTerminalDetector
from mcts_reasoning.prompt import StepByStepPrompt, StrictAnswerPrompt, PromptStrategy
from mcts_reasoning.consensus import MajorityVote, NormalizedVote
from mcts_reasoning.sampling import ValueSampling, PathSampler
from mcts_reasoning.providers import get_provider

BASE_URL = "http://192.168.0.225:11434"
RESULTS_FILE = Path(__file__).parent / "results.jsonl"


# ─── Test Problems ───────────────────────────────────────────────
PROBLEMS = [
    {
        "id": "knights_2person",
        "question": 'A says "B is a knave." B says "We are the same type." Is A a knight or a knave? (Knights always tell truth, knaves always lie.)',
        "ground_truth": "knight",
        "domain": "logic",
        "difficulty": "easy",
    },
    {
        "id": "knights_3person",
        "question": 'There are three people: A, B, and C. Each is either a knight or a knave. A says: "B is a knave or C is a knave (or both)." B says: "A is a knight." C says: "I am a knave." Determine: is A a knight or a knave? (Knights always tell truth, knaves always lie.)',
        "ground_truth": "knight",
        "domain": "logic",
        "difficulty": "medium",
    },
    {
        "id": "water_jugs",
        "question": "You have a 5-gallon jug and a 3-gallon jug, both empty. How can you measure exactly 4 gallons? List the steps. What is the minimum number of pour/fill/empty operations needed?",
        "ground_truth": "6",
        "domain": "planning",
        "difficulty": "medium",
    },
    {
        "id": "arithmetic_chain",
        "question": "What is (17 * 23) - (14 * 19) + 55?",
        "ground_truth": "180",
        "domain": "math",
        "difficulty": "easy",
    },
    {
        "id": "coin_puzzle",
        "question": "You have 12 coins, one of which is counterfeit and either heavier or lighter than the rest. Using a balance scale exactly 3 times, can you always identify the counterfeit coin AND determine whether it is heavier or lighter? Answer yes or no.",
        "ground_truth": "yes",
        "domain": "logic",
        "difficulty": "hard",
    },
]


# ─── Custom Prompt Strategies ────────────────────────────────────

class IncrementalReasoningPrompt(PromptStrategy):
    """Forces very small reasoning steps with explicit state tracking."""

    def __init__(self, terminal_detector):
        self.terminal_detector = terminal_detector

    def format(self, question, state, n=1):
        terminal_instruction = self.terminal_detector.format_instruction()
        system = (
            "You reason by taking tiny, careful steps. Each response contains:\n"
            "1. EXACTLY ONE logical deduction or calculation\n"
            "2. A brief statement of what you now know\n\n"
            "CONSTRAINTS:\n"
            "- Maximum 2 sentences per response\n"
            "- Each step must make progress toward the answer\n"
            "- Do not repeat reasoning already done\n"
            "- Do not try to solve the whole problem at once\n"
            f"- When you can definitively answer the original question: {terminal_instruction}\n"
            "- ANSWER: must respond to the ORIGINAL question, not a sub-question\n"
        )
        if n == 1:
            user = (
                f"Original question: {question}\n\n"
                f"Work so far:\n{state}\n\n"
                f"Next single deduction:"
            )
        else:
            user = (
                f"Original question: {question}\n\n"
                f"Work so far:\n{state}\n\n"
                f"Provide {n} different possible next deductions.\n"
                f"Format: --- CONTINUATION 1 ---\n[deduction]\n--- CONTINUATION 2 ---\n[deduction]"
            )
        return [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

    def parse(self, response, n=1):
        if n == 1:
            return [response]
        import re
        parts = re.split(r"---\s*CONTINUATION\s*\d+\s*---", response)
        continuations = [p.strip() for p in parts if p.strip()]
        return continuations if continuations else [response]


class AssumptionTestingPrompt(PromptStrategy):
    """Explicitly asks the LLM to state and test assumptions."""

    def __init__(self, terminal_detector):
        self.terminal_detector = terminal_detector

    def format(self, question, state, n=1):
        terminal_instruction = self.terminal_detector.format_instruction()
        system = (
            "You solve problems by explicitly stating and testing assumptions.\n\n"
            "Each response must follow this pattern:\n"
            "- ASSUME: [state one specific assumption]\n"
            "- THEN: [derive one consequence from that assumption]\n"
            "- CHECK: [verify if the consequence is consistent or contradictory]\n\n"
            "If CHECK reveals a contradiction, say CONTRADICTION and try a different assumption next time.\n"
            "If CHECK is consistent AND you have enough to answer the original question:\n"
            f"{terminal_instruction}\n"
            "ANSWER: must respond to the ORIGINAL question directly (1-5 words).\n"
        )
        if n == 1:
            user = (
                f"Original question: {question}\n\n"
                f"Work so far:\n{state}\n\n"
                f"Next assumption to test:"
            )
        else:
            user = (
                f"Original question: {question}\n\n"
                f"Work so far:\n{state}\n\n"
                f"Provide {n} different assumptions to test.\n"
                f"Format: --- CONTINUATION 1 ---\n[assumption test]\n--- CONTINUATION 2 ---\n[assumption test]"
            )
        return [
            Message(role="system", content=system),
            Message(role="user", content=user),
        ]

    def parse(self, response, n=1):
        if n == 1:
            return [response]
        import re
        parts = re.split(r"---\s*CONTINUATION\s*\d+\s*---", response)
        continuations = [p.strip() for p in parts if p.strip()]
        return continuations if continuations else [response]


# ─── Experiment Runner ───────────────────────────────────────────

@dataclass
class ExperimentConfig:
    name: str
    model: str
    prompt_strategy: str  # name for logging
    max_tokens: int = 150
    simulations: int = 10
    exploration_constant: float = 1.414
    max_children: int = 3
    max_rollout_depth: int = 10
    temperature: float = 0.7


@dataclass
class ExperimentResult:
    config: dict
    problem_id: str
    question: str
    ground_truth: str
    # MCTS results
    mcts_answer: str | None = None
    mcts_correct: bool = False
    mcts_confidence: float = 0.0
    mcts_terminal_count: int = 0
    mcts_node_count: int = 0
    mcts_max_depth: int = 0
    mcts_time_seconds: float = 0.0
    mcts_all_answers: list = field(default_factory=list)
    # Baseline results
    baseline_answer: str | None = None
    baseline_correct: bool = False
    baseline_time_seconds: float = 0.0
    # Metadata
    timestamp: str = ""


def normalize_answer(answer: str | None, ground_truth: str) -> bool:
    """Check if answer matches ground truth (case-insensitive, flexible)."""
    if not answer:
        return False
    a = answer.lower().strip().rstrip(".")
    g = ground_truth.lower().strip()
    # Direct match
    if g in a or a in g:
        return True
    # Common variations
    if g == "knight" and ("knight" in a or a == "a"):
        return True
    if g == "yes" and a in ("yes", "true", "correct"):
        return True
    if g == "no" and a in ("no", "false", "incorrect"):
        return True
    # Numeric
    try:
        return abs(float(a) - float(g)) < 0.01
    except (ValueError, TypeError):
        pass
    return False


def make_prompt_strategy(name, detector):
    strategies = {
        "step-by-step": lambda: StepByStepPrompt(terminal_detector=detector),
        "strict": lambda: StrictAnswerPrompt(terminal_detector=detector),
        "incremental": lambda: IncrementalReasoningPrompt(terminal_detector=detector),
        "assumption": lambda: AssumptionTestingPrompt(terminal_detector=detector),
    }
    return strategies[name]()


def run_baseline(provider, prompt_strategy, question, max_tokens):
    """Single-pass LLM call (no MCTS)."""
    messages = prompt_strategy.format(question, State(""), n=1)
    # Give baseline more tokens since it needs to solve in one shot
    start = time.time()
    response = provider.generate(messages, max_tokens=max(max_tokens * 5, 500), temperature=0.7)
    elapsed = time.time() - start

    detector = MarkerTerminalDetector()
    check = detector.is_terminal(response)
    return check.answer, elapsed


def run_mcts(provider, prompt_strategy, question, config):
    """Full MCTS search."""
    detector = MarkerTerminalDetector()
    gen = LLMGenerator(
        provider=provider,
        prompt_strategy=prompt_strategy,
        terminal_detector=detector,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
    )
    evaluator = ProcessEvaluator()
    mcts = MCTS(
        generator=gen,
        evaluator=evaluator,
        exploration_constant=config.exploration_constant,
        max_children_per_node=config.max_children,
        max_rollout_depth=config.max_rollout_depth,
    )

    start = time.time()
    state = mcts.search(question, simulations=config.simulations)
    elapsed = time.time() - start

    # Extract results
    sampler = PathSampler(state.root, strategy=ValueSampling(), consensus=MajorityVote())
    vote = sampler.vote()
    all_answers = [t["answer"] for t in state.terminal_states if t.get("answer")]

    return {
        "answer": vote.answer if vote.answer else (all_answers[0] if all_answers else None),
        "confidence": vote.confidence,
        "terminal_count": len(state.terminal_states),
        "node_count": state.root.count_nodes(),
        "max_depth": state.root.max_depth(),
        "time_seconds": elapsed,
        "all_answers": all_answers,
    }


def run_experiment(config: ExperimentConfig, problems: list[dict]) -> list[ExperimentResult]:
    """Run one experiment config across all problems."""
    provider = get_provider("ollama", model=config.model, base_url=BASE_URL)
    detector = MarkerTerminalDetector()
    prompt = make_prompt_strategy(config.prompt_strategy, detector)

    results = []
    for prob in problems:
        print(f"  {prob['id']}...", end=" ", flush=True)

        # Baseline
        try:
            bl_answer, bl_time = run_baseline(provider, prompt, prob["question"], config.max_tokens)
            bl_correct = normalize_answer(bl_answer, prob["ground_truth"])
        except Exception as e:
            bl_answer, bl_time, bl_correct = str(e)[:50], 0, False
            print(f"[baseline err: {e}]", end=" ")

        # MCTS
        try:
            mcts_result = run_mcts(provider, prompt, prob["question"], config)
            mcts_correct = normalize_answer(mcts_result["answer"], prob["ground_truth"])
        except Exception as e:
            mcts_result = {"answer": str(e)[:50], "confidence": 0, "terminal_count": 0,
                          "node_count": 0, "max_depth": 0, "time_seconds": 0, "all_answers": []}
            mcts_correct = False
            print(f"[mcts err: {e}]", end=" ")

        result = ExperimentResult(
            config=asdict(config),
            problem_id=prob["id"],
            question=prob["question"][:80],
            ground_truth=prob["ground_truth"],
            mcts_answer=mcts_result["answer"],
            mcts_correct=mcts_correct,
            mcts_confidence=mcts_result["confidence"],
            mcts_terminal_count=mcts_result["terminal_count"],
            mcts_node_count=mcts_result["node_count"],
            mcts_max_depth=mcts_result["max_depth"],
            mcts_time_seconds=mcts_result["time_seconds"],
            mcts_all_answers=mcts_result["all_answers"],
            baseline_answer=bl_answer,
            baseline_correct=bl_correct,
            baseline_time_seconds=bl_time,
            timestamp=datetime.now().isoformat(),
        )
        results.append(result)

        marker = lambda c: "+" if c else "-"
        print(f"baseline={marker(bl_correct)}({bl_answer}) mcts={marker(mcts_correct)}({mcts_result['answer']}) "
              f"nodes={mcts_result['node_count']} depth={mcts_result['max_depth']} "
              f"terms={mcts_result['terminal_count']} t={mcts_result['time_seconds']:.1f}s")

    return results


def save_results(results: list[ExperimentResult]):
    """Append results to JSONL file."""
    RESULTS_FILE.parent.mkdir(exist_ok=True)
    with open(RESULTS_FILE, "a") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")


def print_summary(all_results: list[ExperimentResult]):
    """Print experiment summary table."""
    from collections import defaultdict
    by_config = defaultdict(list)
    for r in all_results:
        by_config[r.config["name"]].append(r)

    print("\n" + "=" * 90)
    print(f"{'Experiment':<30} {'BL Acc':>8} {'MCTS Acc':>8} {'Lift':>8} {'Avg Nodes':>10} {'Avg Time':>10}")
    print("-" * 90)

    for name, results in by_config.items():
        bl_acc = sum(1 for r in results if r.baseline_correct) / len(results)
        mcts_acc = sum(1 for r in results if r.mcts_correct) / len(results)
        lift = mcts_acc - bl_acc
        avg_nodes = sum(r.mcts_node_count for r in results) / len(results)
        avg_time = sum(r.mcts_time_seconds for r in results) / len(results)
        print(f"{name:<30} {bl_acc:>7.0%} {mcts_acc:>8.0%} {lift:>+7.0%} {avg_nodes:>10.1f} {avg_time:>9.1f}s")

    print("=" * 90)


# ─── Experiment Definitions ──────────────────────────────────────

def get_experiments():
    """Define all experiment configurations."""
    return [
        # Experiment 1: Prompt strategy comparison (same model, same params)
        ExperimentConfig(name="strict_default", model="gemma3:12b",
                        prompt_strategy="strict", simulations=8, max_rollout_depth=10),
        ExperimentConfig(name="incremental", model="gemma3:12b",
                        prompt_strategy="incremental", simulations=8, max_rollout_depth=10),
        ExperimentConfig(name="assumption_testing", model="gemma3:12b",
                        prompt_strategy="assumption", simulations=8, max_rollout_depth=10),

        # Experiment 2: Rollout depth (does deeper search help?)
        ExperimentConfig(name="depth_5", model="gemma3:12b",
                        prompt_strategy="strict", simulations=8, max_rollout_depth=5),
        ExperimentConfig(name="depth_15", model="gemma3:12b",
                        prompt_strategy="strict", simulations=8, max_rollout_depth=15),

        # Experiment 3: Simulation count (does more search help?)
        ExperimentConfig(name="sims_4", model="gemma3:12b",
                        prompt_strategy="strict", simulations=4, max_rollout_depth=10),
        ExperimentConfig(name="sims_16", model="gemma3:12b",
                        prompt_strategy="strict", simulations=16, max_rollout_depth=10),

        # Experiment 4: Exploration constant
        ExperimentConfig(name="explore_low", model="gemma3:12b",
                        prompt_strategy="strict", simulations=8, exploration_constant=0.5),
        ExperimentConfig(name="explore_high", model="gemma3:12b",
                        prompt_strategy="strict", simulations=8, exploration_constant=2.5),

        # Experiment 5: Max tokens (step granularity)
        ExperimentConfig(name="tokens_80", model="gemma3:12b",
                        prompt_strategy="strict", simulations=8, max_tokens=80),
        ExperimentConfig(name="tokens_250", model="gemma3:12b",
                        prompt_strategy="strict", simulations=8, max_tokens=250),
    ]


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", nargs="*", help="Run specific experiments by name (default: all)")
    parser.add_argument("--problems", nargs="*", help="Run specific problems by id (default: all)")
    parser.add_argument("--list", action="store_true", help="List available experiments")
    args = parser.parse_args()

    experiments = get_experiments()

    if args.list:
        for e in experiments:
            print(f"  {e.name:<25} model={e.model} prompt={e.prompt_strategy} "
                  f"sims={e.simulations} depth={e.max_rollout_depth} tokens={e.max_tokens}")
        sys.exit(0)

    if args.experiments:
        experiments = [e for e in experiments if e.name in args.experiments]

    problems = PROBLEMS
    if args.problems:
        problems = [p for p in PROBLEMS if p["id"] in args.problems]

    print(f"Running {len(experiments)} experiments x {len(problems)} problems")
    print(f"Results will be saved to {RESULTS_FILE}\n")

    all_results = []
    for exp in experiments:
        print(f"\n--- {exp.name} (model={exp.model}, prompt={exp.prompt_strategy}, "
              f"sims={exp.simulations}, depth={exp.max_rollout_depth}, tokens={exp.max_tokens}) ---")
        results = run_experiment(exp, problems)
        save_results(results)
        all_results.extend(results)

    print_summary(all_results)
