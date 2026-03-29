"""
Fair comparison: MCTS vs CoT baselines with equal budget and same verifier.

Budget: N total LLM generation calls.
- CoT-SC: N solutions, self-consistency vote
- CoT-BestOfN: N solutions, pick highest verifier score
- CoT-USC: N solutions, universal self-consistency
- MCTS: M initial + (N-M) exploration rounds, UCB-guided, same verifier

The verifier is generated once per problem, shared by both MCTS and CoT-BestOfN.
Classical verification (for measurement) is separate and used only for
benchmarking, never in the reward loop.
"""

import sys
import json
import random
import time

sys.path.insert(0, "/home/spinoza/github/beta/mcts-reasoning")

from mcts_reasoning.providers import get_provider
from mcts_reasoning.types import Message
from mcts_reasoning.judge import LLMJudge
from mcts_reasoning.reward import GeneratedVerifierEvaluator
from mcts_reasoning.pipeline_v3 import PipelineV3, PipelineV3Config
from mcts_reasoning.baselines import (
    cot_generate, cot_self_consistency, cot_best_of_n, cot_universal_sc,
)
from mcts_reasoning.bench.benchmarks.combinatorial import (
    generate_assignment_problem, generate_coloring_problem, generate_seating_problem,
)
from mcts_reasoning.bench.benchmarks.combinatorial_verifier import parse_and_verify


BASE = "http://192.168.0.225:11434"
BUDGET = 10  # total generation calls per method
OUTFILE = "/home/spinoza/github/beta/mcts-reasoning/experiments/fair_comparison.jsonl"


def run_experiment(provider, problems, budget=BUDGET):
    open(OUTFILE, "w").close()

    for i, p in enumerate(problems):
        q = p.question
        print(f"\n[{i+1}/{len(problems)}] {p.metadata['type']}: {q[:50]}...", flush=True)

        # Generate verifier ONCE, share between MCTS and CoT-BestOfN
        print("  Generating verifier...", flush=True)
        verifier = GeneratedVerifierEvaluator(provider, q)
        print(f"  Verifier code:\n    {verifier.code[:100]}...", flush=True)

        # --- CoT baselines (all use full budget on independent solutions) ---

        print(f"  CoT x{budget}...", flush=True)
        solutions = cot_generate(provider, q, budget)

        # CoT-SC: self-consistency vote
        judge = LLMJudge(provider)
        sc_answer = cot_self_consistency(provider, q, solutions, judge)

        # CoT-BestOfN: same verifier as MCTS
        bon_answer, _ = cot_best_of_n(provider, q, solutions, verifier)

        # CoT-USC: universal self-consistency
        usc_answer = cot_universal_sc(provider, q, solutions)

        # --- MCTS (split budget: 6 initial + 4 explore) ---

        print(f"  MCTS (6+4)...", flush=True)
        config = PipelineV3Config(
            n_solutions=budget - 4,  # reserve 4 for exploration
            n_explore=4,
            max_tokens_cot=800,
        )
        pipe = PipelineV3(provider=provider, evaluator=verifier, config=config)
        state = pipe.run(q)
        mcts_answer, mcts_conf = pipe.best_answer(state)

        # --- Measure with classical verifier (for benchmarking only) ---

        def check(answer):
            if not answer:
                return False
            # Check if any CoT solution containing this answer is valid
            for sol in solutions:
                if answer.lower() in sol.lower():
                    valid, _ = parse_and_verify(provider, p.metadata, q, sol)
                    if valid:
                        return True
            # Also try the answer string directly
            valid, _ = parse_and_verify(provider, p.metadata, q, answer)
            return valid

        def check_mcts():
            # Check any MCTS terminal
            for node in pipe._iter_terminals(state.root):
                valid, _ = parse_and_verify(provider, p.metadata, q, str(node.state))
                if valid:
                    return True
            return False

        sc_ok = check(sc_answer)
        bon_ok = check(bon_answer)
        usc_ok = check(usc_answer)
        mcts_ok = check_mcts()

        result = {
            "type": p.metadata["type"],
            "question": q[:80],
            "sc_ok": sc_ok, "sc_answer": str(sc_answer)[:40],
            "bon_ok": bon_ok, "bon_answer": str(bon_answer)[:40],
            "usc_ok": usc_ok, "usc_answer": str(usc_answer)[:40],
            "mcts_ok": mcts_ok, "mcts_answer": str(mcts_answer)[:40],
            "mcts_nodes": state.root.count_nodes(),
            "mcts_terminals": len(state.terminal_states),
            "verifier_code": verifier.code[:200],
        }

        cm = lambda ok: "+" if ok else "-"
        print(f"  SC={cm(sc_ok)} BestOfN={cm(bon_ok)} USC={cm(usc_ok)} MCTS={cm(mcts_ok)}", flush=True)

        with open(OUTFILE, "a") as f:
            f.write(json.dumps(result) + "\n")

    # Summary
    results = [json.loads(line) for line in open(OUTFILE)]
    n = len(results)
    print(f"\n{'='*60}")
    for method in ["sc", "bon", "usc", "mcts"]:
        c = sum(1 for r in results if r[f"{method}_ok"])
        print(f"  {method.upper():<8} {c}/{n} ({c/n:.0%})")


if __name__ == "__main__":
    provider = get_provider("ollama", model="llama3.2", base_url=BASE)

    random.seed(789)
    problems = []
    for _ in range(50):
        for gen in [
            lambda: generate_assignment_problem(4, 4),
            lambda: generate_coloring_problem(5, 0.5, 3),
            lambda: generate_seating_problem(5, 3),
        ]:
            p = gen()
            if p:
                problems.append(p)
        if len(problems) >= 6:
            break

    run_experiment(provider, problems)
