#!/usr/bin/env python3
"""Compare baseline (single-pass) vs MCTS with reasoning actions."""
import json
import time
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from mcts_reasoning.mcts import MCTS
from mcts_reasoning.evaluator import ProcessEvaluator
from mcts_reasoning.terminal import MarkerTerminalDetector
from mcts_reasoning.providers import get_provider
from mcts_reasoning.reasoning_actions import ActionGenerator, logic_actions, math_actions, general_actions
from mcts_reasoning.sampling import PathSampler, ValueSampling
from mcts_reasoning.consensus import MajorityVote
from mcts_reasoning.types import Message

BASE_URL = "http://192.168.0.225:11434"
MODEL = "gemma3:12b"

PROBLEMS = [
    {"id": "knights_2p", "q": 'A says "B is a knave." B says "We are the same type." Is A a knight or a knave? (Knights always tell truth, knaves always lie.)', "gt": "knight", "actions": "logic"},
    {"id": "knights_3p", "q": 'There are three people: A, B, and C. Each is either a knight or knave. A says: "B is a knave or C is a knave (or both)." B says: "A is a knight." C says: "I am a knave." Is A a knight or a knave? (Knights always tell truth, knaves always lie.)', "gt": "knight", "actions": "logic"},
    {"id": "arithmetic", "q": "What is (17 * 23) - (14 * 19) + 55?", "gt": "180", "actions": "math"},
    {"id": "water_jugs", "q": "You have a 5-gallon jug and a 3-gallon jug. How many pour/fill/empty operations do you need to measure exactly 4 gallons?", "gt": "6", "actions": "math"},
    {"id": "coin", "q": "You have 12 coins, one counterfeit (heavier or lighter). Using a balance scale exactly 3 times, can you always find it and determine if heavier or lighter? Answer yes or no.", "gt": "yes", "actions": "logic"},
]

def normalize(answer, gt):
    if not answer: return False
    a = answer.lower().strip().rstrip(".")
    g = gt.lower().strip()
    if g in a or a in g: return True
    if g == "knight" and "knight" in a: return True
    if g == "yes" and a in ("yes", "true"): return True
    try: return abs(float(a) - float(g)) < 0.01
    except: return False

def run_baseline(provider, question):
    messages = [
        Message(role="system", content="Solve the problem step by step. Show full reasoning, then give your final answer as: ANSWER: <answer> (1-5 words)."),
        Message(role="user", content=question),
    ]
    start = time.time()
    response = provider.generate(messages, max_tokens=2000, temperature=0.7)
    elapsed = time.time() - start
    detector = MarkerTerminalDetector()
    check = detector.is_terminal(response)
    return check.answer, elapsed

def run_mcts_actions(provider, question, action_set_name, sims=5):
    detector = MarkerTerminalDetector()
    action_sets = {"logic": logic_actions, "math": math_actions, "general": general_actions}
    actions = action_sets[action_set_name](detector)
    gen = ActionGenerator(provider=provider, actions=actions, terminal_detector=detector)
    mcts = MCTS(generator=gen, evaluator=ProcessEvaluator(), max_rollout_depth=8, max_children_per_node=3)

    start = time.time()
    state = mcts.search(question, simulations=sims)
    elapsed = time.time() - start

    sampler = PathSampler(state.root, strategy=ValueSampling(), consensus=MajorityVote())
    vote = sampler.vote()
    answer = vote.answer if vote.answer else None

    return answer, elapsed, state.root.count_nodes(), state.root.max_depth(), len(state.terminal_states)

if __name__ == "__main__":
    provider = get_provider("ollama", model=MODEL, base_url=BASE_URL)
    sims = int(sys.argv[1]) if len(sys.argv) > 1 else 5

    print(f"Model: {MODEL}, MCTS sims: {sims}")
    print(f"{'Problem':<15} {'BL':>4} {'BL Ans':<20} {'BL Time':>8} {'MCTS':>5} {'MCTS Ans':<20} {'Nodes':>6} {'Depth':>6} {'Terms':>6} {'MCTS Time':>10}")
    print("-" * 120)

    bl_correct = 0
    mcts_correct = 0

    for p in PROBLEMS:
        # Baseline
        bl_ans, bl_time = run_baseline(provider, p["q"])
        bl_ok = normalize(bl_ans, p["gt"])
        bl_correct += bl_ok

        # MCTS with actions
        mcts_ans, mcts_time, nodes, depth, terms = run_mcts_actions(provider, p["q"], p["actions"], sims)
        mcts_ok = normalize(mcts_ans, p["gt"])
        mcts_correct += mcts_ok

        bl_mark = "+" if bl_ok else "-"
        mcts_mark = "+" if mcts_ok else "-"
        bl_display = str(bl_ans or "None")[:18]
        mcts_display = str(mcts_ans or "None")[:18]

        print(f"{p['id']:<15} {bl_mark:>4} {bl_display:<20} {bl_time:>7.1f}s {mcts_mark:>5} {mcts_display:<20} {nodes:>6} {depth:>6} {terms:>6} {mcts_time:>9.1f}s")

    print("-" * 120)
    n = len(PROBLEMS)
    print(f"{'TOTAL':<15} {'':>4} {f'{bl_correct}/{n} ({bl_correct/n:.0%})':<20} {'':>8} {'':>5} {f'{mcts_correct}/{n} ({mcts_correct/n:.0%})':<20}")
