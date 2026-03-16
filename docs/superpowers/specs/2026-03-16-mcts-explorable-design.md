# MCTS Explorable: "Watch an LLM Think"

**Date**: 2026-03-16
**Status**: Draft
**Deliverable**: Self-contained HTML explorable
**Location**: `../explainables/mcts-reasoning/` (relative to this repo)

## Overview

An interactive scrollytelling explorable that teaches how Monte Carlo Tree Search improves LLM reasoning. The hook: you literally watch reasoning paths branch, dead-end, and backtrack. The tree IS the insight.

## Audience

Mixed/broad. Accessible to curious developers, with optional depth for ML practitioners. Layered: the surface is visual and intuitive, the details are available on hover/click/expand.

## Visual Style

Hacker/Terminal. Dark background (`#1a1a2e`), monospace fonts, neon accents (`#e94560` primary, `#4a9eff` secondary). Tree edges glow. Nodes pulse when selected. Like peering into a machine's reasoning process.

Avoid SVG filter effects (`feGaussianBlur`, `feDropShadow`) for animations. Use CSS `box-shadow` or `opacity` transitions on SVG elements instead, as they are GPU-composited.

## Interactivity Model

**Hybrid scrollytelling**:
- Pre-computed trees drive the main narrative (no API calls needed)
- Sticky tree visualization on the right animates as the user scrolls through narrative sections on the left
- Interactive controls (UCB1 slider, sampling strategy toggles) at key moments
- Optional live sandbox at the end (Ollama only)

## Demo Problem

**Puzzle (committed)**: "A says 'B is a knave.' B says 'We are the same type.' Is A a knight or a knave?"

**Correct answer**: A is a knight (and B is a knave).

**Wrong-path reasoning**: Assume A is a knave. Then A's statement ("B is a knave") is false, so B is a knight. But B says "We are the same type." If B is a knight, B tells truth, so A and B are the same type. But we assumed A is a knave and concluded B is a knight. Contradiction.

**Correct-path reasoning**: Assume A is a knight. Then A tells truth, so B is a knave. B says "We are the same type." B is a knave, so B lies. A (knight) and B (knave) are different types, and B lying about being the same type is consistent. No contradiction.

**Branch structure for the tree** (3 main branches):

1. **Branch A (correct)**: Assume A is knight → B is knave → B lies about same type → consistent → ANSWER: A is a knight (score: 1.0)
2. **Branch B (wrong)**: Assume A is knave → B is knight → B says same type truthfully → but they're different → contradiction → ANSWER: A is a knave (score: 0.0)
3. **Branch C (partial)**: Start by analyzing B's statement first → skips explicit assumption about A → jumps to answer without verifying all constraints → ANSWER: A is a knight (score: 0.6). The score is lower because the reasoning does not explicitly state and test an assumption, making it less rigorous even though it reaches the right answer.

## Reasoning Traces (Appendix A)

The reasoning traces must be authored before implementation begins. They are the core creative content of the explorable. Each trace is a sequence of reasoning steps (strings) stored in `data/reasoning_traces.json` and consumed by the data generation script.

### Single-pass wrong attempt (Section 1)

A hand-written reasoning trace embedded directly in the HTML. The LLM assumes A is a knave in the first sentence, reasons through B's statement, and confidently concludes "A is a knave" without checking the alternative. The flaw: it never tests the assumption against all constraints.

### Tree traces (Sections 2-5)

Each branch consists of 2-4 reasoning steps. Each step should be 1-3 sentences. The full set of traces is authored in `data/reasoning_traces.json` (see Reasoning Traces Input Schema under Technical Architecture for the JSON format).

Outline per branch:
- **Branch A, Step 1**: "Let's assume A is a knight. Knights always tell the truth."
- **Branch A, Step 2**: "A says 'B is a knave.' Since A is truthful, B must be a knave."
- **Branch A, Step 3**: "B says 'We are the same type.' Since B is a knave, B lies. A is a knight, B is a knave: they are different types. B's lie is consistent. ANSWER: A is a knight."
- **Branch B, Step 1**: "Let's assume A is a knave. Knaves always lie."
- **Branch B, Step 2**: "A says 'B is a knave.' Since A lies, B must be a knight."
- **Branch B, Step 3**: "B says 'We are the same type.' Since B is a knight, B tells truth. So A and B are the same type. But A is a knave and B is a knight: they are different types. Contradiction. ANSWER: A is a knave."
- **Branch C, Step 1**: "Let me start with B's statement. B says 'We are the same type.' That's interesting."
- **Branch C, Step 2**: "If B is telling the truth, they're the same type. If B is lying, they're different types. Either way, B's statement constrains the relationship between A and B."
- **Branch C, Step 3**: "A's statement about B being a knave seems important. If A is right about B, then B is a knave, and B's statement is a lie. That seems to work. ANSWER: A is a knight." (Note: this reasoning skips the explicit assumption-and-test pattern, never formally checks the A-is-a-knave case, and hand-waves "seems to work" instead of proving consistency. Score: 0.6.)

## Page Structure

### Section 1: "One Shot, One Chance" (The Problem)

**Purpose**: Motivate why MCTS matters.

**Content**: Show the single-pass wrong attempt (see Appendix A). The reasoning *looks* convincing but makes an unchecked assumption early and arrives at the wrong answer. Highlight the flaw with a subtle annotation.

**Transition**: *"What if the LLM could explore both assumptions simultaneously, and backtrack from the one that leads to contradiction?"* The scroll begins, the tree appears.

**Interactivity**: None. Pure narrative setup.

**Tree state**: No tree visible yet.

### Section 2: "The Tree Appears" (Selection)

**Purpose**: Introduce UCB1 and the exploration/exploitation tradeoff.

**Content**: The sticky tree visualization appears on the right. Starting from the root (the question), UCB1 selects which node to explore next. Nodes display two numbers: value (quality) and visits (exploration count).

**Interactivity**:
- **UCB1 slider**: Adjust exploration constant `c` from 0 (pure greedy) to 4 (pure exploration). The slider highlights the full selection path (root to chosen leaf) with a distinct color. As the user drags, the highlighted path updates in real-time. At each internal node along the path, a tooltip shows which child was chosen and why (UCB1 score comparison). Default position: sqrt(2).
- Node hover shows the UCB1 calculation breakdown.

**Key insight**: At c=0, the algorithm always picks the highest-value node (gets stuck). At c=sqrt(2), it balances. At c=4, it wastes visits on weak paths.

**Tree state**: Tree with root + 3 first-level children (the three branch starting points). ~5 simulations worth of data. Branches A and B have 2 levels of depth, Branch C has 1.

### Section 3: "Branching Thoughts" (Expansion)

**Purpose**: Show how the LLM generates new reasoning steps.

**Content**: From the selected node, the LLM generates a continuation. The tree grows a new child. We show the actual reasoning text: "Assume A is a knight..." on one branch, "Assume A is a knave..." on another.

**Interactivity**:
- Click any node to see its full reasoning text in a panel below the tree.
- The tree animates the new node appearing with a brief glow effect.

**Key insight**: Each branch is a different assumption the LLM is trying. Structured exploration, not random.

**Tree state**: Same as Section 2, but one node is highlighted as "selected" and a new child animates into existence.

### Section 4: "Following the Thread" (Rollout)

**Purpose**: Show how reasoning continues to completion.

**Content**: From the new node, the LLM keeps generating steps until it reaches an answer ("ANSWER: Knight") or hits max depth. Nodes appear one by one in a chain, animated.

**Interactivity**: Play/pause button for the rollout animation. Speed control: 1x = 1500ms between node appearances (enough to read the reasoning text), 2x = 750ms, 4x = 375ms.

**Key insight (tree-building rollouts)**: Unlike classical MCTS (which discards rollout results), this implementation *keeps every node in the tree*. A collapsible aside panel (click to expand) shows a small ~5-node mini-tree. Left half: classical rollout where 3 nodes appear then fade to dotted outlines. Right half: same rollout where nodes remain solid. Auto-plays when expanded.

**Tree state**: A rollout chain of 2-3 nodes extending from the expanded node, appearing one at a time. The final node is terminal (has an answer).

### Section 4.5: "What's a Good Answer?" (Scoring)

**Purpose**: Introduce evaluation before backpropagation uses it.

**Content**: The terminal node has an answer, but how do we know if it is *right*? The evaluator checks whether the answer is logically consistent with the puzzle's constraints. A correct derivation with no contradictions scores 1.0. A derivation that hits a contradiction scores 0.0. A derivation that reaches the right answer through weak reasoning scores somewhere in between (e.g., 0.6).

**Interactivity**: None. Brief narrative paragraph (2-3 sentences).

**Tree state**: Same as end of Section 4, with the terminal node now showing its score.

### Section 5: "Scores Flow Upward" (Backpropagation)

**Purpose**: Show how terminal scores propagate through the tree.

**Content**: The terminal node's score propagates up: each ancestor updates its average value and increments its visit count. Animated: nodes glow brighter as values update, edges pulse with the propagating signal.

**Interactivity**: Step-through mode. Click to advance backpropagation one level at a time. Each step shows the value/visits update math.

**Key insight**: Good answers make nearby reasoning paths more attractive for future exploration. The tree *learns* where to search next.

**Tree state**: The backpropagation path (terminal to root) highlights one level at a time. Value/visits labels update visibly.

### Section 6: "Many Paths, One Answer" (Path Sampling & Self-Consistency)

**Purpose**: Show how to extract answers from the completed tree.

**Content**: After the full search completes (display the final tree with ~20 simulations), present the interactive sampling panel. Split into two phases:

**Phase 6a, "Which paths should we look at?"**: Strategy toggle buttons (value, visits, diverse, top-k). Each highlights different paths in the tree with distinct colors. Explanation of what each strategy prioritizes.

**Phase 6b, "How do we pick the final answer?"**: Answer distribution histogram below the tree. Shows how many paths found each answer (e.g., "3 paths say Knight, 1 says Knave"). Voting toggle: switch between majority vote and value-weighted vote. Shows confidence score.

**Key insight**: Self-consistency. When multiple independent reasoning paths agree, we can be more confident in the answer. Different sampling strategies reveal different aspects of the search.

**Tree state**: Full tree (~20 simulations, ~30-50 nodes). All paths visible. Highlighted paths change based on active sampling strategy.

### Section 7: "Try It Yourself" (Live Sandbox)

**Purpose**: Let curious users run MCTS on their own questions.

**Content**: A terminal-styled input box. User types a question, configures Ollama endpoint (defaults to `localhost:11434`), sets parameters (simulations, exploration constant, max depth), and clicks "Search."

**Interactivity**: The tree builds in real-time as simulations run. Same visualization as the narrative sections, but driven by live API calls. Results appear as they complete.

**Implementation**: HTTP calls to a local Ollama instance only (OpenAI/Anthropic APIs block browser CORS requests). The MCTS algorithm runs in JavaScript in the browser: a simplified port of the core loop (estimated 500-800 lines). Falls back with a message: "Connect to Ollama at localhost:11434 to try this."

**Error handling**: If a simulation fails mid-search (empty LLM response, network timeout), skip that simulation and continue with the next. Show a brief inline warning ("Simulation 4 failed, continuing..."). Cap the tree at 50 nodes max to stay within the visualization performance budget. Per-simulation timeout: 30 seconds.

**Scope note**: This is optional polish. The explorable is complete and valuable without it. If scope needs cutting, this goes first.

**Tree state**: Empty initially. Grows as simulations run.

## Technical Architecture

### Single HTML File

Everything lives in one self-contained HTML file:
- Inline CSS (dark theme, responsive layout)
- Inline JavaScript, organized as clearly separated `<script>` blocks: one for the tree visualization engine, one for the scroll observer, one for the sandbox
- Inline SVG for tree rendering
- Pre-computed tree data embedded as JSON

Expected size: 3000-4000 lines, 250-400KB. Maintainability managed through clear section comments and modular script blocks.

### Layout

- Left panel: 40% width for narrative text
- Right panel: 60% width, `position: sticky`, with auto-fit zoom (tree scales to fill available space)
- Tree is always fully visible without independent scrolling within its panel
- Minimum tree panel width: 500px. Below that breakpoint, switch to stacked layout (narrative above, tree below)

### Tree Visualization Engine

- **Renderer**: SVG-based, dynamically generated via JavaScript
- **Layout**: Simple recursive top-down layout (assign x by child index, y by depth, shift subtrees to avoid overlap)
- **Animation**: CSS transitions for node appearance/value changes, requestAnimationFrame for propagation effects. No SVG filters.
- **Interaction**: Click to inspect nodes, hover for UCB1 breakdown

### Scroll Observer

- IntersectionObserver API to detect which narrative section is in view
- Each section maps to a tree state (see Scroll State Map below)
- When a new section enters the viewport while a previous transition is still animating, the current animation is cancelled and the tree snaps to the target state immediately. Only the most recently triggered section's state animates.

### Scroll State Map

| Section | Tree State | Nodes by ID | Highlighted |
|---------|-----------|-------------|-------------|
| 1 | No tree | None | None |
| 2 | Root + 3 branch starts, A/B at depth 2, C at depth 1 | 0, 0.0, 0.0.0, 0.1, 0.1.0, 0.2 (6 nodes) | UCB1 selection path (dynamic via slider) |
| 3 | Same + one new child from expansion | 7 nodes (previous + 0.0.0.0 or similar) | Selected node + new child |
| 4 | Same + rollout chain (2 new nodes) | 9 nodes | Rollout chain appearing one by one |
| 4.5 | Same, terminal scored | 9 nodes | Terminal node with score badge |
| 5 | Same, backprop animating | 9 nodes | Backprop path (root to terminal) |
| 6 | Full tree, ~20 sims | ~30-40 nodes (exact count set by generate_data.py) | Sampling-strategy-dependent paths |
| 7 | Empty (sandbox) | 0 initially | User-driven |

### Node Identity

Each node gets a stable path-based ID: `"0"` for root, `"0.0"` for root's first child, `"0.1"` for root's second child, `"0.1.0"` for the first child of root's second child, etc. This ID is stable across simulation steps and used by the scroll observer to map sections to node highlights.

### Pre-computed Data Schema

Every node object in the tree (root and all descendants) has the same shape:

```json
{
  "id": "0.1.0",
  "state": "Reasoning text for this step...",
  "value": 0.7,
  "visits": 3,
  "is_terminal": false,
  "answer": null,
  "children": [
    { "id": "0.1.0.0", "state": "...", "value": 0.9, "visits": 2, "is_terminal": true, "answer": "A is a knight", "children": [] },
    { "id": "0.1.0.1", "state": "...", "value": 0.3, "visits": 1, "is_terminal": false, "answer": null, "children": [] }
  ]
}
```

The full trace file wraps this in simulation steps:

```json
{
  "puzzle": {
    "question": "A says 'B is a knave.' B says 'We are the same type.' Is A a knight or a knave?",
    "correct_answer": "A is a knight",
    "single_pass_wrong": "Let me think about this. A says B is a knave..."
  },
  "simulations": [
    {
      "step": 1,
      "phase": "select",
      "selected_path": ["0"],
      "tree": { "id": "0", "state": "...", "value": 0.0, "visits": 0, "is_terminal": false, "answer": null, "children": [] }
    },
    {
      "step": 1,
      "phase": "expand",
      "selected_path": ["0"],
      "new_node_id": "0.0",
      "tree": {
        "id": "0", "state": "...", "value": 0.0, "visits": 0, "is_terminal": false, "answer": null,
        "children": [
          { "id": "0.0", "state": "Let's assume A is a knight...", "value": 0.0, "visits": 0, "is_terminal": false, "answer": null, "children": [] }
        ]
      }
    },
    {
      "step": 1,
      "phase": "rollout",
      "rollout_path": ["0.0", "0.0.0"],
      "tree": { "...": "full recursive tree with all id fields" }
    },
    {
      "step": 1,
      "phase": "backprop",
      "backprop_path": ["0.0.0", "0.0", "0"],
      "score": 1.0,
      "tree": { "...": "full recursive tree with updated values" }
    }
  ]
}
```

Each simulation has 4 phases (select, expand, rollout, backprop). Each phase includes the full recursive tree snapshot (every node carries `id`, same shape as above) plus metadata about which nodes are involved. The `selected_path`, `new_node_id`, `rollout_path`, and `backprop_path` fields allow the visualization to highlight the right nodes at each step.

### Data Generation Approach

The pre-computed data is **pedagogical, not a real MCTS run**. The tree structure, reasoning traces, values, and visit counts are all hand-crafted to produce clear, instructive visualizations. Trying to run the actual MCTS library with MockGenerator would couple the trace to MCTS internals (selection order, UCB1 values) and make it fragile.

Instead, `generate_data.py` is a **direct JSON construction script** that:
1. Reads the hand-authored reasoning traces from `data/reasoning_traces.json`
2. Builds the tree snapshots directly as Python dicts, simulating what MCTS *would* do at each phase
3. Assigns path-based node IDs and computes plausible value/visits numbers
4. Outputs `data/knights_trace.json` matching the schema above

This script does **not** import or use the MCTS library. It constructs the JSON trace by hand, giving full control over the pedagogical narrative (which node is selected when, what values look like at each step, how backpropagation updates flow).

### Reasoning Traces Input Schema

`data/reasoning_traces.json` format:

```json
{
  "single_pass_wrong": "Let me think about this. A says B is a knave...",
  "branches": {
    "A": {
      "label": "Assume A is a knight",
      "steps": [
        "Let's assume A is a knight. Knights always tell the truth.",
        "A says 'B is a knave.' Since A is truthful, B must be a knave.",
        "B says 'We are the same type.' Since B is a knave, B lies. A is a knight, B is a knave: they are different types. B's lie is consistent. ANSWER: A is a knight."
      ],
      "answer": "A is a knight",
      "score": 1.0
    },
    "B": {
      "label": "Assume A is a knave",
      "steps": ["..."],
      "answer": "A is a knave",
      "score": 0.0
    },
    "C": {
      "label": "Start with B's statement",
      "steps": ["..."],
      "answer": "A is a knight",
      "score": 0.6
    }
  }
}
```

### Live Sandbox (Section 7)

- Simplified MCTS implementation in JavaScript (estimated 500-800 lines)
- Direct API calls to Ollama (localhost only, CORS-compatible)
- Same tree visualization, driven by live data instead of pre-computed
- Falls back gracefully if no API is available

## File Structure

```
../explainables/
└── mcts-reasoning/
    ├── index.html              # The explorable (self-contained)
    ├── generate_data.py        # Script to generate pre-computed tree data
    ├── data/
    │   ├── reasoning_traces.json  # Hand-authored reasoning steps per branch
    │   └── knights_trace.json     # Generated search trace (output of generate_data.py)
    └── README.md               # What this is, how to regenerate data
```

## Design Constraints

1. **Self-contained**: The HTML file must work when opened directly in a browser (file:// protocol), except for the live sandbox which needs network access.
2. **No build step**: No webpack, no npm, no bundler. Raw HTML/CSS/JS.
3. **Responsive**: Must work on desktop (primary) and degrade gracefully on tablet. Mobile is not a priority.
4. **Performance**: Tree visualization must handle ~50 nodes smoothly. No WebGL needed. No SVG filter effects for animations.
5. **Accessibility**: Keyboard navigation for nodes and interactive controls. Focus indicators on slider and toggle buttons. Semantic HTML (headings, sections, ARIA labels on interactive tree elements). Full WCAG compliance is out of scope for v1.

## What This Is NOT

- Not a full port of the Python library to JavaScript
- Not a documentation site for the library
- Not an academic paper viewer
- Not a general-purpose MCTS visualizer. It is a narrative experience about one specific idea.

## Success Criteria

1. A reader with no ML background can follow the narrative and understand *why* MCTS helps LLM reasoning
2. A reader with ML background learns something from the tree-building rollout insight and the UCB1 slider
3. The pre-computed demo is compelling without any API keys or setup
4. The visual style is distinctive and memorable, not generic "AI blog post"
