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

## Interactivity Model

**Hybrid scrollytelling**:
- Pre-computed trees drive the main narrative (no API calls needed)
- Sticky tree visualization on the right animates as the user scrolls through narrative sections on the left
- Interactive controls (UCB1 slider, sampling strategy toggles) at key moments
- Optional live sandbox at the end (requires Ollama or API key)

## Demo Problem

A **logic puzzle** (knights and knaves), not arithmetic. Example: *"A says 'B is a knave.' B says 'We are the same type.' Is A a knight or a knave?"*

Rationale: Wrong paths are *plausible but logically flawed*. Branches represent genuinely different assumptions ("A is a knight" vs. "A is a knave"), making the tree exploration visually and conceptually interesting. An arithmetic error is boring; a logical fork is compelling.

## Page Structure

### Section 1: "One Shot, One Chance" (The Problem)

**Purpose**: Motivate why MCTS matters.

**Content**: Show a pre-computed single-pass LLM attempt at the logic puzzle. The reasoning *looks* convincing but makes an unchecked assumption early and arrives at the wrong answer. Highlight the flaw.

**Transition**: *"What if the LLM could explore both assumptions simultaneously, and backtrack from the one that leads to contradiction?"* The scroll begins, the tree appears.

**Interactivity**: None. Pure narrative setup.

### Section 2: "The Tree Appears" (Selection)

**Purpose**: Introduce UCB1 and the exploration/exploitation tradeoff.

**Content**: The sticky tree visualization appears on the right. Starting from the root (the question), UCB1 selects which node to explore next. Nodes display two numbers: value (quality) and visits (exploration count).

**Interactivity**:
- **UCB1 slider**: Adjust exploration constant `c` from 0 (pure greedy) to 4 (pure exploration). The tree highlights which node would be selected at each value. Default position: √2.
- Node hover shows the UCB1 calculation breakdown.

**Key insight**: At c=0, the algorithm always picks the highest-value node (gets stuck). At c=√2, it balances. At c=4, it wastes visits on weak paths.

### Section 3: "Branching Thoughts" (Expansion)

**Purpose**: Show how the LLM generates new reasoning steps.

**Content**: From the selected node, the LLM generates a continuation. The tree grows a new child. We show the actual reasoning text: "Assume A is a knight..." on one branch, "Assume A is a knave..." on another.

**Interactivity**:
- Click any node to see its full reasoning text in a panel below the tree.
- The tree animates the new node appearing with a brief glow effect.

**Key insight**: Each branch is a different assumption the LLM is trying. Structured exploration, not random.

### Section 4: "Following the Thread" (Rollout)

**Purpose**: Show how reasoning continues to completion.

**Content**: From the new node, the LLM keeps generating steps until it reaches an answer ("ANSWER: Knight") or hits max depth. Nodes appear one by one in a chain, animated.

**Interactivity**: Play/pause button for the rollout animation. Speed control (1x, 2x, 4x).

**Key insight (tree-building rollouts)**: Unlike classical MCTS (which discards rollout results), this implementation *keeps every node in the tree*. A brief visual aside shows both approaches side-by-side: classical (nodes fade away) vs. tree-building (nodes persist). This is the novel design choice.

### Section 5: "Scores Flow Upward" (Backpropagation)

**Purpose**: Show how terminal scores propagate through the tree.

**Content**: The terminal node gets scored (0.0 to 1.0). The score propagates up: each ancestor updates its average value and increments its visit count. Animated: nodes glow brighter as values update, edges pulse with the propagating signal.

**Interactivity**: Step-through mode. Click to advance backpropagation one level at a time. Each step shows the value/visits update math.

**Key insight**: Good answers make nearby reasoning paths more attractive for future exploration. The tree *learns* where to search next.

### Section 6: "Many Paths, One Answer" (Path Sampling & Self-Consistency)

**Purpose**: Show how to extract answers from the completed tree.

**Content**: After the full search completes (display the final tree with ~20 simulations), present an interactive panel for path sampling.

**Interactivity**:
- **Strategy toggle buttons**: value, visits, diverse, top-k. Each highlights different paths in the tree with distinct colors.
- **Answer distribution histogram**: Below the tree. Shows how many paths found each answer. E.g., "3 paths say Knight, 1 says Knave."
- **Voting toggle**: Switch between majority vote and value-weighted vote. Shows confidence score.

**Key insight**: Self-consistency. When multiple independent reasoning paths agree, we can be more confident in the answer. Different sampling strategies reveal different aspects of the search.

### Section 7: "Try It Yourself" (Live Sandbox)

**Purpose**: Let curious users run MCTS on their own questions.

**Content**: A terminal-styled input box. User types a question, selects a provider (Ollama localhost URL, or paste OpenAI/Anthropic API key), sets parameters (simulations, exploration constant, max depth), and clicks "Search."

**Interactivity**: The tree builds in real-time as simulations run. Same visualization as the narrative sections, but driven by live API calls. Results appear as they complete.

**Implementation**: This section makes HTTP calls to a local Ollama instance or directly to OpenAI/Anthropic APIs from the browser. The MCTS algorithm runs in JavaScript in the browser, a simplified port of the core loop (not the full Python library).

**Scope note**: This is optional polish. The explorable is complete and valuable without it. If scope needs cutting, this goes first.

## Technical Architecture

### Single HTML File

Everything lives in one self-contained HTML file:
- Inline CSS (dark theme, responsive layout)
- Inline JavaScript (tree visualization, scroll observer, animations, optional MCTS engine)
- Inline SVG for tree rendering
- Pre-computed tree data embedded as JSON

### Tree Visualization Engine

- **Renderer**: SVG-based, dynamically generated via JavaScript
- **Layout**: Top-down tree layout using a simple recursive algorithm (Reingold-Tilford or simpler)
- **Animation**: CSS transitions for node appearance/value changes, requestAnimationFrame for propagation effects
- **Interaction**: Click to inspect nodes, hover for UCB1 breakdown

### Scroll Observer

- IntersectionObserver API to detect which narrative section is in view
- Each section maps to a tree state (which nodes exist, which is selected, what values are shown)
- Smooth transitions between states as user scrolls

### Pre-computed Data

- One complete MCTS search trace (knights-and-knaves puzzle, ~20 simulations)
- Stored as JSON: array of simulation steps, each containing the tree state after that simulation
- Generated ahead of time by running the actual Python library and serializing the tree at each step

### Data Generation Script

A Python script (`generate_explorable_data.py`) that:
1. Runs MCTS on the logic puzzle using `MockGenerator` with pre-written reasoning traces
2. Serializes the tree state after each simulation step
3. Outputs a JSON file that gets embedded in the HTML

Rationale for MockGenerator: we want deterministic, pedagogically clear reasoning traces, not whatever a live LLM happens to produce. The "reasoning" in each node should be carefully written to demonstrate the concepts clearly.

### Live Sandbox (Section 7)

- Simplified MCTS implementation in JavaScript (~200 lines)
- Direct API calls to Ollama (localhost) or OpenAI/Anthropic (CORS-permitting)
- Same tree visualization, driven by live data instead of pre-computed
- Falls back gracefully if no API is available ("Connect to Ollama at localhost:11434 to try this")

## File Structure

```
../explainables/
└── mcts-reasoning/
    ├── index.html              # The explorable (self-contained)
    ├── generate_data.py        # Script to generate pre-computed tree data
    ├── data/
    │   └── knights_trace.json  # Pre-computed search trace
    └── README.md               # What this is, how to regenerate data
```

## Design Constraints

1. **Self-contained**: The HTML file must work when opened directly in a browser (file:// protocol), except for the live sandbox which needs network access.
2. **No build step**: No webpack, no npm, no bundler. Raw HTML/CSS/JS.
3. **Responsive**: Must work on desktop (primary) and degrade gracefully on tablet. Mobile is not a priority.
4. **Performance**: Tree visualization must handle ~50 nodes smoothly. No WebGL needed.
5. **Accessibility**: Nodes should be keyboard-navigable. High contrast already provided by the dark theme.

## What This Is NOT

- Not a full port of the Python library to JavaScript
- Not a documentation site for the library
- Not an academic paper viewer
- Not a general-purpose MCTS visualizer. It's a narrative experience about one specific idea.

## Success Criteria

1. A reader with no ML background can follow the narrative and understand *why* MCTS helps LLM reasoning
2. A reader with ML background learns something from the tree-building rollout insight and the UCB1 slider
3. The pre-computed demo is compelling without any API keys or setup
4. The visual style is distinctive and memorable, not generic "AI blog post"
