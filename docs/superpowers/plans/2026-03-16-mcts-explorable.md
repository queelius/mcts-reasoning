# MCTS Explorable Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained scrollytelling HTML explorable that teaches MCTS for LLM reasoning through animated tree visualization of a knights-and-knaves logic puzzle.

**Architecture:** Pre-computed JSON trace data drives a scrollytelling narrative. A Python script constructs pedagogical tree snapshots (no MCTS library dependency). A single HTML file contains all CSS, JS, and embedded data. SVG-based tree visualization animates in response to IntersectionObserver scroll events.

**Tech Stack:** Python 3.8+ (data generation), vanilla HTML/CSS/JS (explorable), SVG (tree rendering), IntersectionObserver API (scroll tracking)

**Spec:** `docs/superpowers/specs/2026-03-16-mcts-explorable-design.md`

**Note:** This project lives in a separate directory (`../explainables/mcts-reasoning/`) from the main `mcts-reasoning` library. It is a standalone repo with its own `git init`. The data generation script does NOT import the mcts_reasoning library.

---

## File Map

```
../explainables/mcts-reasoning/
├── data/
│   ├── reasoning_traces.json    # Hand-authored reasoning steps (input)
│   └── knights_trace.json       # Generated simulation trace (output)
├── generate_data.py             # Builds knights_trace.json from reasoning_traces.json
├── test_generate_data.py        # Tests for data generation
└── index.html                   # The explorable (self-contained, ~3000 lines)
                                 # Internal structure:
                                 #   <style>        - Dark theme, layout, animations
                                 #   <div#narrative> - Sections 1-7 HTML content
                                 #   <div#tree-panel>- Sticky SVG tree + controls
                                 #   <script data-module="tree">    - Tree layout + rendering
                                 #   <script data-module="scroll">  - IntersectionObserver + state machine
                                 #   <script data-module="controls">- UCB1 slider, sampling toggles, backprop stepper
                                 #   <script data-module="sandbox"> - Ollama MCTS engine (optional)
                                 #   <script data-module="data">    - Embedded JSON trace
                                 #   <script data-module="main">    - Init + wiring
```

---

## Chunk 1: Data Foundation

### Task 1: Create project structure and reasoning traces

**Files:**
- Create: `../explainables/mcts-reasoning/data/reasoning_traces.json`

- [ ] **Step 1: Create directory structure**

```bash
mkdir -p ../explainables/mcts-reasoning/data
```

- [ ] **Step 2: Write reasoning_traces.json**

Create the file with the full reasoning traces from the spec (Appendix A). Include the single-pass wrong attempt and all three branches with their steps, answers, and scores. Follow the Reasoning Traces Input Schema from the spec exactly.

The single-pass wrong attempt should be a 4-5 sentence paragraph that assumes A is a knave, reasons plausibly but never checks the alternative, and concludes "A is a knave" with false confidence.

- [ ] **Step 3: Validate JSON is well-formed**

```bash
python3 -c "import json; json.load(open('../explainables/mcts-reasoning/data/reasoning_traces.json')); print('Valid JSON')"
```

Expected: `Valid JSON`

- [ ] **Step 4: Commit**

```bash
cd ../explainables/mcts-reasoning && git init && git add data/reasoning_traces.json
git commit -m "feat: add hand-authored reasoning traces for knights-and-knaves puzzle"
```

---

### Task 2: Write data generation script with tests

**Files:**
- Create: `../explainables/mcts-reasoning/generate_data.py`
- Create: `../explainables/mcts-reasoning/test_generate_data.py`

- [ ] **Step 1: Write failing test for trace loading**

```python
# test_generate_data.py
import json
import pytest
from generate_data import load_traces

def test_load_traces_returns_branches():
    traces = load_traces("data/reasoning_traces.json")
    assert "single_pass_wrong" in traces
    assert "branches" in traces
    assert set(traces["branches"].keys()) == {"A", "B", "C"}
    for branch in traces["branches"].values():
        assert "label" in branch
        assert "steps" in branch
        assert "answer" in branch
        assert "score" in branch
        assert len(branch["steps"]) >= 2
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd ../explainables/mcts-reasoning && python3 -m pytest test_generate_data.py::test_load_traces_returns_branches -v
```

Expected: FAIL (no module named generate_data)

- [ ] **Step 3: Implement load_traces**

```python
# generate_data.py
"""Generate pre-computed MCTS trace data for the explorable.

This script builds pedagogical tree snapshots directly as dicts.
It does NOT import or use the mcts_reasoning library.
"""
import json
from pathlib import Path


def load_traces(path: str) -> dict:
    """Load hand-authored reasoning traces from JSON."""
    with open(path) as f:
        return json.load(f)
```

- [ ] **Step 4: Run test to verify it passes**

```bash
cd ../explainables/mcts-reasoning && python3 -m pytest test_generate_data.py::test_load_traces_returns_branches -v
```

Expected: PASS

- [ ] **Step 5: Write failing test for node construction**

```python
# test_generate_data.py (append)
from generate_data import make_node

def test_make_node_structure():
    node = make_node("0.1", "some reasoning", value=0.7, visits=3)
    assert node["id"] == "0.1"
    assert node["state"] == "some reasoning"
    assert node["value"] == 0.7
    assert node["visits"] == 3
    assert node["is_terminal"] is False
    assert node["answer"] is None
    assert node["children"] == []

def test_make_node_terminal():
    node = make_node("0.0.0", "ANSWER: A is a knight", value=1.0, visits=1,
                     is_terminal=True, answer="A is a knight")
    assert node["is_terminal"] is True
    assert node["answer"] == "A is a knight"
```

- [ ] **Step 6: Run tests to verify they fail**

```bash
cd ../explainables/mcts-reasoning && python3 -m pytest test_generate_data.py -k "make_node" -v
```

Expected: FAIL

- [ ] **Step 7: Implement make_node**

```python
# generate_data.py (append)

def make_node(node_id: str, state: str, value: float = 0.0, visits: int = 0,
              is_terminal: bool = False, answer: str | None = None,
              children: list | None = None) -> dict:
    """Create a node dict matching the pre-computed data schema."""
    return {
        "id": node_id,
        "state": state,
        "value": value,
        "visits": visits,
        "is_terminal": is_terminal,
        "answer": answer,
        "children": children or [],
    }
```

- [ ] **Step 8: Run tests to verify they pass**

```bash
cd ../explainables/mcts-reasoning && python3 -m pytest test_generate_data.py -k "make_node" -v
```

Expected: PASS

- [ ] **Step 9: Write failing test for full trace generation**

```python
# test_generate_data.py (append)

def test_generate_trace_structure():
    trace = generate_trace("data/reasoning_traces.json")
    assert "puzzle" in trace
    assert "simulations" in trace
    assert trace["puzzle"]["correct_answer"] == "A is a knight"

    # Each simulation step has 4 phases
    phases = [s["phase"] for s in trace["simulations"]]
    assert phases[:4] == ["select", "expand", "rollout", "backprop"]

    # Every phase has a tree with an id field
    for sim in trace["simulations"]:
        assert "tree" in sim
        assert "id" in sim["tree"]
        assert sim["tree"]["id"] == "0"  # root is always "0"

def test_generate_trace_node_ids_are_path_based():
    trace = generate_trace("data/reasoning_traces.json")
    # Collect all node IDs from the final simulation step
    final_tree = trace["simulations"][-1]["tree"]
    ids = collect_all_ids(final_tree)
    # Root must be "0"
    assert "0" in ids
    # Children of root must be "0.0", "0.1", "0.2"
    for child_id in ["0.0", "0.1", "0.2"]:
        assert child_id in ids

def collect_all_ids(node: dict) -> set:
    """Recursively collect all node IDs from a tree."""
    ids = {node["id"]}
    for child in node.get("children", []):
        ids |= collect_all_ids(child)
    return ids

def test_generate_trace_has_terminal_nodes():
    trace = generate_trace("data/reasoning_traces.json")
    final_tree = trace["simulations"][-1]["tree"]
    terminals = find_terminals(final_tree)
    assert len(terminals) >= 3  # At least one per branch
    # At least one correct answer
    answers = [t["answer"] for t in terminals]
    assert "A is a knight" in answers

def find_terminals(node: dict) -> list:
    """Recursively find all terminal nodes."""
    result = []
    if node["is_terminal"]:
        result.append(node)
    for child in node.get("children", []):
        result.extend(find_terminals(child))
    return result

def test_section_node_counts():
    """Verify node counts match the spec's Scroll State Map."""
    trace = generate_trace("data/reasoning_traces.json")
    # Section 2 uses phases 0-19 (sims 1-5). Tree should have 6 nodes.
    sec2_tree = trace["simulations"][19]["tree"]  # last phase of sim 5
    assert len(collect_all_ids(sec2_tree)) == 6
    # Section 3 adds one node (7 total)
    sec3_tree = trace["simulations"][20]["tree"]
    assert len(collect_all_ids(sec3_tree)) == 7
    # Full tree should have 30-40 nodes
    final_tree = trace["simulations"][-1]["tree"]
    node_count = len(collect_all_ids(final_tree))
    assert 30 <= node_count <= 40, f"Expected 30-40 nodes, got {node_count}"
```

- [ ] **Step 10: Run tests to verify they fail**

```bash
cd ../explainables/mcts-reasoning && python3 -m pytest test_generate_data.py -k "generate_trace" -v
```

Expected: FAIL

- [ ] **Step 11: Implement generate_trace**

This is the core function. It reads reasoning traces and constructs a sequence of simulation steps, each with 4 phases. The function builds the tree incrementally as Python dicts, simulating what MCTS would do.

**Concrete simulation schedule (20 simulations):**

| Sim | Branch | Action | Node expanded | Notes |
|-----|--------|--------|---------------|-------|
| 1 | A | Expand root, add A step 1 | 0.0 | First branch |
| 2 | B | Expand root, add B step 1 | 0.1 | Second branch |
| 3 | C | Expand root, add C step 1 | 0.2 | Third branch |
| 4 | A | Expand 0.0, add A step 2 | 0.0.0 | Continue A |
| 5 | B | Expand 0.1, add B step 2 | 0.1.0 | Continue B |
| 6 | C | Expand 0.2, add C step 2 | 0.2.0 | Continue C |
| 7 | A | Expand 0.0.0, add A step 3 (terminal) | 0.0.0.0 | A reaches answer (score 1.0) |
| 8 | B | Expand 0.1.0, add B step 3 (terminal) | 0.1.0.0 | B reaches answer (score 0.0) |
| 9 | C | Expand 0.2.0, add C step 3 (terminal) | 0.2.0.0 | C reaches answer (score 0.6) |
| 10-20 | A bias | Revisit Branch A subtree | varies | Add exploration branches off A's path; some reach terminal with score 0.8-1.0 |

Simulations 10-20 add variety to Branch A by expanding new children from nodes 0.0 and 0.0.0 (representing slightly different reasoning continuations). These produce additional terminal nodes with scores 0.7-1.0 to make the sampling section interesting. Each new continuation should be a minor variation (e.g., "Let me verify: A is a knight, so A tells truth..." or "Double-checking our work...").

**Section-to-simulation-phase mapping:**

| Section | Simulation phases (indices into `simulations[]`) |
|---------|---------------------------------------------------|
| 1 | No tree. Data comes from `puzzle.single_pass_wrong`. |
| 2 | Phases 0-19 (sims 1-5, all 4 phases each = 20 entries). Tree has 6 nodes. |
| 3 | Phase 20 (sim 6, expand). Shows new child appearing. |
| 4 | Phases 21-22 (sim 6, rollout + sim 7 expand). Rollout chain. |
| 4.5 | Phase 23 (sim 7, backprop with score on terminal). |
| 5 | Phase 23 same backprop phase, but stepped through interactively. |
| 6 | Final phase (last entry in `simulations[]`). Full tree. |
| 7 | No pre-computed data. Sandbox builds its own tree. |

**Implementation steps:**

1. Start with a root node containing the puzzle question
2. For each simulation in the schedule above:
   - **select**: Walk from root to the target leaf using parent IDs
   - **expand**: Add a new child with the next reasoning step from the branch
   - **rollout**: For sims 1-9, rollout and expand happen in the same step (each step IS the expansion). For sims 10-20, if the expanded node is not terminal, add the rest of the branch as a chain.
   - **backprop**: Update values along the path from the newest terminal (or expanded node) to root. `node.value = total_score_propagated / node.visits`.
3. Snapshot the full tree (deep copy) at each phase. Include `selected_path`, `new_node_id`, `rollout_path`, and `backprop_path` metadata.
4. Values and visits must be internally consistent at every snapshot.

- [ ] **Step 12: Run all tests**

```bash
cd ../explainables/mcts-reasoning && python3 -m pytest test_generate_data.py -v
```

Expected: all PASS

- [ ] **Step 13: Write failing test for the CLI entry point**

```python
# test_generate_data.py (append)
import subprocess

def test_cli_generates_output_file(tmp_path):
    """Running the script produces knights_trace.json."""
    result = subprocess.run(
        ["python3", "generate_data.py",
         "--traces", "data/reasoning_traces.json",
         "--output", str(tmp_path / "output.json")],
        capture_output=True, text=True
    )
    assert result.returncode == 0
    output = json.loads((tmp_path / "output.json").read_text())
    assert "puzzle" in output
    assert "simulations" in output
    assert len(output["simulations"]) > 0
```

- [ ] **Step 14: Add CLI entry point to generate_data.py**

```python
# generate_data.py (append at bottom)
import argparse

def main():
    parser = argparse.ArgumentParser(description="Generate MCTS trace data")
    parser.add_argument("--traces", default="data/reasoning_traces.json",
                        help="Path to reasoning traces JSON")
    parser.add_argument("--output", default="data/knights_trace.json",
                        help="Output path for generated trace")
    args = parser.parse_args()

    trace = generate_trace(args.traces)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(trace, f, indent=2)
    print(f"Generated {output} ({len(trace['simulations'])} phases)")

if __name__ == "__main__":
    main()
```

- [ ] **Step 15: Run all tests**

```bash
cd ../explainables/mcts-reasoning && python3 -m pytest test_generate_data.py -v
```

Expected: all PASS

- [ ] **Step 16: Generate the actual trace data**

```bash
cd ../explainables/mcts-reasoning && python3 generate_data.py
```

Expected: `Generated data/knights_trace.json (N phases)`

- [ ] **Step 17: Commit**

```bash
cd ../explainables/mcts-reasoning
git add generate_data.py test_generate_data.py data/knights_trace.json
git commit -m "feat: data generation script with tests and pre-computed trace"
```

---

## Chunk 2: HTML Shell and Tree Visualization Engine

### Task 3: Create HTML shell with dark theme and scrollytelling layout

**Files:**
- Create: `../explainables/mcts-reasoning/index.html`

- [ ] **Step 1: Write the HTML shell**

Create `index.html` with:
- `<!DOCTYPE html>` and standard head (title, meta charset, viewport)
- `<style>` block with dark theme colors from spec:
  - `--bg: #1a1a2e`, `--bg-dark: #0d0d1a`, `--accent: #e94560`, `--accent2: #4a9eff`
  - `font-family: 'Courier New', monospace`
  - Body: full dark background, no margin
- Hero section with title "Watch an LLM Think" and subtitle
- Two-panel layout:
  - `#narrative` (left, 40% width): scrollable content sections
  - `#tree-panel` (right, 60% width): `position: sticky; top: 0; height: 100vh`
- Responsive breakpoint: below 500px tree panel width, stack vertically
- Placeholder `<section>` elements for each narrative section (1-7), each with `data-section` attribute
- Empty `<svg id="tree-svg">` inside the tree panel

- [ ] **Step 2: Open in browser and verify layout**

```bash
xdg-open ../explainables/mcts-reasoning/index.html 2>/dev/null || echo "Open ../explainables/mcts-reasoning/index.html in browser"
```

Verify: dark background, two-panel layout, placeholder text visible on left, empty right panel.

- [ ] **Step 3: Commit**

```bash
cd ../explainables/mcts-reasoning && git add index.html
git commit -m "feat: HTML shell with dark theme and scrollytelling layout"
```

---

### Task 4: Build tree visualization engine

**Files:**
- Modify: `../explainables/mcts-reasoning/index.html` (add `<script data-module="tree">`)

- [ ] **Step 1: Implement tree layout algorithm**

Add a `<script data-module="tree">` block. Implement:

```javascript
// TreeRenderer: takes a tree JSON object, renders SVG
class TreeRenderer {
  constructor(svgElement, options = {}) {
    this.svg = svgElement;
    this.nodeRadius = options.nodeRadius || 20;
    this.levelHeight = options.levelHeight || 80;
    this.siblingSpacing = options.siblingSpacing || 60;
    this.nodes = new Map();  // id -> {x, y, data, element}
  }

  // Compute layout positions for all nodes (recursive, top-down)
  layout(tree) { ... }

  // Render tree as SVG elements (circles + lines + labels)
  render(tree, highlights = {}) { ... }

  // Update node appearance (value/visits labels, colors)
  updateNode(nodeId, props) { ... }

  // Highlight a path (array of node IDs)
  highlightPath(nodeIds, color) { ... }

  // Clear all highlights
  clearHighlights() { ... }

  // Auto-fit: scale SVG viewBox to contain all nodes
  autoFit() { ... }
}
```

The layout algorithm (simple post-order approach):
1. Assign `y = depth * levelHeight`
2. For leaf nodes, assign `x` sequentially left-to-right with `siblingSpacing` gap
3. For internal nodes, `x = average of children's x`
4. Shift subtrees to prevent overlap: walk the tree in post-order. For each internal node with multiple subtrees, check if the rightmost node of the left subtree and the leftmost node of the right subtree are closer than `siblingSpacing`. If so, shift the right subtree rightward by the difference. Propagate the shift to all descendants.

**Animation note:** All CSS animations must use `transform` and `opacity` only (GPU-composited). Do not animate SVG geometry attributes (`cx`, `cy`, `r`) directly as browser support is inconsistent. Wrap each node in a `<g>` element and animate the `<g>` transform.

Each node renders as:
- A `<circle>` with stroke color based on state (normal: dim, highlighted: accent, terminal: bright)
- A `<text>` below showing `v=X.X` (value)
- A `<text>` showing `n=N` (visits)
- Edges as `<line>` elements from parent center to child center

- [ ] **Step 2: Test with hardcoded sample tree**

Add a temporary `<script>` at the bottom that creates a small test tree and renders it:

```javascript
const testTree = {
  id: "0", state: "Root", value: 0.5, visits: 4, is_terminal: false, answer: null,
  children: [
    { id: "0.0", state: "A", value: 0.8, visits: 2, is_terminal: false, answer: null, children: [
      { id: "0.0.0", state: "A1", value: 1.0, visits: 1, is_terminal: true, answer: "Knight", children: [] }
    ]},
    { id: "0.1", state: "B", value: 0.2, visits: 1, is_terminal: false, answer: null, children: [] },
    { id: "0.2", state: "C", value: 0.6, visits: 1, is_terminal: false, answer: null, children: [] }
  ]
};
const renderer = new TreeRenderer(document.getElementById('tree-svg'));
renderer.render(testTree);
```

Open in browser. Verify: tree renders with circles, edges, and value/visit labels in the sticky right panel.

- [ ] **Step 3: Implement node click-to-inspect**

When a node `<circle>` is clicked, show the node's `state` text in a panel below the tree (`#node-detail`). Add the panel HTML and the click handler.

- [ ] **Step 4: Implement hover for UCB1 breakdown**

On hover, show a tooltip with the UCB1 formula values for that node: `exploitation = value`, `exploration = c * sqrt(ln(parent.visits) / visits)`, `UCB1 = exploitation + exploration`. Use a `<div>` tooltip positioned near the cursor.

- [ ] **Step 5: Remove test tree, commit**

Remove the hardcoded test tree. The tree will be driven by scroll state in the next task.

```bash
cd ../explainables/mcts-reasoning && git add index.html
git commit -m "feat: SVG tree visualization engine with layout, click, and hover"
```

---

### Task 5: Implement tree animations

**Files:**
- Modify: `../explainables/mcts-reasoning/index.html` (extend tree module)

- [ ] **Step 1: Node appearance animation**

When a node is added to the tree, animate it: start at `opacity: 0; transform: scale(0)`, transition to `opacity: 1; transform: scale(1)` over 300ms. Use CSS transitions on SVG `<g>` elements wrapping each node.

- [ ] **Step 2: Value update animation**

When a node's value changes, briefly pulse the circle (scale 1.0 -> 1.15 -> 1.0 over 400ms) and transition the color from current to brighter accent over 300ms.

- [ ] **Step 3: Edge pulse animation**

When backpropagation traverses an edge, animate the edge: briefly increase stroke-width and change color to accent, then return. Use CSS transitions.

- [ ] **Step 4: Path highlight animation**

`highlightPath(nodeIds, color)` should: set all non-path nodes to dimmed opacity (0.3), set path nodes and edges to full opacity with the specified color, and add a subtle pulsing glow (CSS `box-shadow` animation on a `<foreignObject>` wrapper or opacity pulse on the circle).

- [ ] **Step 5: Test animations with hardcoded sequence**

Add a temporary script that renders a tree, waits 1 second, adds a node (tests appearance), waits 1 second, updates a value (tests pulse), waits 1 second, highlights a path. Open in browser and verify each animation.

- [ ] **Step 6: Remove test script, commit**

```bash
cd ../explainables/mcts-reasoning && git add index.html
git commit -m "feat: tree animations (appear, pulse, edge, path highlight)"
```

---

## Chunk 3: Data Embedding, Scroll Observer, and Narrative Content

### Task 6: Embed pre-computed data

**Files:**
- Modify: `../explainables/mcts-reasoning/index.html` (add `<script data-module="data">` and `<script data-module="main">`)

All subsequent tasks need the trace data available in the HTML. Embed it now so Tasks 7-11 can be tested interactively.

- [ ] **Step 1: Embed trace data**

Read `data/knights_trace.json` and embed it as a `<script data-module="data">` block:

```html
<script data-module="data">
const TRACE_DATA = { /* contents of knights_trace.json */ };
</script>
```

- [ ] **Step 2: Write main initialization script**

```html
<script data-module="main">
document.addEventListener('DOMContentLoaded', () => {
  const svg = document.getElementById('tree-svg');
  const renderer = new TreeRenderer(svg);
  // ScrollStateManager will be added in the next task
  // For now, render the Section 2 tree as a smoke test
  const sec2Tree = TRACE_DATA.simulations[19].tree;
  renderer.render(sec2Tree);
});
</script>
```

- [ ] **Step 3: Open in browser, verify tree renders from embedded data**

Expected: the Section 2 tree (6 nodes) renders in the sticky right panel.

- [ ] **Step 4: Commit**

```bash
cd ../explainables/mcts-reasoning && git add index.html
git commit -m "feat: embed pre-computed trace data"
```

---

### Task 7: Implement scroll observer and state machine

**Files:**
- Modify: `../explainables/mcts-reasoning/index.html` (add `<script data-module="scroll">`)

- [ ] **Step 1: Implement ScrollStateManager**

```javascript
class ScrollStateManager {
  constructor(treeRenderer, traceData) {
    this.renderer = treeRenderer;
    this.data = traceData;
    this.currentSection = null;
    this.animationFrame = null;
    this.observers = [];
  }

  // Set up IntersectionObserver for each <section data-section="N">
  init() { ... }

  // Called when a section enters the viewport
  onSectionEnter(sectionId) {
    // Cancel any in-progress animation
    if (this.animationFrame) {
      cancelAnimationFrame(this.animationFrame);
      this.animationFrame = null;
    }
    // Snap to target state if previous animation was running
    // Then animate to new state
    this.transitionTo(sectionId);
  }

  // Map section ID to tree state and render
  transitionTo(sectionId) { ... }
}
```

Section-to-state mapping (from spec Scroll State Map):
- `1`: Hide tree panel entirely
- `2`: Show tree from simulation data at ~5 sims. Enable UCB1 slider.
- `3`: Same tree + highlight selected node + animate new child appearing
- `4`: Same + animate rollout chain
- `4.5`: Same + show score badge on terminal node
- `5`: Same + animate backpropagation
- `6`: Show full tree (final simulation). Enable sampling toggles.
- `7`: Clear tree. Show sandbox UI.

- [ ] **Step 2: Wire up IntersectionObserver**

Each `<section data-section="X">` is observed. Threshold: 0.5 (section must be 50% visible to trigger). When triggered, call `onSectionEnter(X)`.

- [ ] **Step 3: Test scroll transitions with placeholder data**

Create a minimal trace with 2-3 simulation steps. Scroll through the page and verify the tree updates as sections come into view. Verify fast-scroll cancellation works (scroll quickly past multiple sections, tree should snap to final visible section).

- [ ] **Step 4: Commit**

```bash
cd ../explainables/mcts-reasoning && git add index.html
git commit -m "feat: scroll observer with section-to-tree-state mapping"
```

---

### Task 7: Write narrative HTML content for sections 1-5

**Files:**
- Modify: `../explainables/mcts-reasoning/index.html` (fill in `<section>` content)

- [ ] **Step 1: Section 1 content ("One Shot, One Chance")**

Replace placeholder with:
- Intro paragraph presenting the puzzle
- Styled block showing the LLM's single-pass wrong attempt (from `puzzle.single_pass_wrong` in trace data). Style as a terminal/monospace block with the accent color.
- Annotation highlighting the flaw (e.g., a `<span class="flaw">` with a subtle red underline or strikethrough on the wrong assumption)
- Transition paragraph: "What if the LLM could explore both assumptions simultaneously, and backtrack from the one that leads to contradiction?"

- [ ] **Step 2: Section 2 content ("The Tree Appears")**

- Brief explanation of selection: "The algorithm looks at each node and asks: should I explore this more, or try something new?"
- UCB1 formula shown in monospace: `UCB1 = value + c * sqrt(ln(parent_visits) / visits)`
- Explanation of the two terms: exploitation (value) vs. exploration (the sqrt term)
- Placeholder for UCB1 slider (implemented in Task 8)

- [ ] **Step 3: Section 3 content ("Branching Thoughts")**

- Explanation: "From the selected node, the LLM generates a new reasoning step."
- Show two example continuations side by side: "Assume A is a knight..." and "Assume A is a knave..."
- Note: "Click any node in the tree to see its full reasoning."

- [ ] **Step 4: Section 4 content ("Following the Thread")**

- Explanation of rollout: keep reasoning until we reach an answer or hit max depth
- Placeholder for play/pause and speed controls (implemented in Task 8)
- Collapsible aside: "How is this different from regular MCTS?" with the tree-building vs. classical comparison text

- [ ] **Step 5: Section 4.5 content ("What's a Good Answer?")**

- 2-3 sentence paragraph about scoring
- "A correct derivation with no contradictions scores 1.0. A contradiction scores 0.0. Weak reasoning that happens to reach the right answer scores somewhere in between."

- [ ] **Step 6: Section 5 content ("Scores Flow Upward")**

- Explanation of backpropagation: score flows from terminal to root, each ancestor updates
- "Good answers make nearby paths more attractive for future exploration."
- Placeholder for step-through controls (implemented in Task 8)

- [ ] **Step 7: Verify content renders correctly**

Open in browser, scroll through. Verify text content, styling, and transitions.

- [ ] **Step 8: Commit**

```bash
cd ../explainables/mcts-reasoning && git add index.html
git commit -m "feat: narrative content for sections 1-5"
```

---

## Chunk 4: Interactive Controls

### Task 8: UCB1 slider (Section 2)

**Files:**
- Modify: `../explainables/mcts-reasoning/index.html` (add `<script data-module="controls">`)

- [ ] **Step 1: Add slider HTML**

In Section 2, add:
```html
<div class="ucb1-slider">
  <label>exploration constant (c)</label>
  <input type="range" id="ucb1-c" min="0" max="4" step="0.1" value="1.414">
  <div class="slider-labels">
    <span>0 (greedy)</span>
    <span class="current-value">sqrt(2)</span>
    <span>4 (explore)</span>
  </div>
</div>
```

Style with accent colors, monospace font.

- [ ] **Step 2: Implement UCB1 path computation**

```javascript
function computeUCB1Selection(tree, c) {
  // Walk from root to leaf, at each level choosing the child
  // with highest UCB1 score. Return the path as array of node IDs.
  const path = [tree.id];
  let node = tree;
  while (node.children.length > 0) {
    let bestChild = null;
    let bestScore = -Infinity;
    for (const child of node.children) {
      if (child.visits === 0) { bestChild = child; break; }
      const exploit = child.value;
      const explore = c * Math.sqrt(Math.log(node.visits) / child.visits);
      const score = exploit + explore;
      if (score > bestScore) { bestScore = score; bestChild = child; }
    }
    path.push(bestChild.id);
    node = bestChild;
  }
  return path;
}
```

- [ ] **Step 3: Wire slider to tree highlight with path annotations**

On `input` event, compute the selection path for the current `c` value, call `renderer.highlightPath(path, accent)`. Update the `.current-value` label.

Additionally, at each internal node along the highlighted path, show a small annotation (tooltip or inline label) explaining the selection choice. For example, next to node 0.0: "UCB1=1.42 (chosen) vs 0.1: 0.98, 0.2: 1.10". This shows why UCB1 picked that child over its siblings. Use `computeUCB1Selection` to also return the per-child scores and render them as small `<text>` annotations near each path node.

- [ ] **Step 4: Test slider interaction**

Open in browser, scroll to Section 2, drag slider. Verify:
- At c=0: path goes to highest-value child at each level
- At c=1.414: balanced selection
- At c=4: path goes to least-visited child
- Path highlight updates smoothly as slider moves

- [ ] **Step 5: Commit**

```bash
cd ../explainables/mcts-reasoning && git add index.html
git commit -m "feat: UCB1 exploration constant slider with live path highlighting"
```

---

### Task 9: Rollout animation controls (Section 4)

**Files:**
- Modify: `../explainables/mcts-reasoning/index.html` (extend controls module)

- [ ] **Step 1: Add play/pause and speed controls HTML**

```html
<div class="rollout-controls">
  <button id="rollout-play">Play</button>
  <div class="speed-buttons">
    <button data-speed="1" class="active">1x</button>
    <button data-speed="2">2x</button>
    <button data-speed="4">4x</button>
  </div>
</div>
```

- [ ] **Step 2: Implement RolloutAnimator**

```javascript
class RolloutAnimator {
  constructor(renderer, rolloutPath, baseDelay = 1500) {
    this.renderer = renderer;
    this.path = rolloutPath;
    this.baseDelay = baseDelay;
    this.speed = 1;
    this.currentStep = 0;
    this.playing = false;
    this.timer = null;
  }
  play() { ... }   // Start/resume animation
  pause() { ... }  // Pause
  step() { ... }   // Show next node in rollout path
  setSpeed(s) { this.speed = s; }
}
```

Base delay: 1500ms (1x). Each step reveals the next node in the rollout chain using the node appearance animation.

- [ ] **Step 3: Wire controls to animator**

Play/pause button toggles. Speed buttons update `setSpeed()`. Animator is created by ScrollStateManager when Section 4 enters viewport.

- [ ] **Step 4: Test rollout animation**

Open in browser, scroll to Section 4. Click Play, verify nodes appear one at a time. Change speed, verify timing changes. Pause/resume.

- [ ] **Step 5: Commit**

```bash
cd ../explainables/mcts-reasoning && git add index.html
git commit -m "feat: rollout animation with play/pause and speed controls"
```

---

### Task 10: Backpropagation stepper (Section 5)

**Files:**
- Modify: `../explainables/mcts-reasoning/index.html` (extend controls module)

- [ ] **Step 1: Add step button HTML**

```html
<div class="backprop-controls">
  <button id="backprop-step">Step</button>
  <div id="backprop-math" class="math-display"></div>
</div>
```

- [ ] **Step 2: Implement BackpropStepper**

```javascript
class BackpropStepper {
  constructor(renderer, backpropPath, score, treeData) {
    this.renderer = renderer;
    this.path = backpropPath;  // [terminal_id, ..., root_id]
    this.score = score;
    this.tree = treeData;
    this.currentStep = 0;
  }
  step() {
    // Highlight current node in path
    // Update its value/visits display
    // Show math: "value = (old_total + score) / new_visits"
    // Animate edge from previous node
  }
  reset() { this.currentStep = 0; }
}
```

- [ ] **Step 3: Wire step button**

Each click advances one level up the backprop path. The math display shows the value update formula. Edge pulse animation fires on each step.

- [ ] **Step 4: Test backprop stepping**

Open in browser, scroll to Section 5. Click Step repeatedly. Verify: each click highlights the next ancestor, updates its value/visits labels, shows the math, and pulses the edge.

- [ ] **Step 5: Commit**

```bash
cd ../explainables/mcts-reasoning && git add index.html
git commit -m "feat: backpropagation step-through with value update math"
```

---

## Chunk 5: Path Sampling, Full Tree, and Polish

### Task 11: Write narrative content for Section 6 and implement sampling toggles

**Files:**
- Modify: `../explainables/mcts-reasoning/index.html`

- [ ] **Step 1: Section 6 HTML content**

Add Phase 6a ("Which paths should we look at?") narrative text and Phase 6b ("How do we pick the final answer?") text. Add toggle buttons:

```html
<div class="sampling-controls">
  <div class="strategy-toggles">
    <button data-strategy="value" class="active">value</button>
    <button data-strategy="visits">visits</button>
    <button data-strategy="diverse">diverse</button>
    <button data-strategy="topk">top-k</button>
  </div>
</div>
<div id="answer-histogram"></div>
<div class="voting-controls">
  <button data-vote="majority" class="active">majority vote</button>
  <button data-vote="weighted">weighted vote</button>
  <div id="confidence-display"></div>
</div>
```

- [ ] **Step 2: Implement sampling strategies in JS**

```javascript
function samplePaths(tree, strategy, k = 5) {
  const terminals = findTerminals(tree);
  switch (strategy) {
    case 'value':   return terminals.sort((a, b) => b.value - a.value).slice(0, k);
    case 'visits':  return terminals.sort((a, b) => b.visits - a.visits).slice(0, k);
    case 'diverse': return selectDiverse(terminals, k);
    case 'topk':    return terminals.sort((a, b) => b.value - a.value).slice(0, k);
  }
}

function selectDiverse(terminals, k) {
  // Group by answer, take top from each group, then fill remaining
}
```

- [ ] **Step 3: Implement answer histogram**

Render a simple bar chart in `#answer-histogram` using inline SVG. Each bar represents an answer ("Knight" vs "Knave"), height proportional to count. Color-coded by answer.

- [ ] **Step 4: Implement voting display**

Majority vote: count answers, pick most common. Weighted vote: sum values per answer, pick highest. Display confidence as percentage.

- [ ] **Step 5: Wire toggles to tree and histogram**

Strategy buttons update `renderer.highlightPath()` for each sampled path. Voting buttons update the confidence display and histogram highlighting.

- [ ] **Step 6: Test sampling interaction**

Open in browser, scroll to Section 6. Toggle strategies, verify different paths highlight. Toggle voting, verify confidence changes.

- [ ] **Step 7: Commit**

```bash
cd ../explainables/mcts-reasoning && git add index.html
git commit -m "feat: path sampling toggles, answer histogram, and voting display"
```

---

### Task 12: End-to-end integration test

**Files:**
- Modify: `../explainables/mcts-reasoning/index.html` (update main init script)

Data was embedded in Task 6. Now wire the ScrollStateManager to the real trace data (replacing the smoke test renderer).

- [ ] **Step 1: Update main init to use ScrollStateManager**

Replace the smoke test in `<script data-module="main">`:

```javascript
document.addEventListener('DOMContentLoaded', () => {
  const svg = document.getElementById('tree-svg');
  const renderer = new TreeRenderer(svg);
  const scrollManager = new ScrollStateManager(renderer, TRACE_DATA);
  scrollManager.init();
});
```

- [ ] **Step 2: End-to-end scroll test**

Open in browser. Scroll through entire narrative from Section 1 to Section 6. Verify:
- Section 1: No tree, narrative text visible
- Section 2: Tree appears, UCB1 slider works
- Section 3: Node expansion animation
- Section 4: Rollout animation with play/pause
- Section 4.5: Score badge appears
- Section 5: Backpropagation stepping
- Section 6: Full tree, sampling toggles, histogram, voting

- [ ] **Step 3: Fix any visual issues**

Adjust spacing, font sizes, colors, animation timing based on the end-to-end test.

- [ ] **Step 4: Commit**

```bash
cd ../explainables/mcts-reasoning && git add index.html
git commit -m "feat: wire scroll state manager to trace data for full integration"
```

---

### Task 13: Tree-building rollout aside (Section 4 collapsible)

**Files:**
- Modify: `../explainables/mcts-reasoning/index.html`

- [ ] **Step 1: Add collapsible aside HTML**

In Section 4, add:
```html
<details class="aside">
  <summary>How is this different from regular MCTS?</summary>
  <div class="aside-content">
    <p>Classical MCTS discards rollout nodes. This implementation keeps them.</p>
    <div class="split-comparison" id="rollout-comparison">
      <!-- Two mini SVG trees rendered by JS -->
    </div>
  </div>
</details>
```

- [ ] **Step 2: Implement mini-tree comparison animation**

When `<details>` opens, render two small ~5-node trees side by side:
- Left ("Classical"): Rollout nodes appear then fade to dotted outlines
- Right ("Tree-building"): Rollout nodes appear and stay solid
Auto-play on open using a MutationObserver or `toggle` event.

- [ ] **Step 3: Test aside**

Click the summary to expand. Verify both mini-trees animate. Click again to collapse.

- [ ] **Step 4: Commit**

```bash
cd ../explainables/mcts-reasoning && git add index.html
git commit -m "feat: collapsible tree-building vs classical rollout comparison"
```

---

## Chunk 6: Live Sandbox (Optional)

### Task 14: Section 7 sandbox UI

**Files:**
- Modify: `../explainables/mcts-reasoning/index.html` (add `<script data-module="sandbox">`)

- [ ] **Step 1: Section 7 HTML content**

```html
<section data-section="7">
  <h2>Try It Yourself</h2>
  <div class="sandbox">
    <div class="terminal-input">
      <span class="prompt">></span>
      <input type="text" id="sandbox-question" placeholder="What is the capital of France?">
    </div>
    <div class="sandbox-config">
      <label>Ollama endpoint: <input type="text" id="sandbox-endpoint" value="http://localhost:11434"></label>
      <label>Model: <input type="text" id="sandbox-model" value="llama3.2"></label>
      <label>Simulations: <input type="number" id="sandbox-sims" value="10" min="1" max="30"></label>
      <label>Exploration (c): <input type="number" id="sandbox-c" value="1.414" step="0.1" min="0" max="4"></label>
      <label>Max depth: <input type="number" id="sandbox-depth" value="5" min="2" max="10"></label>
    </div>
    <button id="sandbox-run">Search</button>
    <div id="sandbox-status"></div>
  </div>
</section>
```

- [ ] **Step 2: Implement JS MCTS engine**

A simplified MCTS loop in JavaScript:
- `select(root, c)`: UCB1 traversal to leaf
- `expand(node, question, endpoint, model)`: Call Ollama `/api/generate`, get continuation, add child
- `rollout(node, ...)`: Keep expanding until "ANSWER:" found or max depth
- `backpropagate(path, score)`: Update values up the tree
- `search(question, sims, c, endpoint, model)`: Run the loop, yielding after each simulation so the tree can re-render

Score terminal nodes: 1.0 if they contain "ANSWER:", 0.5 otherwise (simple heuristic for the sandbox).

- [ ] **Step 3: Wire sandbox to tree renderer**

After each simulation, call `renderer.render(tree)` to update the visualization. Show progress in `#sandbox-status` ("Simulation 3/10...").

Error handling:
- Network failure: skip simulation, show inline warning
- Empty response: skip simulation
- 50 node cap: stop early
- 30s per-simulation timeout: skip and continue

- [ ] **Step 4: Implement fallback message**

If the first API call fails (connection refused), show: "Connect Ollama at localhost:11434 to try this. Install: https://ollama.ai"

- [ ] **Step 5: Test with Ollama running locally**

Start Ollama, type a question, click Search. Verify tree builds in real-time.

- [ ] **Step 6: Test fallback without Ollama**

Stop Ollama, click Search. Verify fallback message appears.

- [ ] **Step 7: Commit**

```bash
cd ../explainables/mcts-reasoning && git add index.html
git commit -m "feat: live sandbox with Ollama MCTS engine"
```

---

## Chunk 7: Final Polish

### Task 15: Visual polish and responsive layout

**Files:**
- Modify: `../explainables/mcts-reasoning/index.html`

- [ ] **Step 1: Hero section polish**

- Subtitle with letter-spacing: "AN EXPLORABLE EXPLANATION"
- Gradient background on hero
- Smooth scroll-down indicator (animated chevron)

- [ ] **Step 2: Typography and spacing**

- Section headings with accent color and letter-spacing
- Narrative text line-height and max-width for readability
- Code/formula styling (monospace blocks with subtle background)
- Transition paragraphs in italic

- [ ] **Step 3: Responsive breakpoint**

Test at various widths. At <1000px total width (500px tree panel minimum), switch to stacked layout. Verify tree still renders correctly when stacked.

- [ ] **Step 4: Keyboard accessibility**

- Tab through interactive elements (slider, buttons, nodes)
- Focus indicators (outline in accent color)
- ARIA labels on tree nodes: `aria-label="Node: Assume A is a knight. Value 0.8, 3 visits"`

- [ ] **Step 5: Final end-to-end test**

Open in browser. Complete full scroll-through. Test all interactions. Verify no visual glitches, all animations smooth, all controls functional.

- [ ] **Step 6: Commit**

```bash
cd ../explainables/mcts-reasoning && git add index.html
git commit -m "feat: visual polish, responsive layout, and keyboard accessibility"
```

---

### Task 16: README

**Files:**
- Create: `../explainables/mcts-reasoning/README.md`

- [ ] **Step 1: Write README**

Brief README covering:
- What this is (explorable explanation of MCTS for LLM reasoning)
- How to view it (open `index.html` in a browser)
- How to regenerate trace data (`python3 generate_data.py`)
- How to use the live sandbox (requires Ollama)
- Link to the mcts-reasoning library

- [ ] **Step 2: Commit**

```bash
cd ../explainables/mcts-reasoning && git add README.md
git commit -m "docs: add README"
```
