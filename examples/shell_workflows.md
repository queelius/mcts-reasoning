# MCTS Shell Example Workflows

Real-world examples of using the MCTS-Reasoning Shell for various tasks.

## Table of Contents

1. [Mathematics](#mathematics)
2. [Problem Solving](#problem-solving)
3. [Code Analysis](#code-analysis)
4. [Research and Analysis](#research-and-analysis)
5. [Quality Assurance](#quality-assurance)
6. [Batch Processing](#batch-processing)

---

## Mathematics

### Solving Quadratic Equations

```bash
# Simple solve
mcts> ask "Solve x^2 + 5x + 6 = 0" | search 50 | best

# With verification
mcts> ask "Solve x^2 + 5x + 6 = 0" | search 50 | best | verify

# Multiple solutions for comparison
mcts> ask "Solve x^2 + 5x + 6 = 0" | search 50 | sample 3 | format table
```

### Finding Prime Numbers

```bash
# Find primes with diverse approaches
mcts> ask "Find all prime numbers less than 100" | \
      search 100 | \
      sample 5 --strategy diverse | \
      best

# Export results
mcts> ask "Find all prime numbers less than 100" | \
      search 100 | \
      export markdown > primes_solution.md
```

### Proving Mathematical Theorems

```bash
# With RAG guidance
mcts> use rag math
mcts> ask "Prove that the square root of 2 is irrational" --rag math | \
      search 200 | \
      sample 3 | \
      verify | \
      save proof.txt

# Check consistency across proofs
mcts> ask "Prove that the square root of 2 is irrational" --rag math | \
      search 200 | \
      consistency 20
```

### Calculus Problems

```bash
# Derivative
mcts> ask "Find the derivative of x^3 * sin(x)" | search 50 | best

# Integral
mcts> ask "Integrate x * e^x dx" | search 100 | best | verify

# Multiple methods
mcts> ask "Integrate x * e^x dx" | \
      search 100 | \
      sample 10 --strategy diverse | \
      grep "integration by parts"
```

---

## Problem Solving

### Logic Puzzles

```bash
# Solve with multiple approaches
mcts> ask "There are 100 prisoners numbered 1-100. A room contains a cupboard with 100 drawers. Each drawer contains one prisoner's number. Prisoners enter the room one at a time, can open up to 50 drawers, and must leave with all drawers closed. If every prisoner finds their number, they all go free. What strategy maximizes their probability?" | \
      search 300 | \
      sample 5 --strategy diverse | \
      format table

# Compare different reasoning paths
mcts> ask "<same puzzle>" | search 300 | save puzzle_tree.json
mcts> load puzzle_tree.json | sample 10 | diff
```

### Optimization Problems

```bash
# Traveling Salesman (small instance)
mcts> ask "Find shortest route visiting cities: A(0,0), B(3,4), C(6,0), D(3,-4), returning to A" | \
      search 100 | \
      sample 10 | \
      filter --min-value 0.7 | \
      best

# Resource allocation
mcts> ask "Allocate 100 units among 3 projects to maximize ROI: Project A (high risk, high return), B (medium/medium), C (low risk, stable return)" | \
      search 150 | \
      sample 10 | \
      sort --by value | \
      head 3
```

### Riddles and Lateral Thinking

```bash
# Multiple interpretations
mcts> ask "A man lives on the 10th floor. Every day he takes the elevator down to the ground floor. On the way up, he takes it to the 7th floor and walks the rest. Why?" | \
      search 100 | \
      sample 10 --strategy diverse | \
      unique

# Verify creativity
mcts> ask "<riddle>" | search 100 | sample 5 | explain
```

---

## Code Analysis

### Algorithm Design

```bash
# Design sorting algorithm
mcts> ask "Design an efficient algorithm to find the kth largest element in an unsorted array" | \
      search 100 | \
      sample 3 | \
      verify

# Compare approaches
mcts> ask "Design an efficient algorithm to find the kth largest element in an unsorted array" | \
      search 150 | \
      sample 5 --strategy diverse | \
      grep "quickselect|heap|partition" -i | \
      format table
```

### Debugging Strategies

```bash
# Debug approach
mcts> ask "A binary search function returns wrong results for arrays with duplicates. Diagnose and fix." | \
      search 100 | \
      best

# Multiple solutions
mcts> ask "A binary search function returns wrong results for arrays with duplicates. Diagnose and fix." | \
      search 100 | \
      sample 5 | \
      filter --min-value 0.8
```

### Code Review

```bash
# Review for bugs
mcts> ask "Review this code for potential bugs and edge cases: [code snippet]" | \
      search 100 | \
      sample 5 --strategy diverse | \
      format table

# Security analysis
mcts> ask "Identify security vulnerabilities in this authentication code: [code]" | \
      search 150 | \
      consistency 20
```

---

## Research and Analysis

### Literature Analysis

```bash
# Summarize paper
mcts> ask "Summarize the key contributions of the 'Attention Is All You Need' paper" | \
      search 100 | \
      sample 3 | \
      best

# Compare interpretations
mcts> ask "What are the main arguments in [paper]?" | \
      search 150 | \
      sample 5 --strategy diverse | \
      diff
```

### Hypothesis Generation

```bash
# Generate hypotheses
mcts> ask "Generate plausible hypotheses to explain the observation: [data/observation]" | \
      search 200 | \
      sample 10 --strategy diverse | \
      filter --min-value 0.7 | \
      format table

# Evaluate hypotheses
mcts> ask "Evaluate this hypothesis: [hypothesis]" | \
      search 100 | \
      best | \
      verify
```

### Data Analysis Planning

```bash
# Design analysis pipeline
mcts> ask "Design a data analysis pipeline for predicting customer churn with features: age, usage_hours, support_tickets, tenure" | \
      search 150 | \
      sample 5 | \
      best | \
      export markdown > analysis_plan.md
```

---

## Quality Assurance

### Consistency Checking

```bash
# Check answer consistency
mcts> ask "What is the capital of Australia?" | \
      search 50 | \
      consistency 30

# For controversial questions
mcts> ask "What are the main causes of [controversial topic]?" | \
      search 200 | \
      consistency 50 | \
      save consistency_report.txt
```

### Cross-Validation

```bash
# Multiple sampling strategies
mcts> ask "Explain quantum entanglement" | search 100 | save quantum_tree.json

# Value-based sampling
mcts> load quantum_tree.json | sample 5 --strategy value | save value_samples.json

# Visit-based sampling
mcts> load quantum_tree.json | sample 5 --strategy visits | save visits_samples.json

# Diverse sampling
mcts> load quantum_tree.json | sample 5 --strategy diverse | save diverse_samples.json

# Compare
mcts> load value_samples.json | best
mcts> load visits_samples.json | best
mcts> load diverse_samples.json | best
```

### Verification Pipelines

```bash
# Multi-stage verification
mcts> ask "Prove the Pythagorean theorem" | \
      search 200 | \
      sample 5 | \
      verify | \
      grep "CORRECT" | \
      save verified_proofs.txt
```

---

## Batch Processing

### Process Multiple Questions

```bash
# Save base configuration
mcts> set provider openai
mcts> set model gpt-4
mcts> set exploration 1.5

# Question 1
mcts> ask "Question 1" | search 100 | best > q1_answer.txt

# Question 2
mcts> ask "Question 2" | search 100 | best > q2_answer.txt

# Question 3
mcts> ask "Question 3" | search 100 | best > q3_answer.txt
```

### Parameter Sweep

```bash
# Try different exploration constants
mcts> ask "Complex problem" | search 100 | save base_tree.json

mcts> set exploration 0.5
mcts> load base_tree.json | search 50 | best > low_explore.txt

mcts> set exploration 1.414
mcts> load base_tree.json | search 50 | best > medium_explore.txt

mcts> set exploration 2.5
mcts> load base_tree.json | search 50 | best > high_explore.txt

# Compare results
mcts> cat low_explore.txt
mcts> cat medium_explore.txt
mcts> cat high_explore.txt
```

### Comparative Analysis

```bash
# Same question, different providers
mcts> set provider openai
mcts> ask "Explain black holes" | search 100 | save openai_result.json

mcts> set provider anthropic
mcts> ask "Explain black holes" | search 100 | save anthropic_result.json

mcts> set provider ollama
mcts> ask "Explain black holes" | search 100 | save ollama_result.json

# Compare approaches
mcts> load openai_result.json | tree 3
mcts> load anthropic_result.json | tree 3
mcts> load ollama_result.json | tree 3
```

---

## Complex Workflows

### Research Workflow

```bash
# Setup
mcts> set provider openai
mcts> set model gpt-4
mcts> use rag research

# Phase 1: Initial exploration
mcts> ask "What are the current challenges in renewable energy storage?" --rag research | \
      search 200 | \
      save exploration_tree.json

# Phase 2: Generate hypotheses
mcts> load exploration_tree.json | \
      sample 10 --strategy diverse | \
      export markdown > hypotheses.md

# Phase 3: Evaluate top hypotheses
mcts> load exploration_tree.json | \
      sample 20 | \
      filter --min-value 0.8 | \
      head 5 | \
      verify | \
      save verified_hypotheses.txt

# Phase 4: Consistency check
mcts> load exploration_tree.json | \
      consistency 50 | \
      save consistency_report.txt

# Phase 5: Final report
mcts> load exploration_tree.json | \
      export markdown > final_report.md
```

### Competitive Analysis Workflow

```bash
# Setup
mcts> set provider anthropic
mcts> set model claude-3-5-sonnet-20250219

# Analyze competitor strategies
mcts> ask "Analyze competitor strategies in the EV market: Tesla, BYD, Volkswagen" | \
      search 300 | \
      save competitor_analysis.json

# Extract insights
mcts> load competitor_analysis.json | \
      sample 15 --strategy diverse | \
      filter --min-value 0.75 | \
      export csv > competitor_insights.csv

# Generate recommendations
mcts> load competitor_analysis.json | \
      sample 10 | \
      grep "recommendation|strategy|advantage" -i | \
      format table > recommendations.txt

# Consistency check on key findings
mcts> load competitor_analysis.json | \
      consistency 30 | \
      save key_findings.txt
```

### Decision-Making Workflow

```bash
# Setup
mcts> set provider openai
mcts> set exploration 2.0  # High exploration for diverse options

# Phase 1: Generate options
mcts> ask "Should we invest in Option A (high risk, high reward) or Option B (stable, moderate returns)? Consider: market conditions, timeline, risk tolerance." | \
      search 200 | \
      save decision_tree.json

# Phase 2: Explore different perspectives
mcts> load decision_tree.json | \
      sample 20 --strategy diverse | \
      save diverse_perspectives.json

# Phase 3: Filter by quality
mcts> load diverse_perspectives.json | \
      filter --min-value 0.7 | \
      sort --by value | \
      export table > ranked_options.txt

# Phase 4: Verify top recommendations
mcts> load decision_tree.json | \
      sample 5 | \
      verify | \
      save verified_recommendations.txt

# Phase 5: Check consensus
mcts> load decision_tree.json | \
      consistency 40 | \
      save consensus.txt

# Phase 6: Final decision report
mcts> load decision_tree.json | \
      export markdown > decision_report.md
```

---

## Tips for Production Use

### 1. Save Expensive Searches

```bash
# Do expensive search once
mcts> ask "complex question" | search 500 | save big_search.json

# Reuse for different analyses
mcts> load big_search.json | sample 10
mcts> load big_search.json | consistency 30
mcts> load big_search.json | tree 5
```

### 2. Progressive Refinement

```bash
# Start small
mcts> ask "problem" | search 10 | best

# If promising, scale up
mcts> ask "problem" | search 50 | sample 5 | stats

# Final deep search
mcts> ask "problem" | search 200 | consistency 30 | save final.json
```

### 3. Quality Gates

```bash
# Only keep high-quality results
mcts> ask "question" | \
      search 100 | \
      sample 20 | \
      filter --min-value 0.8 --min-visits 5 | \
      verify | \
      grep "CORRECT"
```

### 4. Automated Reporting

```bash
# Generate comprehensive report
mcts> ask "question" | search 200 | save analysis.json
mcts> load analysis.json | stats > stats.txt
mcts> load analysis.json | tree 5 > tree.txt
mcts> load analysis.json | best | verify > verification.txt
mcts> load analysis.json | export markdown > report.md

# Combine into final report
# (using external tools)
```

### 5. Version Control

```bash
# Tag results with timestamps
mcts> ask "problem" | search 100 | save results_$(date +%Y%m%d_%H%M%S).json

# Keep history
mcts> ask "problem" | search 100 | save results_v1.json
mcts> ask "refined problem" | search 150 | save results_v2.json
```

---

## Integration with External Tools

### With Graphviz

```bash
# Generate visualization
mcts> ask "problem" | search 100 | export dot > tree.dot
$ dot -Tpng tree.dot > tree.png
$ dot -Tsvg tree.dot > tree.svg
```

### With Pandoc

```bash
# Generate PDF report
mcts> ask "problem" | search 100 | export markdown > report.md
$ pandoc report.md -o report.pdf
```

### With Spreadsheet Tools

```bash
# Export for Excel/Google Sheets
mcts> ask "problem" | search 100 | sample 50 | export csv > results.csv
# Import into Excel for pivot tables, charts, etc.
```

### With Version Control

```bash
# Track reasoning trees in git
mcts> ask "problem" | search 100 | save tree.json
$ git add tree.json
$ git commit -m "Initial reasoning tree"

# Update and track changes
mcts> load tree.json | search 50 | save tree.json
$ git diff tree.json
$ git commit -am "Refined with 50 more simulations"
```

---

**Explore, compose, and automate your reasoning workflows!** 🚀
