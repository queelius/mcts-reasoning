# Compositional Prompting for LLM Reasoning

This directory contains the academic paper describing the compositional prompting system implemented in the MCTS-Reasoning codebase.

## Paper Structure

- `main.tex` - Main LaTeX document
- `references.bib` - Bibliography in BibTeX format
- `README.md` - This file

## Compiling the Paper

### Using pdflatex and bibtex

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Using latexmk (recommended)

```bash
latexmk -pdf main.tex
```

### Using the Makefile

```bash
make          # Build the PDF
make clean    # Remove auxiliary files
make distclean # Remove all generated files including PDF
```

## Paper Contents

### Abstract
Introduces the compositional prompting system combining MCTS with a 5-dimensional action space for LLM reasoning.

### 1. Introduction
Motivates the problem of systematic prompt engineering and introduces the compositional approach. Outlines key contributions.

### 2. Related Work
Reviews prior work in:
- Prompting strategies (Chain-of-Thought, Tree-of-Thoughts, Self-Consistency)
- Monte Carlo Tree Search
- Prompt engineering and meta-learning
- Tool use and external knowledge

### 3. Method
Technical details of the system:
- Compositional action space definition (ω, φ, σ, κ, τ)
- Prompt construction algorithm
- Compatibility rules for semantic coherence
- MCTS integration (selection, expansion, rollout, backpropagation)
- Smart termination detection
- Tool-aware reasoning with MCP
- Sampling and consistency checking strategies

### 4. Experimental Design
Outlines planned evaluation:
- Evaluation domains (math, logic, commonsense, programming)
- Baselines and metrics
- Ablation studies
- Preliminary observations

### 5. Discussion
Analyzes:
- Advantages (systematic exploration, interpretability, flexibility)
- Limitations (computational cost, action space size, LLM dependence)
- Design choices and rationale
- Broader implications for prompt engineering

### 6. Future Work
Proposes extensions:
- Learned compatibility rules
- Adaptive weighting
- Hierarchical reasoning
- Multi-model reasoning
- Formal verification
- Interactive reasoning

### 7. Conclusion
Summarizes contributions and impact.

## Key Contributions

1. **Compositional Action Space**: 5-dimensional framework creating 30,000+ possible action combinations
2. **MCTS Integration**: UCB1-based selection with weighted sampling and compatibility rules
3. **Smart Termination Detection**: Hybrid pattern matching + LLM-based assessment
4. **Tool-Aware Reasoning**: MCP integration with ~40% of actions encouraging tool usage
5. **Multiple Sampling Strategies**: Value-based, visit-based, diverse sampling, and consistency checking

## Target Venues

This paper is suitable for submission to:
- **NeurIPS** (Conference on Neural Information Processing Systems)
- **ICML** (International Conference on Machine Learning)
- **ICLR** (International Conference on Learning Representations)
- **ACL** (Association for Computational Linguistics)
- **AAAI** (Association for the Advancement of Artificial Intelligence)

The paper emphasizes technical rigor, clear motivation, and honest presentation of both capabilities and limitations.

## Citation

If you use this work, please cite:

```bibtex
@article{compositional-prompting-mcts,
  title={Compositional Prompting for LLM Reasoning: A Monte Carlo Tree Search Framework with Structured Action Spaces},
  author={Anonymous Authors},
  journal={Under Review},
  year={2024}
}
```

## Requirements

To compile the paper, you need a LaTeX distribution with:
- pdflatex
- bibtex
- Standard packages: amsmath, amssymb, amsthm, graphicx, booktabs, algorithm, algorithmic, natbib, hyperref, xcolor

Most modern LaTeX distributions (TeX Live, MiKTeX) include these by default.
