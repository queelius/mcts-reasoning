# MCTS Shell Quick Reference

A one-page reference for the MCTS-Reasoning Shell.

## Getting Started

```bash
# Run the shell
python mcts_shell.py

# Or after installation
mcts-shell
```

## Basic Syntax

```bash
# Simple command
command arg1 arg2

# With flags
command arg1 --flag value --flag2

# Piping
command1 | command2 | command3

# Output redirection
command > file.txt       # Overwrite
command >> file.txt      # Append

# Input redirection
command < file.json
```

## Core Workflow

```bash
# 1. Ask a question
ask "Your question here"

# 2. Search for solutions
ask "question" | search 100

# 3. Get the best answer
ask "question" | search 100 | best

# Complete workflow
ask "What are prime numbers < 20?" | search 100 | best | verify
```

## Essential Commands

### Reasoning
- `ask <question>` - Start reasoning task
- `search <N>` - Run N MCTS simulations
- `sample <N>` - Sample N paths
- `best` - Get best solution

### Filtering
- `filter --min-value 0.8` - Filter by quality
- `sort --by value` - Sort paths
- `head <N>`, `tail <N>` - Take first/last N
- `grep <pattern>` - Search content

### I/O
- `load <file>` - Load tree from file
- `save <file>` - Save to file
- `export <format>` - Export (markdown, json, dot, csv)
- `format <type>` - Format for display

### Analysis
- `stats` - Show statistics
- `tree [depth]` - Display tree
- `verify` - Verify correctness
- `consistency <N>` - Check consistency

### Configuration
- `set <key> <value>` - Set config
- `get [key]` - Get config
- `use rag <name>` - Load RAG store

### Utilities
- `help [command]` - Show help
- `exit` - Quit shell

## Common Patterns

### Simple Question Answering
```bash
ask "question" | search 100 | best
```

### Quality Control
```bash
ask "question" | search 200 | sample 20 | filter --min-value 0.8 | best
```

### Consistency Checking
```bash
ask "question" | search 100 | consistency 30
```

### Export for Documentation
```bash
ask "question" | search 100 | export markdown > report.md
```

### Batch Analysis
```bash
# Save expensive search
ask "complex question" | search 500 | save tree.json

# Reuse for different analyses
load tree.json | sample 10
load tree.json | consistency 50
load tree.json | tree 5
```

### Comparison Workflow
```bash
ask "question" | search 100 | sample 5 --strategy diverse | diff
```

## Command Flags

### `ask`
- `--rag <name>` - Use RAG guidance
- `--compositional false` - Disable compositional

### `search`
- Takes simulation count as argument

### `sample`
- `--strategy value|visits|diverse`
- `--temperature <float>` - Control randomness

### `filter`
- `--min-value`, `--max-value`
- `--min-visits`, `--max-visits`
- `--min-length`, `--max-length`

### `sort`
- `--by value|visits|length`
- `--reverse true|false`

### `export`
- `markdown`, `json`, `dot`, `csv`

### `format`
- `text`, `json`, `table`, `tree`, `solution`

## Configuration

```bash
# Set LLM provider
set provider openai
set model gpt-4

# Set MCTS parameters
set exploration 1.5
set temperature 0.7

# View configuration
get
get provider
```

## Tips

1. **Save Intermediate Results**
   ```bash
   ask "question" | search 500 | save big_tree.json
   load big_tree.json | sample 10
   ```

2. **Progressive Refinement**
   ```bash
   ask "question" | search 10 | best        # Quick test
   ask "question" | search 100 | best       # If promising
   ask "question" | search 500 | best       # Final
   ```

3. **Quality Gates**
   ```bash
   sample 100 | filter --min-value 0.8 | verify | grep "CORRECT"
   ```

4. **Combine with Unix Tools**
   ```bash
   best | format solution | wc -w
   export markdown | pandoc -o report.pdf
   ```

## Examples

### Mathematics
```bash
ask "Solve x^2 + 5x + 6 = 0" | search 50 | best
ask "Find primes < 100" | search 100 | sample 5 | format table
```

### Problem Solving
```bash
ask "Logic puzzle: ..." | search 200 | sample 10 --strategy diverse
```

### Code Analysis
```bash
ask "Debug this code: ..." | search 100 | best | verify
```

### Research
```bash
ask "Summarize paper: ..." | search 150 | consistency 30
```

## Keyboard Shortcuts

- `Ctrl+C` - Cancel current line
- `Ctrl+D` - Exit shell
- `Ctrl+R` - Search command history
- `Tab` - Command completion (if prompt_toolkit installed)

## Getting Help

```bash
help                    # List all commands
help <command>          # Help for specific command
```

---

**Full documentation:** `docs/SHELL_GUIDE.md`
**Example workflows:** `examples/shell_workflows.md`
