# MCTS-Reasoning Shell Guide

A comprehensive guide to the Unix-style composable shell for MCTS reasoning.

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Command Reference](#command-reference)
5. [Piping and Composition](#piping-and-composition)
6. [I/O Redirection](#io-redirection)
7. [Example Workflows](#example-workflows)
8. [Advanced Usage](#advanced-usage)

---

## Introduction

The MCTS-Reasoning Shell is a Unix-style command-line interface for LLM-based reasoning using Monte Carlo Tree Search. It embraces the Unix philosophy:

- **Each command does one thing well**
- **Commands are composable via pipes**
- **Text/JSON streams flow between commands**
- **Scriptable and automatable**

Think of it as `bash` for reasoning - pipe commands together to build complex reasoning workflows.

---

## Quick Start

### Installation

```bash
# Install with shell support
pip install -e ".[tui]"

# Run the shell
python mcts_shell.py

# Or after installation
mcts-shell
```

### Your First Command

```bash
mcts> ask "What are the prime numbers less than 20?"
```

This creates a reasoning task. Now search for solutions:

```bash
mcts> ask "What are the prime numbers less than 20?" | search 50
```

Get the best answer:

```bash
mcts> ask "What are the prime numbers less than 20?" | search 50 | best
```

---

## Core Concepts

### Streams

Data flows through the shell as **streams**. There are several stream types:

1. **MCTSStream** - Contains an MCTS tree
2. **PathStream** - Contains sampled paths from a tree
3. **SolutionStream** - Contains final solutions/answers
4. **TextStream** - Plain text output
5. **StatsStream** - Statistics and analysis data

Commands consume and produce streams, enabling composition.

### Commands as Filters

Each command is a **filter** that transforms streams:

```
ask → MCTSStream
search → MCTSStream → MCTSStream
sample → MCTSStream → PathStream
best → PathStream → SolutionStream
format → Stream → TextStream
```

### Piping

Connect commands with `|` to create pipelines:

```bash
command1 | command2 | command3
```

Data flows left to right through the pipeline.

---

## Command Reference

### Core Reasoning Commands

#### `ask <question>`

Create a new reasoning task.

```bash
ask "Solve x^2 + 5x + 6 = 0"
ask "Find the sum of squares of first 10 primes" --rag math
```

**Options:**
- `--rag <name>` - Use RAG guidance
- `--compositional false` - Disable compositional actions

**Output:** MCTSStream

---

#### `search <N>`

Run N MCTS simulations to explore reasoning paths.

```bash
search 100        # Run 100 simulations
search 50         # Run 50 simulations
```

**Input:** MCTSStream
**Output:** MCTSStream (updated with search results)

---

#### `explore <depth>`

Explore tree to a specific depth.

```bash
explore 10        # Explore to depth 10
explore 5         # Explore to depth 5
```

**Input:** MCTSStream
**Output:** MCTSStream

---

#### `sample <N> [options]`

Sample reasoning paths from the tree.

```bash
sample 5                           # Sample 5 paths by value
sample 10 --strategy visits        # Sample by visit count
sample 3 --strategy diverse        # Sample diverse paths
sample 5 --temperature 0.5         # Control randomness
```

**Options:**
- `--strategy` - value (default), visits, diverse
- `--temperature` - 0.0 (greedy) to high (random)

**Input:** MCTSStream
**Output:** PathStream

---

#### `best`

Get the best solution from tree or paths.

```bash
best              # Get highest-value solution
```

**Input:** MCTSStream or PathStream
**Output:** SolutionStream

---

### Filter Commands

#### `filter [options]`

Filter paths by criteria.

```bash
filter --min-value 0.8             # Keep paths with value >= 0.8
filter --min-visits 10             # Keep paths with visits >= 10
filter --max-length 8              # Keep paths with length <= 8
```

**Options:**
- `--min-value`, `--max-value` - Filter by average value
- `--min-visits`, `--max-visits` - Filter by visit count
- `--min-length`, `--max-length` - Filter by path length

**Input:** PathStream
**Output:** PathStream

---

#### `sort [options]`

Sort paths by criterion.

```bash
sort                               # Sort by value (descending)
sort --by visits                   # Sort by visits
sort --by length                   # Sort by length
sort --reverse false               # Sort ascending
```

**Input:** PathStream or TextStream
**Output:** PathStream or TextStream

---

#### `head <N>`, `tail <N>`

Take first or last N items.

```bash
head 10           # First 10 items
tail 5            # Last 5 items
```

**Input:** Any stream
**Output:** Same stream type with subset

---

#### `grep <pattern> [options]`

Search for pattern in stream content.

```bash
grep "prime"                       # Find items containing "prime"
grep "\\d+" -i                     # Case-insensitive regex search
```

**Options:**
- `-i` - Case-insensitive

**Input:** PathStream, SolutionStream, or TextStream
**Output:** Same stream type (filtered)

---

#### `unique`

Remove duplicate items.

```bash
unique            # Remove duplicates
```

**Input:** SolutionStream or TextStream
**Output:** Same stream type (deduplicated)

---

#### `count`

Count items in stream.

```bash
count             # Count items
```

**Input:** Any stream
**Output:** TextStream with count

---

### I/O Commands

#### `load <filepath>`

Load MCTS tree or data from file.

```bash
load session.json                  # Load tree
load results.json                  # Load any JSON data
```

**Output:** Appropriate stream type (auto-detected from file)

---

#### `save <filepath> [options]`

Save stream to file.

```bash
save result.json                   # Save as JSON
save solution.txt --format text    # Save as text
```

**Options:**
- `--format` - json (default), text

**Input:** Any stream
**Output:** Same stream (pass-through)

---

#### `export <format>`

Export stream in various formats.

```bash
export markdown                    # Markdown report
export json                        # JSON data
export dot                         # Graphviz DOT format
export csv                         # CSV spreadsheet
```

**Formats:**
- `markdown`, `md` - Markdown report
- `json` - JSON data
- `text`, `txt` - Plain text
- `dot` - Graphviz DOT (for trees)
- `csv` - CSV spreadsheet (for paths)

**Input:** Any stream
**Output:** TextStream

---

#### `format <type>`

Format stream for display.

```bash
format text                        # Plain text
format json                        # JSON format
format table                       # ASCII table (for paths)
format tree                        # ASCII tree (for MCTS)
format solution                    # Just the solution text
```

**Input:** Any stream
**Output:** TextStream

---

#### `cat <file>`

Display file contents.

```bash
cat session.json                   # Display file
cat                                # Pass through input
```

---

### Analysis Commands

#### `stats`

Show statistics about the stream.

```bash
stats             # Show detailed statistics
```

**Input:** Any stream
**Output:** StatsStream

---

#### `tree [max_depth]`

Display MCTS tree structure.

```bash
tree              # Show full tree
tree 3            # Show tree to depth 3
```

**Input:** MCTSStream
**Output:** TextStream (ASCII tree)

---

#### `verify`

Verify solution correctness using LLM.

```bash
verify            # Verify solutions
```

**Input:** SolutionStream, PathStream, or MCTSStream
**Output:** TextStream (verification results)

---

#### `consistency <N> [options]`

Check solution consistency across multiple samples.

```bash
consistency 20                     # Check across 20 samples
consistency 50 --temperature 0.8   # With custom temperature
```

**Options:**
- `--temperature` - Sampling temperature

**Input:** MCTSStream
**Output:** TextStream (consistency report)

---

#### `diff [options]`

Compare paths or solutions.

```bash
diff                               # Compare first two
diff --path1 0 --path2 2           # Compare specific paths
```

**Input:** PathStream or SolutionStream
**Output:** TextStream (comparison)

---

#### `explain`

Explain the reasoning process.

```bash
explain           # Show reasoning explanation
```

**Input:** Any stream
**Output:** TextStream

---

### Configuration Commands

#### `set <key> <value>`

Set configuration variable.

```bash
set provider openai                # Set LLM provider
set model gpt-4                    # Set model
set exploration 2.0                # Set exploration constant
set temperature 0.7                # Set temperature
```

**Keys:**
- `provider` - openai, anthropic, ollama, mock
- `model` - Model name
- `exploration` - MCTS exploration constant (default: 1.414)
- `temperature` - Sampling temperature (default: 1.0)

---

#### `get [key]`

Get configuration variable.

```bash
get provider                       # Get specific value
get                                # Get all config
```

---

#### `use <resource>`

Load resources like RAG stores.

```bash
use rag math                       # Load math RAG store
use rag code                       # Load code RAG store
```

---

### Utility Commands

#### `help [command]`

Show help for commands.

```bash
help              # Show all commands
help ask          # Help for specific command
```

---

#### `echo <text>`

Print text to output.

```bash
echo "Hello"                       # Print text
echo $variable                     # Print variable (future)
```

---

#### `exit`, `quit`, `q`

Exit the shell.

```bash
exit
quit
q
```

---

## Piping and Composition

### Basic Piping

Connect commands with `|`:

```bash
ask "problem" | search 100 | best
```

### Multi-Stage Pipelines

Build complex workflows:

```bash
ask "Find primes < 100" | search 50 | sample 10 | filter --min-value 0.8 | best
```

### Branching (via save/load)

Save intermediate results:

```bash
# Stage 1: Create and search
ask "complex problem" | search 200 | save temp.json

# Stage 2: Try different sampling strategies
load temp.json | sample 10 --strategy diverse | save diverse.json
load temp.json | sample 10 --strategy value | save value.json

# Stage 3: Compare results
load diverse.json | best
load value.json | best
```

---

## I/O Redirection

### Output Redirection

#### Overwrite (`>`)

```bash
ask "problem" | search 100 | best > solution.txt
```

#### Append (`>>`)

```bash
ask "problem" | search 50 | stats >> log.txt
```

### Input Redirection (`<`)

```bash
search 100 < tree.json
```

### Combining Pipes and Redirection

```bash
ask "problem" | search 100 > tree.json
load tree.json | sample 5 | format table > results.txt
```

---

## Example Workflows

### 1. Simple Question Answering

```bash
mcts> ask "What is the sum of prime numbers less than 20?" | search 100 | best
```

### 2. Explore and Sample Multiple Solutions

```bash
mcts> ask "Solve x^2 - 7x + 12 = 0" | search 50 | sample 5 --strategy diverse
```

### 3. Quality Control with Filtering

```bash
mcts> ask "Complex problem" | search 200 | sample 20 | filter --min-value 0.8 | head 5
```

### 4. Consistency Checking

```bash
mcts> ask "Controversial question" | search 100 | consistency 30
```

### 5. Export and Visualization

```bash
mcts> ask "Math problem" | search 100 | export markdown > report.md
mcts> ask "Math problem" | search 100 | export dot > tree.dot
# Then: dot -Tpng tree.dot > tree.png
```

### 6. Comparison Workflow

```bash
# Sample and compare different reasoning paths
mcts> ask "problem" | search 100 | sample 5 | diff
```

### 7. Batch Processing

```bash
# Save tree for reuse
mcts> ask "complex question" | search 500 | save complex_tree.json

# Try different analyses
mcts> load complex_tree.json | sample 10 | stats
mcts> load complex_tree.json | consistency 50
mcts> load complex_tree.json | sample 20 --strategy diverse | best
```

### 8. CSV Export for Analysis

```bash
mcts> ask "problem" | search 100 | sample 50 | export csv > paths.csv
# Import into Excel, pandas, etc. for further analysis
```

### 9. Verification Pipeline

```bash
mcts> ask "mathematical proof" | search 100 | sample 3 | verify
```

### 10. Full Workflow with RAG

```bash
# Set up
mcts> set provider openai
mcts> set model gpt-4
mcts> use rag math

# Reasoning with guidance
mcts> ask "Prove that sqrt(2) is irrational" --rag math | \
      search 100 | \
      sample 5 --strategy diverse | \
      verify | \
      save verified_proof.txt
```

---

## Advanced Usage

### Setting Up Provider

```bash
# Configure OpenAI
mcts> set provider openai
mcts> set model gpt-4

# Configure Anthropic
mcts> set provider anthropic
mcts> set model claude-3-5-sonnet-20250219

# Configure Ollama (local)
mcts> set provider ollama
mcts> set model llama3.2
```

### Using RAG Stores

```bash
# Load RAG store for guided reasoning
mcts> use rag math

# Ask question with RAG context
mcts> ask "Solve differential equation" --rag math | search 100
```

### Controlling Exploration

```bash
# Higher exploration = more diverse search
mcts> set exploration 2.0

# Lower exploration = more focused on best paths
mcts> set exploration 0.5
```

### Temperature Control

```bash
# Greedy sampling (deterministic)
mcts> sample 5 --temperature 0

# Random sampling
mcts> sample 5 --temperature 2.0

# Balanced (default)
mcts> sample 5 --temperature 1.0
```

### Combining with Unix Tools

```bash
# Use with standard Unix tools
mcts> ask "problem" | search 100 | best | format solution | wc -w

# Pipe to files
mcts> ask "problem" | search 100 | export markdown | pandoc -o report.pdf
```

### Script Files (Future)

```bash
# Create workflow script (workflow.mcts)
set provider openai
set model gpt-4
use rag math

ask "$1" --rag math | \
  search 100 | \
  sample 10 --strategy diverse | \
  filter --min-value 0.8 | \
  consistency 20 | \
  save results/$1.json

# Run script
mcts> source workflow.mcts "complex problem"
```

---

## Tips and Best Practices

### 1. Start Small, Then Scale

```bash
# Test with small search first
ask "problem" | search 10 | best

# If good, scale up
ask "problem" | search 100 | best
```

### 2. Save Intermediate Results

```bash
# Expensive search - save it
ask "complex problem" | search 500 | save big_tree.json

# Reuse for different analyses
load big_tree.json | sample 10
load big_tree.json | consistency 50
load big_tree.json | tree 5
```

### 3. Use Filtering for Quality

```bash
# Generate many samples, filter for quality
sample 100 | filter --min-value 0.8 | head 10
```

### 4. Verify Important Results

```bash
# For critical questions, verify
ask "important question" | search 200 | best | verify
```

### 5. Check Consistency for Controversial Topics

```bash
# When answers might vary
ask "controversial question" | search 100 | consistency 50
```

### 6. Export for Documentation

```bash
# Create reports
export markdown > report.md
export dot > diagram.dot
```

---

## Troubleshooting

### "Unknown command" Error

Make sure you're typing the command name correctly. Use `help` to see all available commands.

### "Command requires input" Error

Some commands need input from a pipe or file:

```bash
# Wrong
mcts> search 100

# Right
mcts> ask "problem" | search 100
mcts> load tree.json | search 100
```

### No LLM Provider Configured

Set up a provider first:

```bash
mcts> set provider openai
mcts> set model gpt-4
```

### Empty Results

Make sure you've run search before sampling:

```bash
# Wrong
mcts> ask "problem" | sample 5

# Right
mcts> ask "problem" | search 100 | sample 5
```

---

## Comparison: Shell vs TUI

| Feature | Shell | TUI |
|---------|-------|-----|
| **Piping** | ✅ Full support | ❌ No piping |
| **Composability** | ✅ Highly composable | ⚠️ Limited |
| **Automation** | ✅ Scriptable | ❌ Interactive only |
| **Learning Curve** | Medium | Easy |
| **Power User** | ✅✅✅ | ⚠️ |
| **Beginner Friendly** | ⚠️ | ✅✅✅ |
| **Rich Formatting** | ⚠️ Basic | ✅ Beautiful |
| **Visual Tree** | ⚠️ ASCII | ✅ Rich |

**Recommendation:**
- **Beginners**: Start with TUI for interactive exploration
- **Power Users**: Use Shell for complex workflows and automation
- **Best of Both**: Use TUI for exploration, Shell for production workflows

---

## Future Features

### Planned Enhancements

1. **Parallel Execution**
   ```bash
   (search 50 &); (search 50 &); wait | merge
   ```

2. **Variables**
   ```bash
   result=$(ask "problem" | search 100)
   echo $result | verify
   ```

3. **Script Files**
   ```bash
   source workflow.mcts
   run analyze.mcts --input data.json
   ```

4. **Background Jobs**
   ```bash
   ask "problem" | search 1000 &
   jobs
   fg 1
   ```

5. **Plugin Commands**
   - Custom command registration
   - Third-party extensions

---

## Getting Help

- **In-shell help**: `help` or `help <command>`
- **Documentation**: See `docs/` directory
- **Examples**: See `examples/shell_workflows.md`
- **Issues**: GitHub issues

---

**Happy Reasoning!** 🚀
