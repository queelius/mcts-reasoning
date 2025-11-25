# Non-Interactive CLI Guide (mcts)

The `mcts` command provides a non-interactive command-line interface for MCTS-Reasoning, perfect for scripting, automation, and integration with other tools.

## Overview

Unlike the interactive shell (`mcts-shell`), the `mcts` command:

- Executes a single command and exits
- Perfect for shell scripts and automation
- Can work with saved sessions
- Outputs results to stdout for piping
- Supports all major MCTS-Reasoning features

## Installation

```bash
# Install with pip
pip install -e ".[tui]"

# The 'mcts' command will be available
mcts --help
```

## Basic Usage

### Starting a New Session

```bash
# Ask a question and run search
mcts ask "What is the sum of primes less than 100?" --search 50

# The output will show the best solution found
```

### Working with Sessions

Most CLI commands require a saved session file:

```bash
# In interactive shell, save a session:
# > /save my_session.json

# Then use it with CLI:
mcts solution --session my_session.json
mcts verify --session my_session.json
mcts export json output.json --session my_session.json
```

## Commands

### Reasoning Commands

**ask** - Start new reasoning session
```bash
mcts ask "Your question here" [--search N]
```

**search** - Run MCTS simulations
```bash
mcts search 100 --session my_session.json
```

**solution** - Show best solution
```bash
mcts solution --session my_session.json
```

### Analysis Commands

**sample** - Sample diverse paths
```bash
# Sample 5 paths
mcts sample 5 --session my_session.json

# Sample 10 paths
mcts sample 10 --session my_session.json
```

**consistency** - Check solution consistency
```bash
# Default trials
mcts consistency --session my_session.json

# Specify trials
mcts consistency 20 --session my_session.json
```

**verify** - Verify solution correctness using LLM
```bash
# Verify current solution
mcts verify --session my_session.json

# Verify specific node
mcts verify 5 --session my_session.json
```

### Export Commands

**export** - Export tree in various formats
```bash
# Export as JSON
mcts export json output.json --session my_session.json

# Export as Markdown report
mcts export markdown report.md --session my_session.json

# Export as Graphviz DOT
mcts export dot graph.dot --session my_session.json

# Export as CSV
mcts export csv data.csv --session my_session.json
```

### Tree Inspection

**tree** - Visualize search tree
```bash
mcts tree --session my_session.json
```

**nodes** - List all nodes
```bash
mcts nodes --session my_session.json
```

**inspect** - Inspect specific node (preview)
```bash
mcts inspect 5 --session my_session.json
```

**inspect-full** - Inspect node (full state)
```bash
mcts inspect-full 5 --session my_session.json
```

**path** - Show reasoning path to node
```bash
mcts path 5 --session my_session.json
```

**compare** - Compare multiple nodes
```bash
mcts compare 5 12 18 --session my_session.json
```

### Session Management

**save** - Save current session
```bash
mcts save my_session.json --session current.json
```

**load** - Load saved session
```bash
mcts load my_session.json
```

**status** - Show session status
```bash
mcts status --session my_session.json
```

**stats** - Show session statistics
```bash
mcts stats --session my_session.json
```

**solutions** - List finalized solutions
```bash
mcts solutions --session my_session.json
```

### Configuration

**model** - Configure LLM model
```bash
# Show current model
mcts model --session my_session.json

# Switch model
mcts model openai gpt-4 --session my_session.json
```

**temperature** - Set temperature
```bash
mcts temperature 0.7 --session my_session.json
```

**exploration** - Set exploration constant
```bash
mcts exploration 1.414 --session my_session.json
```

## Global Options

**--session, -s** - Specify session file
```bash
mcts solution -s my_session.json
```

**--provider** - Set LLM provider
```bash
mcts ask "Question" --provider openai
```

**--verbose, -v** - Verbose output
```bash
mcts verify --session my_session.json --verbose
```

## Examples

### Complete Workflow

```bash
# 1. Ask a question and search
mcts ask "Find all prime numbers less than 100" --search 50 > initial.txt

# This creates a default session, but for subsequent commands we need to save it
# Use interactive shell for this:
mcts-shell
> /ask Find all prime numbers less than 100
> /search 50
> /save primes_session.json
> /exit

# 2. Verify the solution
mcts verify --session primes_session.json

# 3. Export in different formats
mcts export json primes_tree.json --session primes_session.json
mcts export markdown primes_report.md --session primes_session.json

# 4. Check consistency
mcts consistency 20 --session primes_session.json

# 5. Sample diverse paths
mcts sample 10 --session primes_session.json
```

### Automation Script

```bash
#!/bin/bash
# Batch process multiple questions

questions=(
    "What is 2+2?"
    "List prime numbers less than 20"
    "Solve x^2 = 16"
)

for i in "${!questions[@]}"; do
    echo "Processing question $i: ${questions[$i]}"

    # Ask and search
    mcts ask "${questions[$i]}" --search 50 > "result_$i.txt"

    # Verify
    mcts verify --session "session_$i.json" >> "result_$i.txt"
done
```

### Integration with Other Tools

```bash
# Export to JSON and process with jq
mcts export json tree.json --session my_session.json
cat tree.json | jq '.nodes[] | select(.visits > 10)'

# Generate visualization with Graphviz
mcts export dot graph.dot --session my_session.json
dot -Tpng graph.dot -o graph.png

# Extract data with CSV
mcts export csv data.csv --session my_session.json
# Process with spreadsheet or pandas
```

## Comparison with Interactive Shell

| Feature | mcts (CLI) | mcts-shell (Interactive) |
|---------|------------|--------------------------|
| Command execution | Single command | Multiple commands in REPL |
| Session persistence | Requires explicit save | Automatic in session |
| Output | Stdout | Formatted in terminal |
| Best for | Scripts, automation | Interactive exploration |
| History | No | Yes, with Ctrl+R search |
| Tab completion | No | Yes |
| Rich formatting | No | Yes (optional) |

## Tips

1. **Use sessions for complex workflows**: Save your session after initial reasoning, then run analysis commands on the saved session.

2. **Combine with shell tools**: The CLI outputs to stdout, making it easy to pipe to other commands or redirect to files.

3. **Automation**: Use the CLI in CI/CD pipelines, cron jobs, or batch processing scripts.

4. **Export formats**: Choose the export format based on your needs:
   - JSON for data processing
   - Markdown for reports
   - DOT for visualization
   - CSV for spreadsheet analysis

5. **Verbose mode**: Use `--verbose` to see what commands are being executed, helpful for debugging.

## Error Handling

The CLI uses exit codes:

- `0` - Success
- `1` - Error (command failed)
- `130` - Interrupted (Ctrl+C)

Example error handling in bash:

```bash
#!/bin/bash

if mcts verify --session my_session.json; then
    echo "Verification passed"
    mcts export markdown report.md --session my_session.json
else
    echo "Verification failed"
    exit 1
fi
```

## See Also

- [Interactive Shell Guide](tui-guide.md) - For interactive usage
- [Quick Start](../getting-started/quick-start.md) - Getting started guide
- [Examples](../getting-started/examples.md) - Code examples
