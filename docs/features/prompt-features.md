# Enhanced Prompt Features

The TUI now uses `prompt_toolkit` for a professional command-line experience similar to IPython.

## Features

### 1. **Command History**

All your commands are automatically saved to `~/.mcts-reasoning/history` and persist across sessions.

**Navigation:**
- **↑ (Up Arrow)** - Previous command
- **↓ (Down Arrow)** - Next command
- **Ctrl+R** - Search command history (reverse search)
- **Ctrl+S** - Forward search (if enabled)

**Example:**
```
> /model ollama llama3.2:latest
> /ask What is 2+2?
> /search 50

# Press ↑ to cycle through previous commands
# Press Ctrl+R and type "model" to find all model commands
```

### 2. **Tab Completion**

Press **Tab** to auto-complete commands and arguments.

**Command Completion:**
```
> /mo<TAB>
# Completes to: /model

> /mod<TAB>
# Shows options:
  /model       - Switch LLM or show current
  /models      - List available models
  /model-info  - Show model information
```

**Provider Completion:**
```
> /model o<TAB>
# Completes to: /model ollama

> /model <TAB>
# Shows all providers:
  openai     - OpenAI LLM provider
  anthropic  - Anthropic LLM provider
  ollama     - Ollama LLM provider
  mock       - Mock LLM provider
```

### 3. **Syntax Highlighting**

Commands are highlighted as you type:
- **Commands** (`/ask`, `/model`, etc.) - Cyan
- **Arguments** - White

### 4. **Smart Completion**

The completer understands context:
- After `/model`, suggests provider names
- Shows descriptions next to completions
- Complete-while-typing for instant feedback

### 5. **Keyboard Shortcuts**

**Emacs-style (default):**
- **Ctrl+A** - Beginning of line
- **Ctrl+E** - End of line
- **Ctrl+K** - Kill (delete) to end of line
- **Ctrl+U** - Kill to beginning of line
- **Ctrl+W** - Delete word backwards
- **Alt+D** - Delete word forwards
- **Ctrl+L** - Clear screen
- **Ctrl+D** - Exit (if line is empty)
- **Ctrl+C** - Cancel current line

**History:**
- **Ctrl+R** - Reverse search history
- **Ctrl+P** / **↑** - Previous command
- **Ctrl+N** / **↓** - Next command

**Completion:**
- **Tab** - Trigger completion
- **↑/↓** - Navigate completions
- **Enter** - Accept completion
- **Esc** - Cancel completion

## Interactive Demo

Try the test script to explore features:

```bash
python test_prompt_features.py
```

This will show you:
1. How tab completion works
2. How to navigate history
3. How to search with Ctrl+R

## Examples

### Basic Usage

```bash
$ python mcts_tui.py

> /h<TAB>
  /help  - Show help

> /help
[Shows help message]

> /model o<TAB>llama llama3.2:latest
✓ Switched to Ollama-llama3.2:latest

> /ask What is the sum of the first 10 primes?
✓ Starting reasoning session...

# Press ↑ to see previous /ask command
# Press Ctrl+R and type "model" to find model switch command
```

### Advanced History Search

```bash
> /model ollama llama3.2:latest
> /ask Question 1
> /search 50
> /model ollama mistral:latest
> /ask Question 2
> /search 30

# Now press Ctrl+R and type "llama"
# You'll see: /model ollama llama3.2:latest
# Press Ctrl+R again to cycle through all "llama" commands

# Type Ctrl+R and "ask"
# You'll see all /ask commands in reverse order
```

### Multi-Session History

```bash
# Session 1
$ python mcts_tui.py
> /model ollama llama3.2:latest
> /ask What are prime numbers?
> /exit

# Session 2 (later)
$ python mcts_tui.py
> <Press ↑>
# Shows: /ask What are prime numbers?
> <Press ↑>
# Shows: /model ollama llama3.2:latest

# Your history persists!
```

## Configuration

History is stored in:
```
~/.mcts-reasoning/history
```

You can:
- **View history:** `cat ~/.mcts-reasoning/history`
- **Clear history:** `rm ~/.mcts-reasoning/history`
- **Backup history:** `cp ~/.mcts-reasoning/history ~/.mcts-reasoning/history.backup`

## Tips

1. **Use Tab liberally** - It's faster than typing full commands

2. **Search, don't scroll** - Ctrl+R is faster than pressing ↑ many times

3. **Learn key shortcuts** - Ctrl+A, Ctrl+E, Ctrl+K are very handy

4. **History is your friend** - Don't retype long commands, use ↑ and edit

5. **Complete providers** - Type `/model <Tab>` to see all available providers

## Comparison with Basic Input

### Before (basic input)
```
> /model ollama llama3.2:latest   # Type everything manually
> /ask What is...                  # No history, start from scratch
> /mdoel ...                       # Typo, have to retype
```

### After (enhanced prompt)
```
> /mo<Tab>del ollama<Tab> llama3.2:latest   # Tab complete
> /ask What is...                            # Type normally
> <Press ↑>                                  # Get previous /ask
> /m<Tab>                                    # Tab catches typos
```

## Troubleshooting

**Completions not showing:**
- Make sure you're pressing Tab
- Start typing the command (e.g., `/m` then Tab)

**History not working:**
- Check that `~/.mcts-reasoning/` directory exists
- Verify `~/.mcts-reasoning/history` has write permissions

**Weird characters appearing:**
- Your terminal may not support prompt_toolkit fully
- Try a different terminal (iTerm2, Windows Terminal, modern gnome-terminal)

**Ctrl+R not working:**
- This is reverse search, it's enabled by default
- Press Ctrl+R, then start typing to search
- Press Ctrl+R again to cycle through matches

## Technical Details

The enhanced prompt uses [prompt_toolkit](https://python-prompt-toolkit.readthedocs.io/), the same library powering:
- **IPython** - Interactive Python shell
- **ptpython** - Better Python REPL
- **pgcli** - Postgres CLI with auto-completion

Features provided:
- FileHistory for persistent history
- Custom Completer for command/argument completion
- Custom Lexer for syntax highlighting
- Configurable key bindings (Emacs-mode default)
- Complete-while-typing
- History search with Ctrl+R

## Future Enhancements

Potential additions:
- Vi mode support (with command to toggle)
- Custom key bindings configuration
- Multi-line input for long questions
- Command aliases
- Smart completion based on current context (e.g., suggest models from /models output)
- Fuzzy completion
- Command snippets/templates
