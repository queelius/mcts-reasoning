# Command Syntax Update - Slash Optional

## Summary

Commands in the TUI no longer require a leading slash. Both forms work:

```bash
# New style (cleaner)
ask What are the prime numbers less than 20?
search 50
nodes

# Old style (still works for backward compatibility)
/ask What are the prime numbers less than 20?
/search 50
/nodes
```

## Rationale

- **Cleaner interface**: Removing the slash requirement makes the TUI feel more like a standard CLI
- **Easier to type**: One less character per command
- **Familiar**: Most CLIs don't require slash prefixes
- **Backward compatible**: Existing scripts/habits with slashes still work

## What Changed

### Code Changes

**File:** `mcts_reasoning/tui/commands.py`

**Before:**
```python
if not line.startswith('/'):
    return None  # Reject non-slash commands

line = line[1:]  # Remove required slash
```

**After:**
```python
# Remove leading slash if present (optional)
if line.startswith('/'):
    line = line[1:]
# Continue parsing...
```

### Documentation Updates

All documentation updated to show commands without slashes:

- **Help text**: `help` command now shows clean syntax
- **CLAUDE.md**: Updated examples to remove slashes
- **TREE_DIAGNOSTICS.md**: All examples updated
- **CONTEXT_MANAGEMENT.md**: All examples updated
- **NEW_FEATURES.md**: All examples updated
- **BUG_FIXES.md**: All examples updated

## Examples

### Before
```bash
/model ollama llama3.2 base_url=http://192.168.0.225:11434
/ask What is x^3 - 3x^2 + x - 10 = 0?
/search 50
/nodes
/inspect 5
/show-prompt 5
/export-tree analysis.json
```

### After (preferred)
```bash
model ollama llama3.2 base_url=http://192.168.0.225:11434
ask What is x^3 - 3x^2 + x - 10 = 0?
search 50
nodes
inspect 5
show-prompt 5
export-tree analysis.json
```

### Both Work!
```bash
# Mix and match if you want
ask What is 2+2?        # No slash
/search 20              # With slash (still works)
nodes                   # No slash
/inspect 5              # With slash (still works)
```

## Migration Guide

**For Users:**
- You can start using commands without slashes immediately
- Your muscle memory with slashes will still work
- No action required - this is fully backward compatible

**For Scripts:**
- Existing scripts that use `/command` syntax will continue to work
- You can update scripts to remove slashes for cleaner look (optional)

**For Documentation:**
- Examples in docs now show slash-free syntax
- Mention "slash optional for backward compatibility" where relevant

## Testing

All command parsing tested with both syntaxes:

```bash
python test_command_parsing.py
# ✅ All 14 tests passed!
# Both slash and non-slash commands work correctly.
```

Test coverage includes:
- Simple commands: `ask`, `search`, `nodes`
- Commands with args: `model ollama llama3.2`
- Commands with URLs: `probe http://...`
- Commands with files: `export-tree file.json`
- All tested with and without leading slash

## Impact

**Positive:**
- ✅ Cleaner, more intuitive interface
- ✅ Faster to type
- ✅ More consistent with standard CLI tools
- ✅ Fully backward compatible

**Neutral:**
- Documentation now shows preferred syntax (no slash)
- Old `/command` syntax still documented as "optional for backward compatibility"

**None:**
- No breaking changes
- No migration required
- No performance impact

## Related Commands

All commands support optional slash:

**Session Management:**
- `ask`, `search`, `continue`, `solution`, `save`, `load`, `status`

**Configuration:**
- `model`, `models`, `model-info`, `probe`, `temperature`, `exploration`

**Analysis:**
- `tree`, `sample`, `consistency`

**Tree Diagnostics:**
- `nodes`, `inspect`, `inspect-full`, `show-prompt`, `path`, `export-tree`

**MCP:**
- `mcp-enable`, `mcp-connect`, `mcp-list`, `mcp-tools`

**Other:**
- `help`, `exit`

## Future Considerations

Potential enhancements:
- Command aliases (e.g., `s` for `search`, `i` for `inspect`)
- Tab completion for commands
- Command history with up/down arrows
- Command suggestions for typos

---

**Recommendation:** Use the cleaner syntax without slashes. It's simpler and more intuitive!
