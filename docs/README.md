# Documentation

This directory contains the MkDocs-based documentation for MCTS-Reasoning.

## Building the Documentation

### Install Dependencies

```bash
# Install with documentation dependencies
pip install -e ".[docs]"

# Or install mkdocs directly
pip install mkdocs mkdocs-material mkdocs-autorefs pymdown-extensions
```

### Local Development

```bash
# Serve documentation locally with auto-reload
mkdocs serve

# Open browser to http://127.0.0.1:8000
```

The documentation will automatically reload when you edit files.

### Building Static Site

```bash
# Build static HTML site
mkdocs build

# Output will be in site/ directory
# Open site/index.html in browser
```

### Deployment to GitHub Pages

```bash
# Deploy to GitHub Pages
mkdocs gh-deploy

# This will:
# 1. Build the documentation
# 2. Push to gh-pages branch
# 3. Make it available at https://yourusername.github.io/mcts-reasoning/
```

## Documentation Structure

```
docs/
├── index.md                    # Home page
├── getting-started/
│   ├── installation.md         # Installation guide
│   ├── quick-start.md          # Quick start tutorial
│   └── examples.md             # Code examples
├── guides/
│   ├── shell-guide.md          # Shell interface guide
│   ├── shell-quick-reference.md # Shell command reference
│   ├── shell-comparison.md     # Shell vs TUI comparison
│   ├── tui-guide.md            # TUI interface guide
│   └── remote-ollama-setup.md  # Remote Ollama configuration
├── features/
│   ├── compositional-actions.md # 5D action space
│   ├── prompt-features.md      # Prompt engineering
│   ├── solution-detection.md   # Auto-detection
│   ├── meta-reasoning.md       # LLM-guided actions
│   ├── reflection.md           # Self-critique
│   ├── learning.md             # Pattern learning
│   ├── context-management.md   # Summarization
│   └── tree-diagnostics.md     # Analysis tools
├── advanced/
│   ├── mcp-integration.md      # MCP tools
│   ├── benchmarking.md         # Evaluation
│   └── automatic-summarization.md # Context compression
└── development/
    ├── testing.md              # Test suite
    ├── changelog.md            # Version history
    ├── bug-fixes.md            # Recent fixes
    ├── new-features.md         # New features
    ├── command-syntax-update.md # Syntax changes
    ├── improvements-complete.md # Completed work
    ├── short-term-improvements.md # Roadmap
    └── medium-term-improvements.md # Future plans
```

## Writing Documentation

### Style Guidelines

1. **Use clear, concise language**
2. **Include code examples** for concepts
3. **Add cross-references** to related docs
4. **Use admonitions** for important notes
5. **Keep sections focused** and scannable

### Markdown Extensions

The documentation uses several Markdown extensions:

#### Code Blocks with Syntax Highlighting

````markdown
```python
from mcts_reasoning import ReasoningMCTS, get_llm

llm = get_llm()
mcts = ReasoningMCTS().with_llm(llm)
```
````

#### Admonitions

```markdown
!!! note "Important Information"
    This is a note with a title.

!!! warning
    This is a warning without a title.

!!! tip "Pro Tip"
    This is a helpful tip.
```

#### Tabs

```markdown
=== "Python"
    ```python
    print("Hello")
    ```

=== "Shell"
    ```bash
    echo "Hello"
    ```
```

#### Tables

```markdown
| Feature | Status |
|---------|--------|
| MCTS    | ✅     |
| TUI     | ✅     |
```

### Adding New Pages

1. Create the markdown file in appropriate directory
2. Add to navigation in `mkdocs.yml`:

```yaml
nav:
  - Features:
    - New Feature: features/new-feature.md
```

3. Add cross-references from related pages

### Cross-Referencing

Use relative paths for links:

```markdown
See [Compositional Actions](../features/compositional-actions.md) for details.
```

## Configuration

Documentation configuration is in `/mkdocs.yml` at the project root.

Key settings:

- **theme**: Material theme with dark/light mode
- **plugins**: Search, autorefs
- **markdown_extensions**: Code highlighting, admonitions, tabs, etc.
- **nav**: Site navigation structure

## Testing

Before committing documentation changes:

```bash
# Check for broken links
mkdocs build --strict

# Serve locally and manually test
mkdocs serve

# Check all pages render correctly
# Verify code examples are accurate
# Test cross-references work
```

## Continuous Integration

Add to `.github/workflows/docs.yml`:

```yaml
name: Build Documentation

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  build:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
        with:
          python-version: '3.10'
      - run: pip install -e ".[docs]"
      - run: mkdocs build --strict
```

## Tips

1. **Use `mkdocs serve` during writing** - See changes instantly
2. **Keep pages focused** - One topic per page
3. **Add examples liberally** - Code examples help understanding
4. **Update navigation** - Keep `mkdocs.yml` nav in sync
5. **Test code examples** - Ensure they actually work
6. **Cross-reference related topics** - Help users find information
7. **Use consistent formatting** - Follow established patterns

## Troubleshooting

### "File not found" errors

- Check file paths are correct
- Use relative paths from current file
- Verify file exists in docs/ directory

### Navigation not updating

- Check `mkdocs.yml` syntax
- Restart `mkdocs serve`
- Clear browser cache

### Broken build

```bash
# Build with verbose output
mkdocs build --verbose

# Check for syntax errors in markdown
# Verify all referenced files exist
# Check mkdocs.yml is valid YAML
```

## Resources

- [MkDocs Documentation](https://www.mkdocs.org/)
- [Material for MkDocs](https://squidfunk.github.io/mkdocs-material/)
- [PyMdown Extensions](https://facelessuser.github.io/pymdown-extensions/)
