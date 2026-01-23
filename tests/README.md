# MCTS-Reasoning Test Suite

This directory contains comprehensive tests for the MCTS-Reasoning project, following Test-Driven Development (TDD) principles.

## Quick Start

### Run All Tests
```bash
pytest tests/
```

### Run Specific Test Files
```bash
# Core MCTS algorithm tests (77 tests)
pytest tests/test_core.py -v

# Sampling strategies tests (53 tests)
pytest tests/test_sampling.py -v

# Compositional prompting tests (59 tests)
pytest tests/test_examples.py -v

# RAG integration tests (57 tests)
pytest tests/test_rag.py -v
```

### Run with Coverage
```bash
# Coverage for specific modules
pytest tests/test_core.py tests/test_sampling.py \
    --cov=mcts_reasoning.core \
    --cov=mcts_reasoning.sampling \
    --cov-report=term-missing

# Coverage for entire package
pytest tests/ --cov=mcts_reasoning --cov-report=html

# View HTML report
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
```

### Run Specific Tests
```bash
# Run a single test class
pytest tests/test_core.py::TestMCTSNode -v

# Run a single test method
pytest tests/test_core.py::TestMCTSNode::test_ucb1_unvisited_node_returns_infinity -v

# Run tests matching a pattern
pytest tests/ -k "ucb1" -v
```

### Run with Different Verbosity
```bash
# Quiet mode (just pass/fail counts)
pytest tests/ -q

# Verbose mode (show each test name)
pytest tests/ -v

# Very verbose (show test output)
pytest tests/ -vv
```

## Test Organization

### Unit Tests (Fast, Isolated)
- `test_core.py` - Core MCTS algorithm (97% coverage)
- `test_sampling.py` - Sampling strategies (96% coverage)
- `test_examples.py` - Few-shot learning examples
- `test_rag.py` - RAG stores and guidance
- `test_probe.py` - LLM provider endpoint probing

### Integration Tests (Component Interactions)
- `test_rag_integration.py` - RAG with MCTS integration
- `test_real_problems.py` - End-to-end problem solving
- `test_meta_reasoning.py` - Meta-reasoning capabilities
- `test_reflection.py` - Self-reflection mechanisms
- `test_learning.py` - Learning from experience

### System Tests (Full Workflows)
- `test_tui_workflow.py` - TUI interface workflows
- `test_shell.py` - Shell command parsing
- `test_benchmarking.py` - Benchmarking system

## Test Fixtures

Common fixtures are defined in `conftest.py`:

- `mock_llm` - Basic MockLLMProvider instance
- `mock_llm_with_responses` - MockLLMProvider with predefined responses
- `sample_example_dict` - Sample example data for testing

## Coverage Goals

| Module | Coverage Goal | Current |
|--------|--------------|---------|
| core.py | 95%+ | 97% |
| sampling.py | 95%+ | 96% |
| reasoning.py | 90%+ | TBD |
| compositional/actions.py | 90%+ | TBD |
| compositional/providers.py | 85%+ | TBD |
| compositional/__init__.py | 85%+ | TBD |

## Testing Best Practices

When writing new tests:

1. **Test Behavior, Not Implementation**
   ```python
   # Good: Test observable behavior
   assert mcts.best_path is not None

   # Bad: Test internal structure
   assert mcts._internal_state == expected_state
   ```

2. **One Assertion Per Test (Logical)**
   ```python
   # Good: One logical assertion
   def test_backpropagate_updates_visits(self):
       mcts._backpropagate(node, 1.0)
       assert node.visits == 1
       assert node.parent.visits == 1  # Same logical assertion

   # Bad: Multiple unrelated assertions
   def test_everything(self):
       assert mcts.root is not None
       assert mcts.llm is not None  # Different concerns
   ```

3. **Use Descriptive Names**
   ```python
   # Good
   def test_sample_by_value_returns_path_to_leaf(self):

   # Bad
   def test_sample(self):
   ```

4. **Given-When-Then Structure**
   ```python
   def test_expand_creates_child(self):
       # Given: A parent node
       parent = MCTSNode(state="parent")

       # When: We expand it
       child = mcts._expand(parent)

       # Then: A child should be created
       assert child.parent is parent
   ```

5. **Test Edge Cases**
   - Empty inputs
   - None values
   - Zero/negative numbers
   - Boundary conditions
   - Error conditions

## Common Testing Patterns

### Testing MCTS Trees
```python
def test_tree_operation(self):
    # Create test tree
    mcts = MCTS()
    mcts.root = MCTSNode(state="root", visits=10, value=5.0)
    child = MCTSNode(state="child", parent=mcts.root, visits=5, value=2.5)
    mcts.root.children = [child]

    # Perform operation and verify
    result = mcts.some_operation()
    assert result.expected_property == expected_value
```

### Testing Sampling
```python
def test_sampling_distribution(self):
    # Create tree with known probabilities
    mcts = create_test_tree()

    # Sample many times
    results = [mcts.sample() for _ in range(100)]

    # Verify statistical properties
    assert len(results) == 100
    assert all(isinstance(r, SampledPath) for r in results)
```

### Testing with MockLLM
```python
def test_with_mock_llm(self, mock_llm):
    mcts = MCTS().with_llm(mock_llm)
    result = mcts.search("question", simulations=5)
    assert mcts.root is not None
```

## Continuous Integration

Tests run automatically on:
- Every commit (if CI/CD configured)
- Before merges to main
- On pull requests

Required: All tests must pass before merge.

## Troubleshooting

### Tests Running Slowly
```bash
# Run in parallel with xdist
pytest tests/ -n auto
```

### Test Failures
```bash
# Show full output on failure
pytest tests/ -vv --tb=long

# Stop at first failure
pytest tests/ -x

# Show local variables on failure
pytest tests/ -l
```

### Coverage Issues
```bash
# See which lines are missing coverage
pytest tests/ --cov=mcts_reasoning --cov-report=term-missing

# Generate detailed HTML report
pytest tests/ --cov=mcts_reasoning --cov-report=html
```

## Adding New Tests

1. Choose appropriate test file (or create new one)
2. Follow naming convention: `test_<module>.py`
3. Organize into test classes: `class TestFeatureName:`
4. Write descriptive test names: `test_<what>_<when>_<expected>`
5. Add docstring explaining the test
6. Follow Given-When-Then structure
7. Verify test passes: `pytest tests/test_<module>.py::TestClass::test_method -v`
8. Check coverage: `pytest --cov=mcts_reasoning.<module> tests/test_<module>.py`

## Resources

- [TDD Testing Summary](/home/spinoza/github/beta/mcts-reasoning/TDD_TESTING_SUMMARY.md) - Comprehensive testing guide
- [pytest Documentation](https://docs.pytest.org/) - Official pytest docs
- [Coverage.py](https://coverage.readthedocs.io/) - Coverage tool docs
- [Project CLAUDE.md](/home/spinoza/github/beta/mcts-reasoning/CLAUDE.md) - Development guidelines

## Test Statistics

- **Total Tests:** 305+
- **New TDD Tests:** 130 (test_core.py + test_sampling.py)
- **Core Module Coverage:** 97% (mcts_reasoning/core.py)
- **Sampling Module Coverage:** 96% (mcts_reasoning/sampling.py)
- **Test Execution Time:** < 1 second for unit tests
- **All Tests Passing:** Yes ✓

---

For detailed information about the TDD approach and test design, see [TDD_TESTING_SUMMARY.md](/home/spinoza/github/beta/mcts-reasoning/TDD_TESTING_SUMMARY.md).
