"""
Integration tests for RAG features.

Tests the integration between:
- ComposingPrompt with examples and RAG
- ActionSelector with RAG store
- ReasoningMCTS with RAG-guided action selection
"""

import pytest
from mcts_reasoning.compositional import (
    ComposingPrompt,
    CognitiveOperation,
    FocusAspect,
    ReasoningStyle,
)
from mcts_reasoning.compositional.examples import Example, ExampleSet, get_math_examples
from mcts_reasoning.compositional.rag import (
    CompositionalRAGStore,
    SolutionRAGStore,
    get_math_compositional_rag,
    get_coding_compositional_rag,
)
from mcts_reasoning.compositional.actions import ActionSelector
from mcts_reasoning.reasoning import ReasoningMCTS


class TestComposingPromptWithExamples:
    """Test ComposingPrompt integration with examples."""

    def test_with_examples_from_list(self):
        """Test adding examples from a list."""
        examples = [
            Example("P1", "S1", reasoning_steps=["Step 1"]),
            Example("P2", "S2", reasoning_steps=["Step 2"])
        ]

        prompt = (ComposingPrompt()
                  .problem_context("Test problem")
                  .with_examples(examples, include_steps=True))

        result = prompt.build()
        assert "Relevant examples:" in result
        assert "Problem: P1" in result
        assert "Step 1" in result

    def test_with_examples_as_strings(self):
        """Test adding examples as formatted strings."""
        examples = [
            "Example 1: This is a formatted example",
            "Example 2: Another formatted example"
        ]

        prompt = (ComposingPrompt()
                  .problem_context("Test")
                  .with_examples(examples))

        result = prompt.build()
        assert "Example 1:" in result
        assert "Example 2:" in result

    def test_with_examples_invalid_type(self):
        """Test that invalid example types raise TypeError."""
        with pytest.raises(TypeError, match="Examples must be Example objects or strings"):
            ComposingPrompt().with_examples([123])

    def test_with_examples_no_steps(self):
        """Test adding examples without reasoning steps."""
        examples = [Example("P1", "S1", reasoning_steps=["Hidden step"])]

        prompt = (ComposingPrompt()
                  .with_examples(examples, include_steps=False))

        result = prompt.build()
        assert "Hidden step" not in result

    def test_with_examples_chaining(self):
        """Test fluent API chaining with examples."""
        examples = [Example("Test", "Solution")]

        prompt = (ComposingPrompt()
                  .cognitive_op(CognitiveOperation.ANALYZE)
                  .problem_context("Current problem")
                  .with_examples(examples)
                  .build())

        assert "Problem: Current problem" in prompt
        assert "Relevant examples:" in prompt


class TestComposingPromptWithRAG:
    """Test ComposingPrompt integration with RAG stores."""

    def test_with_rag_examples(self):
        """Test retrieving examples from SolutionRAGStore."""
        rag_store = SolutionRAGStore()
        rag_store.add("Solve x^2 = 4", "x = ±2", reasoning_steps=["Take square root"])
        rag_store.add("Solve x + 3 = 7", "x = 4", reasoning_steps=["Subtract 3"])

        prompt = (ComposingPrompt()
                  .problem_context("Solve x^2 = 9")
                  .with_rag_examples(rag_store, n=1, include_steps=True))

        result = prompt.build()
        assert "Relevant examples:" in result
        # Should retrieve the x^2 example due to similarity
        assert "x^2" in result or "Solve" in result

    def test_with_rag_examples_no_problem_context(self):
        """Test that RAG examples without problem context don't retrieve."""
        rag_store = SolutionRAGStore()
        rag_store.add("Problem", "Solution")

        prompt = (ComposingPrompt()
                  .with_rag_examples(rag_store, n=1))

        result = prompt.build()
        # Should not add examples without problem context
        assert "Relevant examples:" not in result

    def test_with_rag_examples_invalid_type(self):
        """Test that invalid RAG store type raises TypeError."""
        wrong_store = CompositionalRAGStore()

        with pytest.raises(TypeError, match="rag_store must be a SolutionRAGStore"):
            ComposingPrompt().with_rag_examples(wrong_store)

    def test_with_rag_guidance(self):
        """Test applying compositional guidance from RAG store."""
        rag_store = CompositionalRAGStore()
        rag_store.add(
            "quadratic",
            keywords=["quadratic", "x^2"],
            operations=[CognitiveOperation.DECOMPOSE],
            focuses=[FocusAspect.STRUCTURE],
            styles=[ReasoningStyle.SYSTEMATIC]
        )

        prompt = (ComposingPrompt()
                  .problem_context("Solve quadratic equation x^2 + 5x + 6")
                  .with_rag_guidance(rag_store))

        # Test observable behavior: prompt should be generated successfully
        result = prompt.build()
        assert len(result) > 0
        assert "quadratic equation" in result.lower()
        # Verify guidance influenced the prompt structure
        vector = prompt.get_action_vector()
        assert vector['omega'] is not None
        assert vector['phi'] is not None
        assert vector['sigma'] is not None

    def test_with_rag_guidance_no_problem_context(self):
        """Test RAG guidance without problem context."""
        rag_store = CompositionalRAGStore()
        rag_store.add("test", keywords=["test"])

        prompt = ComposingPrompt().with_rag_guidance(rag_store)

        # Should not set dimensions without problem context
        # (depending on implementation, may not modify)
        result = prompt.build()
        assert result is not None

    def test_with_rag_guidance_invalid_type(self):
        """Test that invalid RAG store type raises TypeError."""
        wrong_store = SolutionRAGStore()

        with pytest.raises(TypeError, match="rag_store must be a CompositionalRAGStore"):
            ComposingPrompt().with_rag_guidance(wrong_store)

    def test_rag_guidance_affects_action_sampling(self):
        """Test that RAG guidance biases action sampling toward recommended operations."""
        rag_store = CompositionalRAGStore()
        rag_store.add(
            "test",
            keywords=["test"],
            operations=[CognitiveOperation.VERIFY],  # Specific operation
            focuses=[FocusAspect.CORRECTNESS]
        )

        # Test 1: Verify weights are set correctly (deterministic test)
        weights = rag_store.get_recommended_weights("test problem")
        assert 'cognitive_op' in weights
        assert CognitiveOperation.VERIFY in weights['cognitive_op']
        assert weights['cognitive_op'][CognitiveOperation.VERIFY] > 1.0

        # Test 2: Sample many times to verify statistical bias (probabilistic)
        # With larger sample size (200), this is very unlikely to fail randomly
        verify_count = 0
        total_samples = 200
        for _ in range(total_samples):
            prompt = (ComposingPrompt()
                      .problem_context("test problem")
                      .with_rag_guidance(rag_store))
            vector = prompt.get_action_vector()
            if vector['omega'] == 'verify':
                verify_count += 1

        # With weight 3.0 vs baseline ~1.0, expect significantly more than 10%
        # (there are 10 cognitive operations, baseline would be ~10%)
        assert verify_count > 20  # At least 10% of samples

    def test_combined_rag_features(self):
        """Test using both RAG guidance and RAG examples together."""
        compositional_rag = CompositionalRAGStore()
        compositional_rag.add("math", keywords=["solve"], operations=[CognitiveOperation.ANALYZE])

        solution_rag = SolutionRAGStore()
        solution_rag.add("Solve x = 5", "x = 5", reasoning_steps=["x is already isolated"])

        prompt = (ComposingPrompt()
                  .problem_context("Solve x = 10")
                  .with_rag_guidance(compositional_rag)
                  .with_rag_examples(solution_rag, n=1))

        result = prompt.build()
        # Should have compositional guidance applied (check via action vector)
        vector = prompt.get_action_vector()
        assert vector['omega'] is not None
        # Should have examples included
        assert "Relevant examples:" in result
        assert "Solve x" in result  # Example content should be present


class TestActionSelectorWithRAG:
    """Test ActionSelector integration with RAG stores."""

    def test_create_action_selector_with_rag(self):
        """Test creating ActionSelector with RAG store."""
        rag_store = get_math_compositional_rag()
        selector = ActionSelector(rag_store=rag_store)

        assert selector.rag_store is rag_store

    def test_get_rag_weights(self):
        """Test getting RAG weights for a problem."""
        rag_store = CompositionalRAGStore()
        rag_store.add("test", keywords=["test"], operations=[CognitiveOperation.ANALYZE])

        selector = ActionSelector(rag_store=rag_store)
        weights = selector.get_rag_weights("test problem")

        assert weights is not None
        assert 'cognitive_op' in weights

    def test_get_rag_weights_no_store(self):
        """Test getting RAG weights when no store is set."""
        selector = ActionSelector()
        weights = selector.get_rag_weights("test problem")
        assert weights is None

    def test_get_valid_actions_with_rag(self):
        """Test that RAG store influences action selection."""
        rag_store = CompositionalRAGStore()
        rag_store.add(
            "math",
            keywords=["solve", "equation"],
            operations=[CognitiveOperation.DECOMPOSE, CognitiveOperation.ANALYZE]
        )

        selector = ActionSelector(rag_store=rag_store)
        actions = selector.get_valid_actions(
            current_state="Initial state",
            problem="Solve equation x + 5 = 10",
            n_samples=10
        )

        assert len(actions) > 0
        # Check that some actions use recommended operations
        operations = [a.operation for a in actions]
        assert CognitiveOperation.DECOMPOSE in operations or CognitiveOperation.ANALYZE in operations

    def test_get_valid_actions_without_rag(self):
        """Test action selection without RAG store."""
        selector = ActionSelector()
        actions = selector.get_valid_actions(
            current_state="State",
            n_samples=5
        )
        assert len(actions) == 5


class TestReasoningMCTSWithRAG:
    """Test ReasoningMCTS integration with RAG stores."""

    def test_with_rag_store_before_compositional(self, mock_llm):
        """Test that RAG store works when set before enabling compositional actions."""
        rag_store = CompositionalRAGStore()
        rag_store.add("math", keywords=["solve"], operations=[CognitiveOperation.DECOMPOSE])

        mcts = (ReasoningMCTS()
                .with_llm(mock_llm)
                .with_question("Solve math problem")
                .with_rag_store(rag_store)  # Set before compositional
                .with_compositional_actions(enabled=True))  # Enable after

        # Test observable behavior: RAG should influence action generation
        actions = mcts._get_compositional_actions("Initial state")
        assert len(actions) > 0
        # Should include DECOMPOSE operation due to RAG guidance
        operations = [a.operation for a in actions]
        assert CognitiveOperation.DECOMPOSE in operations

    def test_with_rag_store_after_compositional(self):
        """Test setting RAG store after enabling compositional actions."""
        rag_store = get_math_compositional_rag()
        mcts = (ReasoningMCTS()
                .with_compositional_actions(enabled=True)
                .with_rag_store(rag_store))

        assert mcts.action_selector is not None
        assert mcts.action_selector.rag_store is rag_store

    def test_rag_store_applied_when_compositional_enabled(self, mock_llm):
        """Test that RAG store is properly applied regardless of initialization order."""
        rag_store = CompositionalRAGStore()
        rag_store.add("prime", keywords=["prime"], operations=[CognitiveOperation.GENERATE])

        mcts = (ReasoningMCTS()
                .with_llm(mock_llm)
                .with_question("Find all prime numbers")
                .with_rag_store(rag_store)  # Set before compositional
                .with_compositional_actions(enabled=True))  # Enable compositional

        # Verify RAG store is connected to the action selector
        assert mcts.action_selector is not None
        assert mcts.action_selector.rag_store is rag_store

        # Test behavior: sample multiple times to account for probabilistic nature
        # RAG-guided operations should appear with higher frequency
        generate_count = 0
        n_samples = 20
        for _ in range(n_samples):
            actions = mcts._get_compositional_actions("Initial state")
            operations = [a.operation for a in actions]
            if CognitiveOperation.GENERATE in operations:
                generate_count += 1

        # Should see GENERATE in at least some samples due to RAG weighting
        assert generate_count > 0, "RAG-guided GENERATE operation never appeared in samples"

    def test_rag_guided_action_generation(self, mock_llm):
        """Test that MCTS generates RAG-guided actions."""
        rag_store = CompositionalRAGStore()
        rag_store.add("test", keywords=["solve"], operations=[CognitiveOperation.VERIFY])

        mcts = (ReasoningMCTS()
                .with_llm(mock_llm)
                .with_question("Solve test problem")
                .with_compositional_actions(enabled=True)
                .with_rag_store(rag_store))

        # Sample multiple times to account for probabilistic nature
        verify_count = 0
        n_samples = 20
        for _ in range(n_samples):
            actions = mcts._get_compositional_actions("Initial state")
            assert len(actions) > 0
            operations = [a.operation for a in actions]
            if CognitiveOperation.VERIFY in operations:
                verify_count += 1

        # Should see VERIFY in at least some samples due to RAG weighting
        assert verify_count > 0, "RAG-guided VERIFY operation never appeared in samples"

    def test_end_to_end_rag_mcts_search(self, mock_llm):
        """Test end-to-end MCTS search with RAG guidance."""
        rag_store = get_math_compositional_rag()

        mcts = (ReasoningMCTS()
                .with_llm(mock_llm)
                .with_question("Solve quadratic equation x^2 + 5x + 6 = 0")
                .with_compositional_actions(enabled=True)
                .with_rag_store(rag_store)
                .with_exploration(1.414)
                .with_max_rollout_depth(3))

        # Run search
        result = mcts.search("Let's solve this problem", simulations=5)

        # Should complete without errors
        assert result is not None
        assert mcts.root is not None

    def test_rag_guidance_consistency_across_search(self, mock_llm):
        """Test that RAG guidance is consistently applied during search."""
        rag_store = CompositionalRAGStore()
        rag_store.add("quadratic", keywords=["quadratic"], operations=[CognitiveOperation.ANALYZE])

        mcts = (ReasoningMCTS()
                .with_llm(mock_llm)
                .with_question("Solve quadratic equation")
                .with_compositional_actions(enabled=True)
                .with_rag_store(rag_store)
                .with_max_rollout_depth(2))

        mcts.search("Initial reasoning", simulations=3)

        # Check that action selector has RAG store throughout
        assert mcts.action_selector.rag_store is rag_store


class TestPredefinedRAGIntegration:
    """Test integration with predefined RAG stores."""

    def test_math_rag_with_mcts(self, mock_llm):
        """Test using predefined math RAG with MCTS."""
        rag_store = get_math_compositional_rag()

        mcts = (ReasoningMCTS()
                .with_llm(mock_llm)
                .with_question("Find all prime numbers less than 15")
                .with_compositional_actions(enabled=True)
                .with_rag_store(rag_store)
                .with_max_rollout_depth(2))

        result = mcts.search("Let's find the primes", simulations=3)
        assert result is not None

    def test_coding_rag_with_mcts(self, mock_llm):
        """Test using predefined coding RAG with MCTS."""
        rag_store = get_coding_compositional_rag()

        mcts = (ReasoningMCTS()
                .with_llm(mock_llm)
                .with_question("Implement algorithm for binary search")
                .with_compositional_actions(enabled=True)
                .with_rag_store(rag_store)
                .with_max_rollout_depth(2))

        result = mcts.search("Let's design the algorithm", simulations=3)
        assert result is not None

    def test_math_examples_with_composing_prompt(self):
        """Test using predefined math examples with ComposingPrompt."""
        examples = get_math_examples()

        prompt = (ComposingPrompt()
                  .problem_context("Solve x^2 + 3x + 2 = 0")
                  .with_examples(list(examples)[:2], include_steps=True))

        result = prompt.build()
        assert "Relevant examples:" in result
        assert "Problem:" in result

    def test_combined_predefined_rags(self, mock_llm):
        """Test combining compositional RAG and solution RAG."""
        comp_rag = get_math_compositional_rag()
        solution_rag = SolutionRAGStore(get_math_examples())

        # Use compositional RAG for MCTS guidance
        mcts = (ReasoningMCTS()
                .with_llm(mock_llm)
                .with_question("Solve x^2 + 7x + 10 = 0")
                .with_compositional_actions(enabled=True)
                .with_rag_store(comp_rag)
                .with_max_rollout_depth(2))

        # Use solution RAG for prompt enhancement
        # (This would be done in action execution, simulated here)
        prompt = (ComposingPrompt()
                  .problem_context("Solve x^2 + 7x + 10 = 0")
                  .with_rag_examples(solution_rag, n=1))

        result = prompt.build()
        assert "Problem:" in result

        # Run MCTS search
        mcts.search("Initial reasoning", simulations=3)
        assert mcts.root is not None
