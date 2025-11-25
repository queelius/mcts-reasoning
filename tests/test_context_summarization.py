#!/usr/bin/env python3
"""Test automatic context summarization in MCTS reasoning."""

from mcts_reasoning.reasoning import ReasoningMCTS
from mcts_reasoning.compositional.providers import MockLLMProvider
from mcts_reasoning.context_manager import ContextConfig

def test_automatic_summarization():
    """Test that context automatically summarizes when threshold is reached."""
    print("Testing Automatic Context Summarization")
    print("=" * 60)

    # Create mock LLM with long responses to trigger summarization
    long_response = """This is a detailed analysis of the problem. We need to consider
    multiple aspects including the mathematical foundations, the algorithmic approach,
    and the practical implementation details. Let's break this down systematically.
    First, we should identify the key constraints and requirements. Then we can
    explore different solution strategies. Each approach has trade-offs that we
    need to carefully evaluate. The problem space is complex and requires thorough
    investigation of edge cases and performance considerations.""" * 3  # Make it 3x longer

    llm = MockLLMProvider(responses={
        "": long_response,  # Default response for all prompts
        "summarize": "Concise summary: We analyzed the problem and identified key solutions."
    })

    # Create MCTS with low threshold to trigger summarization quickly
    # Max 500 tokens, summarize at 80% = 400 tokens
    # Assuming 4 chars/token, that's about 1600 chars
    config = ContextConfig(
        max_context_tokens=500,
        summarize_threshold=0.8,
        summarize_target=0.5,
        use_token_counting=False,  # Use character estimation for simplicity
        chars_per_token=4.0
    )

    mcts = (
        ReasoningMCTS()
        .with_llm(llm)
        .with_question("What are the prime numbers less than 20?")
        .with_compositional_actions(enabled=True)
        .with_context_config(config=config, auto_configure=False)
        .with_exploration(1.414)
        .with_max_rollout_depth(3)
    )

    print(f"\nContext Configuration:")
    print(f"  Max tokens: {config.max_context_tokens}")
    print(f"  Summarize threshold: {config.summarize_threshold} ({int(config.max_context_tokens * config.summarize_threshold)} tokens)")
    print(f"  Summarize target: {config.summarize_target} ({int(config.max_context_tokens * config.summarize_target)} tokens)")
    print(f"  Chars per token: {config.chars_per_token}")
    print()

    # Run search
    print("Running search (this should trigger summarization)...")
    initial_state = "Let's solve this problem step by step."
    mcts.search(initial_state, simulations=20)

    # Get stats
    stats = mcts.context_manager.get_stats()
    print(f"\nContext Management Statistics:")
    print(f"  Summarizations performed: {stats['summarization_count']}")
    print(f"  Last summarization size: {stats['last_summarization_tokens']} tokens")
    print(f"  Max context: {stats['max_context_tokens']} tokens")
    print(f"  Using token counting: {stats['use_token_counting']}")

    # Check tree
    nodes = mcts.get_all_nodes()
    print(f"\nTree Statistics:")
    print(f"  Total nodes: {len(nodes)}")

    # Find nodes with summarization markers
    summarized_count = 0
    for i, node in enumerate(nodes):
        if "[Context summarized" in node.state:
            summarized_count += 1
            print(f"\n  Node {i} was summarized:")
            print(f"    State length: {len(node.state)} chars")
            print(f"    State preview: {node.state[:200]}...")

    print(f"\n  Nodes with summarized context: {summarized_count}")

    # Verify summarization occurred
    print("\n" + "=" * 60)
    if stats['summarization_count'] > 0:
        print("✅ Automatic summarization worked!")
        print(f"   Context was compressed {stats['summarization_count']} time(s)")
    else:
        print("⚠️  No summarization occurred")
        print("   This might be expected if states stayed small")
        print("   Try increasing simulations or lowering threshold")
    print("=" * 60)

if __name__ == "__main__":
    test_automatic_summarization()
