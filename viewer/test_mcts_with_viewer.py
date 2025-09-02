#!/usr/bin/env python3
"""
Test script to run MCTS with live visualization.

This demonstrates the complete system:
1. Start the viewer server (in another terminal)
2. Run this script to see MCTS in action with live updates
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mcts_clean.mcts_with_ipc import ReasoningMCTSWithIPC
from mcts_clean.llm_adapters import get_llm, MockLLMAdapter


def test_with_math_problem():
    """Test with a math problem that has clear solution."""
    
    print("=" * 60)
    print("MCTS WITH LIVE VISUALIZATION - MATH PROBLEM")
    print("=" * 60)
    print("\n📡 Make sure the viewer is running: python server.py")
    print("🌐 Open browser at http://localhost:8000")
    print("\n" + "=" * 60)
    
    # Give user time to open browser
    print("\nStarting in 5 seconds...")
    time.sleep(5)
    
    # Setup LLM - try to auto-detect, fallback to mock
    try:
        llm = get_llm()
        print(f"Using LLM: {llm.__class__.__name__}")
    except:
        print("Using MockLLM for demonstration")
        llm = MockLLMAdapter({
            "37 * 43": "Let me calculate: 37 * 43 = 37 * 40 + 37 * 3 = 1480 + 111 = 1591",
            "verify": "Verification: 37 * 43 = 1591 is correct.",
            "terminal": "YES",
            "evaluate": "0.95"
        })
    
    # Create MCTS with IPC
    question = "What is 37 * 43? Show your work step by step."
    
    mcts = ReasoningMCTSWithIPC(
        llm_client=llm,
        original_question=question,
        exploration_constant=1.414,
        max_rollout_depth=3,  # Shallow for visualization
        use_compositional=True,
        ipc_host="localhost",
        ipc_port=9999,
        enable_ipc=True
    )
    
    # Run search with fewer simulations for clear visualization
    initial_state = f"Question: {question}\n\nLet's solve this step by step:"
    
    print(f"\n🎯 Running MCTS for: {question}")
    print(f"📊 Simulations: 20 (limited for clear visualization)")
    print("-" * 50)
    
    root = mcts.search(initial_state, num_simulations=20)
    
    # Get best path
    best_path = mcts.get_best_path()
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    print(f"\n📊 Tree Statistics:")
    print(f"  • Root visits: {root.visits}")
    print(f"  • Root value: {root.value:.3f}")
    print(f"  • Children explored: {len(root.children)}")
    
    if best_path:
        print(f"\n🎯 Best Action Sequence:")
        for i, (action, state) in enumerate(best_path[:3], 1):
            print(f"  {i}. {action}")
        
        final_state = best_path[-1][1] if best_path else initial_state
        print(f"\n💡 Final State (last 500 chars):")
        print("-" * 40)
        print(final_state[-500:])
        print("-" * 40)
    
    # Keep connection alive for a bit to see final updates
    print("\n⏳ Keeping connection alive for visualization...")
    time.sleep(5)
    
    mcts.close()
    print("\n✅ Test completed! Check the viewer for the tree structure.")


def test_with_logic_puzzle():
    """Test with a logic puzzle requiring compositional reasoning."""
    
    print("\n" * 3)
    print("=" * 60)
    print("MCTS WITH LIVE VISUALIZATION - LOGIC PUZZLE")
    print("=" * 60)
    
    # Setup LLM
    try:
        llm = get_llm()
        print(f"Using LLM: {llm.__class__.__name__}")
    except:
        print("Using MockLLM for demonstration")
        llm = MockLLMAdapter({
            "analyze": "All labels are wrong, so Box A doesn't contain apples.",
            "decompose": "We need to find one box to sample from that gives maximum information.",
            "solve": "Pick from the 'Apples and Oranges' box - since it's wrong, it contains only one type.",
            "verify": "If we get an apple, then Box C has only apples, Box B has mixed, Box A has oranges.",
            "terminal": "YES",
            "evaluate": "0.85"
        })
    
    # Create MCTS with IPC
    question = """Three boxes contain fruits. Box A is labeled "Apples", Box B is labeled "Oranges", 
    and Box C is labeled "Apples and Oranges". However, all labels are wrong. 
    You can pick one fruit from one box. How do you determine the correct labels?"""
    
    mcts = ReasoningMCTSWithIPC(
        llm_client=llm,
        original_question=question,
        exploration_constant=1.5,  # More exploration for logic puzzle
        max_rollout_depth=4,
        use_compositional=True,
        ipc_host="localhost",
        ipc_port=9999,
        enable_ipc=True
    )
    
    initial_state = f"Question: {question}\n\nLet's reason through this systematically:"
    
    print(f"\n🎯 Running MCTS for logic puzzle")
    print(f"📊 Simulations: 30")
    print("-" * 50)
    
    root = mcts.search(initial_state, num_simulations=30)
    
    # Results
    best_path = mcts.get_best_path()
    
    print("\n" + "=" * 50)
    print("RESULTS")
    print("=" * 50)
    
    print(f"\n📊 Tree Statistics:")
    print(f"  • Root visits: {root.visits}")
    print(f"  • Root value: {root.value:.3f}")
    print(f"  • Children explored: {len(root.children)}")
    
    # Show how compositional actions were used
    if root.children:
        print(f"\n🧩 Compositional Actions Used:")
        for i, child in enumerate(root.children[:5], 1):
            if child.action_taken:
                print(f"  {i}. {child.action_taken}")
                print(f"     Visits: {child.visits}, Value: {child.value:.3f}")
    
    # Keep alive
    print("\n⏳ Keeping connection alive...")
    time.sleep(5)
    
    mcts.close()
    print("\n✅ Logic puzzle test completed!")


def test_compositional_exploration():
    """Test to specifically show compositional action diversity."""
    
    print("\n" * 3)
    print("=" * 60)
    print("MCTS - COMPOSITIONAL ACTION EXPLORATION")
    print("=" * 60)
    
    # Use mock for consistent demonstration
    llm = MockLLMAdapter({
        "analyze": "Analysis reveals multiple architectural patterns.",
        "decompose": "Breaking down: 1) Load balancing 2) Caching 3) Database scaling",
        "solve": "Solution: Use microservices with horizontal scaling.",
        "verify": "Verification shows this handles the required load.",
        "synthesize": "Combining all aspects into a cohesive architecture.",
    })
    
    # Simple question to focus on action diversity
    question = "What are the key factors in designing a scalable web service?"
    
    mcts = ReasoningMCTSWithIPC(
        llm_client=llm,
        original_question=question,
        exploration_constant=2.0,  # High exploration
        max_rollout_depth=2,  # Shallow to see more breadth
        use_compositional=True,
        ipc_host="localhost",
        ipc_port=9999,
        enable_ipc=True
    )
    
    initial_state = f"Question: {question}\n\nAnalyzing:"
    
    print(f"\n🎯 Testing compositional action diversity")
    print(f"📊 Simulations: 40 (with high exploration)")
    print("-" * 50)
    
    root = mcts.search(initial_state, num_simulations=40)
    
    print("\n" + "=" * 50)
    print("COMPOSITIONAL ACTION DIVERSITY")
    print("=" * 50)
    
    # Analyze action diversity
    from collections import Counter
    
    operations = Counter()
    focuses = Counter()
    styles = Counter()
    
    for child in root.children:
        if child.action_taken:
            action = child.action_taken
            operations[action.operation.value] += child.visits
            focuses[action.focus.value] += child.visits
            styles[action.style.value] += child.visits
    
    print(f"\n📊 Action Component Distribution:")
    
    print(f"\n  Operations (top 3):")
    for op, count in operations.most_common(3):
        print(f"    • {op}: {count} visits")
    
    print(f"\n  Focus Areas (top 3):")
    for focus, count in focuses.most_common(3):
        print(f"    • {focus}: {count} visits")
    
    print(f"\n  Reasoning Styles (top 3):")
    for style, count in styles.most_common(3):
        print(f"    • {style}: {count} visits")
    
    print(f"\n🌳 Tree Shape:")
    print(f"  • Root children: {len(root.children)}")
    print(f"  • Unique action combinations: {len(set(c.action_taken for c in root.children if c.action_taken))}")
    
    # Keep alive
    time.sleep(5)
    mcts.close()
    print("\n✅ Compositional exploration test completed!")


def test_with_mock_fast():
    """Fast test with mock LLM for quick visualization testing."""
    
    print("=" * 60)
    print("MCTS FAST TEST WITH MOCK LLM")
    print("=" * 60)
    
    # Mock LLM with deterministic responses
    llm = MockLLMAdapter()
    
    question = "How do we solve world hunger?"
    
    mcts = ReasoningMCTSWithIPC(
        llm_client=llm,
        original_question=question,
        exploration_constant=1.414,
        max_rollout_depth=3,
        use_compositional=True,
        ipc_host="localhost",
        ipc_port=9999,
        enable_ipc=True
    )
    
    initial_state = f"Question: {question}\n\nLet's approach this systematically:"
    
    print(f"\n🎯 Running fast mock test")
    print(f"📊 Simulations: 50 (mock is fast)")
    print("-" * 50)
    
    root = mcts.search(initial_state, num_simulations=50)
    
    print(f"\n📊 Results:")
    print(f"  • Root visits: {root.visits}")
    print(f"  • Root value: {root.value:.3f}")
    print(f"  • Children explored: {len(root.children)}")
    print(f"  • Mock LLM calls: {llm.call_count}")
    
    time.sleep(3)
    mcts.close()
    print("\n✅ Fast test completed!")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test MCTS with live viewer")
    parser.add_argument(
        "--test",
        choices=["math", "logic", "compositional", "mock", "all"],
        default="mock",
        help="Which test to run (default: mock for fast testing)"
    )
    parser.add_argument(
        "--llm",
        choices=["auto", "mock", "ollama", "openai", "anthropic"],
        default="auto",
        help="Which LLM to use (default: auto-detect)"
    )
    
    args = parser.parse_args()
    
    print("\n🚀 MCTS WITH LIVE VISUALIZATION TEST")
    print("=" * 60)
    print("Prerequisites:")
    print("1. Start viewer: cd integrations/mcts_live_viewer && python server.py")
    print("2. Open browser: http://localhost:8000")
    print("3. Run this test script")
    print("=" * 60)
    
    # Override LLM if specified
    if args.llm != "auto":
        os.environ["LLM_PROVIDER"] = args.llm
    
    if args.test == "mock" or args.test == "all":
        test_with_mock_fast()
    
    if args.test == "math" or args.test == "all":
        test_with_math_problem()
    
    if args.test == "logic" or args.test == "all":
        test_with_logic_puzzle()
    
    if args.test == "compositional" or args.test == "all":
        test_compositional_exploration()
    
    print("\n" + "=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
    print("\n📊 Check the viewer at http://localhost:8000 for the tree visualization!")
    print("💡 The tree shows:")
    print("  • Node colors: Green (high value) to Red (low value)")
    print("  • Node size: Based on visit count")
    print("  • Click nodes to see details")
    print("  • Watch MCTS phases in real-time")