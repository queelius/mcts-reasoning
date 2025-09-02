"""
Integration with Ollama for MCTS Reasoning

This connects the clean MCTS implementation to your existing Ollama setup.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from typing import Optional
from mcts_clean.reasoning_mcts import ReasoningMCTS, CompositionalAction, run_reasoning_mcts
from mcts_clean.mcts_core import MCTSNode
from tree_of_thought_mcts.llm.ollama_client import OllamaClient, LLMManager


class OllamaMCTS(ReasoningMCTS):
    """
    MCTS with Ollama integration.
    
    This bridges the clean MCTS with your existing Ollama infrastructure.
    """
    
    def __init__(
        self,
        original_question: str,
        host: str = "localhost",
        port: int = 11434,
        reasoning_model: str = "llama2",
        exploration_constant: float = 1.414,
        max_rollout_depth: int = 5,
        use_compositional: bool = True
    ):
        # Initialize Ollama client
        ollama_client = OllamaClient(host=host, port=port)
        self.llm_manager = LLMManager(
            ollama_client=ollama_client,
            reasoning_model=reasoning_model,
            evaluation_model=reasoning_model  # Same model for now
        )
        
        # Create wrapper that matches expected interface
        llm_wrapper = OllamaWrapper(self.llm_manager)
        
        super().__init__(
            llm_client=llm_wrapper,
            original_question=original_question,
            exploration_constant=exploration_constant,
            max_rollout_depth=max_rollout_depth,
            use_compositional=use_compositional
        )


class OllamaWrapper:
    """Wrapper to match the expected LLM interface"""
    
    def __init__(self, llm_manager: LLMManager):
        self.llm_manager = llm_manager
    
    def generate(self, prompt: str, temperature: float = 0.7, max_tokens: int = 150) -> str:
        """Generate response using Ollama"""
        response = self.llm_manager.reason(prompt, temperature=temperature)
        # Truncate to max_tokens approximation (rough: 1 token ≈ 4 chars)
        text = response.text if hasattr(response, 'text') else str(response)
        max_chars = max_tokens * 4
        if len(text) > max_chars:
            text = text[:max_chars] + "..."
        return text


def run_mcts_with_ollama(
    question: str,
    num_simulations: int = 100,
    host: str = "localhost",
    port: int = 11434,
    model: str = "llama2",
    show_tree: bool = True
):
    """
    Run MCTS reasoning with Ollama.
    
    This is the main entry point for Ollama users.
    """
    print(f"🎯 Starting MCTS for: {question}")
    print(f"📊 Simulations: {num_simulations}")
    print(f"🤖 Model: {model}")
    print("-" * 50)
    
    # Create MCTS instance
    mcts = OllamaMCTS(
        original_question=question,
        host=host,
        port=port,
        reasoning_model=model,
        exploration_constant=1.414,
        max_rollout_depth=5,
        use_compositional=True
    )
    
    # Run search
    initial_state = f"Question: {question}\n\nLet's solve this step by step:"
    root = mcts.search(initial_state, num_simulations=num_simulations)
    
    # Get best path
    best_path = mcts.get_best_path()
    
    print("\n" + "=" * 50)
    print("MCTS SEARCH COMPLETE")
    print("=" * 50)
    
    # Show tree statistics
    print(f"\n📊 Tree Statistics:")
    print(f"  • Root visits: {root.visits}")
    print(f"  • Root value: {root.value:.3f}")
    print(f"  • Children explored: {len(root.children)}")
    
    # Show best action sequence
    if best_path:
        print(f"\n🎯 Best Action Sequence:")
        for i, (action, _) in enumerate(best_path[:5], 1):
            print(f"  {i}. {action}")
        
        # Show final solution
        final_state = best_path[-1][1] if best_path else initial_state
        print(f"\n💡 Final Solution:")
        print("-" * 40)
        # Show last 1000 chars of solution
        solution_text = final_state[-1000:] if len(final_state) > 1000 else final_state
        print(solution_text)
        print("-" * 40)
    
    # Optionally show tree structure
    if show_tree:
        print(f"\n🌳 Tree Structure (top 3 levels):")
        _print_tree(root, max_depth=3)
    
    return root


def _print_tree(node: MCTSNode, depth: int = 0, max_depth: int = 3, prefix: str = ""):
    """Pretty print the tree structure"""
    if depth > max_depth:
        return
    
    # Print current node
    action_str = str(node.action_taken) if node.action_taken else "ROOT"
    value_str = f"V:{node.value:.2f}" if node.visits > 0 else "V:?"
    visits_str = f"N:{node.visits}"
    
    print(f"{prefix}├─ {action_str} [{visits_str}, {value_str}]")
    
    # Print children
    for i, child in enumerate(node.children):
        is_last = i == len(node.children) - 1
        child_prefix = prefix + ("    " if is_last else "│   ")
        _print_tree(child, depth + 1, max_depth, child_prefix)


def compare_with_without_compositional(question: str, num_simulations: int = 50):
    """
    Compare MCTS with and without compositional actions.
    
    This demonstrates the value of compositional prompting.
    """
    print("🔬 Comparing MCTS with and without compositional actions\n")
    
    # Run without compositional
    print("1️⃣ WITHOUT Compositional Actions:")
    print("-" * 40)
    mcts_simple = OllamaMCTS(
        original_question=question,
        use_compositional=False
    )
    initial_state = f"Question: {question}"
    root_simple = mcts_simple.search(initial_state, num_simulations=num_simulations)
    print(f"Simple MCTS: Root value = {root_simple.value:.3f}")
    
    # Run with compositional
    print("\n2️⃣ WITH Compositional Actions:")
    print("-" * 40)
    mcts_comp = OllamaMCTS(
        original_question=question,
        use_compositional=True
    )
    root_comp = mcts_comp.search(initial_state, num_simulations=num_simulations)
    print(f"Compositional MCTS: Root value = {root_comp.value:.3f}")
    
    # Compare
    print("\n📊 Comparison:")
    print(f"  • Simple action space: {3} actions")
    print(f"  • Compositional space: {5*5*3} = {75} actions")
    print(f"  • Value improvement: {(root_comp.value - root_simple.value):.3f}")
    print(f"  • Children explored (simple): {len(root_simple.children)}")
    print(f"  • Children explored (compositional): {len(root_comp.children)}")


if __name__ == "__main__":
    # Example 1: Simple question
    print("Example 1: Simple Math Problem")
    print("=" * 50)
    run_mcts_with_ollama(
        question="What is 15 * 17?",
        num_simulations=30,
        model="llama2",
        show_tree=True
    )
    
    print("\n\n")
    
    # Example 2: Complex reasoning
    print("Example 2: Complex Reasoning Problem")
    print("=" * 50)
    run_mcts_with_ollama(
        question="How can I design a distributed system that handles 1 million requests per second?",
        num_simulations=50,
        model="llama2",
        show_tree=False
    )
    
    print("\n\n")
    
    # Example 3: Compare compositional vs simple
    print("Example 3: Compositional vs Simple Actions")
    print("=" * 50)
    compare_with_without_compositional(
        question="What are the trade-offs between microservices and monolithic architecture?",
        num_simulations=30
    )