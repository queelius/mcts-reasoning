#!/usr/bin/env python3
"""
Test script for the enhanced prompt system.

This demonstrates:
1. Tab completion for commands
2. Persistent history
3. Ctrl+R for history search
4. Arrow keys for navigation
"""

from mcts_reasoning.tui.prompt import create_enhanced_prompt

def main():
    print("=" * 60)
    print("Enhanced Prompt Test")
    print("=" * 60)
    print()
    print("Features to try:")
    print("  1. Type '/' and press TAB - see command completions")
    print("  2. Type '/mod' and press TAB - completes to '/model'")
    print("  3. Type '/model ' and press TAB - see provider completions")
    print("  4. Press UP arrow - see previous commands")
    print("  5. Press Ctrl+R - search history")
    print("  6. Type '/exit' to quit")
    print()
    print("Try typing some commands:")
    print()

    prompt = create_enhanced_prompt()
    prompt.update_context("ollama", "llama3.2:latest")

    # Simulate a few commands in history
    print("> /model ollama llama3.2:latest  (simulated history)")
    print("> /ask What is the sum of primes?  (simulated history)")
    print()

    while True:
        try:
            line = prompt.prompt(use_rich=False)

            if not line:
                continue

            print(f"You entered: {line}")

            if line == '/exit':
                print("Goodbye!")
                break

            if line.startswith('/'):
                print(f"  ✓ Recognized command: {line.split()[0]}")
            else:
                print("  ℹ Not a command (should start with /)")

        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

    print()
    print("=" * 60)
    print("History saved to: ~/.mcts-reasoning/history")
    print("Run this script again to see your history!")
    print("=" * 60)


if __name__ == "__main__":
    main()
