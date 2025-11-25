#!/usr/bin/env python3
"""Test that commands work with or without slash prefix."""

from mcts_reasoning.tui.commands import CommandParser

def test_command_parsing():
    """Test command parsing with and without slashes."""
    print("Testing Command Parsing")
    print("=" * 60)

    test_cases = [
        # (input, expected_name, expected_args)
        ("ask What is 2+2?", "ask", ["What", "is", "2+2?"]),
        ("/ask What is 2+2?", "ask", ["What", "is", "2+2?"]),
        ("model ollama llama3.2", "model", ["ollama", "llama3.2"]),
        ("/model ollama llama3.2", "model", ["ollama", "llama3.2"]),
        ("search 50", "search", ["50"]),
        ("/search 50", "search", ["50"]),
        ("nodes", "nodes", []),
        ("/nodes", "nodes", []),
        ("inspect 5", "inspect", ["5"]),
        ("/inspect 5", "inspect", ["5"]),
        ("probe http://192.168.0.225:11434", "probe", ["http://192.168.0.225:11434"]),
        ("/probe http://192.168.0.225:11434", "probe", ["http://192.168.0.225:11434"]),
        ("export-tree my_tree.json", "export-tree", ["my_tree.json"]),
        ("/export-tree my_tree.json", "export-tree", ["my_tree.json"]),
    ]

    all_passed = True

    for i, (input_str, expected_name, expected_args) in enumerate(test_cases, 1):
        cmd = CommandParser.parse(input_str)

        if cmd is None:
            print(f"❌ Test {i}: Failed to parse: {input_str}")
            all_passed = False
            continue

        if cmd.name != expected_name:
            print(f"❌ Test {i}: Wrong name")
            print(f"   Input:    {input_str}")
            print(f"   Expected: {expected_name}")
            print(f"   Got:      {cmd.name}")
            all_passed = False
            continue

        if cmd.args != expected_args:
            print(f"❌ Test {i}: Wrong args")
            print(f"   Input:    {input_str}")
            print(f"   Expected: {expected_args}")
            print(f"   Got:      {cmd.args}")
            all_passed = False
            continue

        # Check that slash prefix was removed
        slash_prefix = "/" if input_str.startswith("/") else ""
        print(f"✅ Test {i}: {slash_prefix}'{input_str}' → command='{cmd.name}', args={cmd.args}")

    print()
    print("=" * 60)
    if all_passed:
        print("✅ All tests passed!")
        print("Both slash and non-slash commands work correctly.")
    else:
        print("❌ Some tests failed!")
    print("=" * 60)

    # Use assertion instead of return
    assert all_passed, "Some command parsing tests failed"

if __name__ == "__main__":
    try:
        test_command_parsing()
        exit(0)
    except AssertionError:
        exit(1)
