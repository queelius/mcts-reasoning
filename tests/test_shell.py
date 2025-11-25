#!/usr/bin/env python3
"""
Quick test script for the shell.

Tests basic command parsing and execution.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from mcts_reasoning.shell.parser import ShellParser
from mcts_reasoning.shell.core import Shell
from mcts_reasoning.config import Config


def test_parser():
    """Test command parsing."""
    parser = ShellParser()

    # Test simple command
    pipeline = parser.parse("help")
    assert pipeline is not None
    assert len(pipeline.commands) == 1
    assert pipeline.commands[0].name == "help"
    print("✓ Simple command parsing")

    # Test pipe
    pipeline = parser.parse("ask 'test' | search 100")
    assert pipeline is not None
    assert len(pipeline.commands) == 2
    assert pipeline.commands[0].name == "ask"
    assert pipeline.commands[1].name == "search"
    print("✓ Pipe parsing")

    # Test flags
    pipeline = parser.parse("sample 5 --strategy diverse")
    assert pipeline is not None
    assert pipeline.commands[0].kwargs['strategy'] == 'diverse'
    print("✓ Flag parsing")

    # Test output redirection
    pipeline = parser.parse("echo test > output.txt")
    assert pipeline is not None
    assert pipeline.output_redirect is not None
    print("✓ Output redirection parsing")


def test_shell_creation():
    """Test shell creation."""
    config = Config()
    shell = Shell(config=config, use_rich=False)
    assert shell is not None
    assert shell.registry is not None
    assert len(shell.registry.command_names()) > 0
    print(f"✓ Shell created with {len(shell.registry.command_names())} commands")


def test_command_registry():
    """Test command registry."""
    config = Config()
    shell = Shell(config=config, use_rich=False)

    # Test command lookup
    help_cmd = shell.registry.get('help')
    assert help_cmd is not None
    print("✓ Command lookup works")

    # Test aliases
    exit_cmd = shell.registry.get('q')
    assert exit_cmd is not None
    print("✓ Command aliases work")

    # List all commands
    all_commands = shell.registry.all_commands()
    print(f"✓ Total commands registered: {len(all_commands)}")


def main():
    """Run all tests."""
    print("Testing MCTS-Reasoning Shell\n")

    try:
        test_parser()
        test_shell_creation()
        test_command_registry()

        print("\n✅ All tests passed!")

    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        sys.exit(1)

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
