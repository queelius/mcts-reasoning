#!/bin/bash
# Quick test to verify shell launches correctly

echo "Testing MCTS Shell Launch..."
echo ""

echo "Test 1: Launch with mock provider"
echo "help" | python mcts_shell.py --provider mock --no-rich 2>&1 | grep -q "Available commands"
if [ $? -eq 0 ]; then
    echo "✓ Shell launches successfully"
else
    echo "✗ Shell failed to launch"
    exit 1
fi

echo ""
echo "Test 2: Check help command"
echo "help" | python mcts_shell.py --provider mock --no-rich 2>&1 | grep -q "ask"
if [ $? -eq 0 ]; then
    echo "✓ Help command works"
else
    echo "✗ Help command failed"
    exit 1
fi

echo ""
echo "Test 3: Simple command test"
echo "echo test" | python mcts_shell.py --provider mock --no-rich 2>&1 | grep -q "test"
if [ $? -eq 0 ]; then
    echo "✓ Echo command works"
else
    echo "✗ Echo command failed"
    exit 1
fi

echo ""
echo "✅ All tests passed! Shell is working."
echo ""
echo "To launch the shell, run:"
echo "  python mcts_shell.py"
echo ""
echo "Or install and use:"
echo "  pip install -e ."
echo "  mcts-shell"
