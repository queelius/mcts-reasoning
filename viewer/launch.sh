#!/bin/bash

# Launch script for MCTS Live Viewer demo

echo "🚀 MCTS LIVE VIEWER LAUNCHER"
echo "============================"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    kill $SERVER_PID 2>/dev/null
    exit
}

trap cleanup EXIT INT TERM

# Start the viewer server
echo "📡 Starting viewer server..."
python server.py &
SERVER_PID=$!

# Wait for server to start
sleep 2

# Check if server is running
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "❌ Failed to start server"
    exit 1
fi

echo "✅ Server running on http://localhost:8000"
echo ""
echo "🌐 Open your browser at: http://localhost:8000"
echo ""
echo "Waiting 5 seconds for you to open the browser..."
sleep 5

# Run the test
echo ""
echo "🎯 Starting MCTS test..."
echo "============================"
python test_mcts_with_viewer.py --test math

echo ""
echo "Test completed! Keep server running for exploration."
echo "Press Ctrl+C to stop the server."

# Keep running
wait $SERVER_PID