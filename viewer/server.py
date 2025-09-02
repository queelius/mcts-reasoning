#!/usr/bin/env python3
"""
Live MCTS Tree Viewer

Real-time visualization of MCTS tree growth with compositional actions.
Shows the four phases of MCTS as they happen.
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, Any, List
import threading
import queue

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
import uvicorn


app = FastAPI(title="MCTS Live Tree Viewer")

# Global state
tree_data = {
    "nodes": {},  # node_id -> node_data
    "edges": [],  # List of {from: id, to: id}
    "current_phase": "idle",
    "statistics": {
        "total_simulations": 0,
        "current_simulation": 0,
        "best_value": 0,
        "total_nodes": 0
    }
}

connected_clients: List[WebSocket] = []
event_queue = queue.Queue()


@app.get("/")
async def get_index():
    """Serve the visualization page"""
    return HTMLResponse(content=HTML_CONTENT)


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket for browser clients"""
    await websocket.accept()
    connected_clients.append(websocket)
    
    # Send current tree state
    await websocket.send_json({
        "type": "full_update",
        "data": tree_data
    })
    
    try:
        while True:
            # Keep connection alive
            await websocket.receive_text()
    except WebSocketDisconnect:
        connected_clients.remove(websocket)


@app.get("/api/tree")
async def get_tree():
    """Get current tree data"""
    return tree_data


@app.get("/api/stats")
async def get_stats():
    """Get current statistics"""
    return tree_data["statistics"]


async def broadcast_update(update_type: str, data: Dict[str, Any]):
    """Broadcast update to all connected clients"""
    message = {
        "type": update_type,
        "data": data,
        "timestamp": datetime.now().isoformat()
    }
    
    disconnected = []
    for client in connected_clients:
        try:
            await client.send_json(message)
        except:
            disconnected.append(client)
    
    for client in disconnected:
        if client in connected_clients:
            connected_clients.remove(client)


def handle_mcts_event(event_data: Dict[str, Any]):
    """Handle incoming MCTS event"""
    event_type = event_data.get("event_type", "")
    data = event_data.get("data", {})
    
    # Update tree based on event type
    if event_type == "mcts.initialized":
        # Reset tree
        tree_data["nodes"] = {}
        tree_data["edges"] = []
        tree_data["statistics"]["total_simulations"] = data.get("num_simulations", 0)
        tree_data["statistics"]["current_simulation"] = 0
        
        # Create root node
        root_id = "root"
        tree_data["nodes"][root_id] = {
            "id": root_id,
            "label": "ROOT",
            "value": 0,
            "visits": 0,
            "action": None,
            "depth": 0
        }
    
    elif event_type == "node.created":
        node_id = str(data.get("node_id"))
        parent_id = str(data.get("parent_id"))
        
        # Add node
        tree_data["nodes"][node_id] = {
            "id": node_id,
            "label": data.get("action", "?")[:20],  # Truncate action
            "value": 0,
            "visits": 0,
            "action": data.get("action"),
            "depth": data.get("depth", 0),
            "state_preview": data.get("state_preview", "")
        }
        
        # Add edge
        tree_data["edges"].append({
            "from": parent_id,
            "to": node_id
        })
        
        tree_data["statistics"]["total_nodes"] = len(tree_data["nodes"])
    
    elif event_type == "node.updated":
        node_id = str(data.get("node_id"))
        if node_id in tree_data["nodes"]:
            tree_data["nodes"][node_id]["visits"] = data.get("visits", 0)
            tree_data["nodes"][node_id]["value"] = data.get("value", 0)
    
    elif event_type.startswith("phase."):
        # Update current phase
        if "start" in event_type:
            phase = event_type.replace("phase.", "").replace("_start", "")
            tree_data["current_phase"] = phase
        elif "end" in event_type:
            tree_data["current_phase"] = "idle"
        
        # Track simulation progress
        if event_type == "phase.selection_start":
            tree_data["statistics"]["current_simulation"] += 1
    
    elif event_type == "reasoning.initialized":
        # Store question and action space info
        tree_data["question"] = data.get("question", "")
        tree_data["action_space_size"] = data.get("action_space_size", 0)
        tree_data["use_compositional"] = data.get("use_compositional", False)
    
    # Queue event for async broadcast
    event_queue.put({
        "type": "event",
        "event": event_type,
        "data": data
    })


def run_tcp_listener(host: str = "localhost", port: int = 9999):
    """Run TCP listener for MCTS events"""
    import socket
    import struct
    
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(5)
    
    print(f"🎯 MCTS IPC listener started on {host}:{port}")
    
    while True:
        try:
            client, addr = server.accept()
            print(f"📡 MCTS client connected from {addr}")
            
            while True:
                # Read length prefix
                length_data = client.recv(4)
                if not length_data:
                    break
                
                length = struct.unpack('!I', length_data)[0]
                
                # Read data
                data = b''
                while len(data) < length:
                    chunk = client.recv(min(4096, length - len(data)))
                    if not chunk:
                        break
                    data += chunk
                
                if data:
                    event_data = json.loads(data.decode('utf-8'))
                    handle_mcts_event(event_data)
            
            client.close()
            print(f"📡 MCTS client disconnected")
            
        except Exception as e:
            print(f"TCP listener error: {e}")


# HTML Content
HTML_CONTENT = """
<!DOCTYPE html>
<html>
<head>
    <title>MCTS Live Tree Viewer</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <script src="https://unpkg.com/chart.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            margin: 0;
            background: #1a1a2e;
            color: #eee;
            display: flex;
            height: 100vh;
        }
        
        #sidebar {
            width: 350px;
            background: #16213e;
            padding: 20px;
            overflow-y: auto;
            border-right: 2px solid #0f3460;
        }
        
        #main {
            flex: 1;
            display: flex;
            flex-direction: column;
        }
        
        #header {
            background: #0f3460;
            padding: 15px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        #phases {
            display: flex;
            gap: 10px;
        }
        
        .phase {
            padding: 8px 15px;
            background: #16213e;
            border-radius: 20px;
            font-size: 12px;
            transition: all 0.3s;
        }
        
        .phase.active {
            background: #e94560;
            animation: pulse 1s infinite;
        }
        
        @keyframes pulse {
            0%, 100% { opacity: 1; }
            50% { opacity: 0.7; }
        }
        
        #network {
            flex: 1;
            background: #0f3460;
            margin: 10px;
            border-radius: 8px;
        }
        
        .stat-box {
            background: #0f3460;
            padding: 15px;
            margin: 10px 0;
            border-radius: 8px;
        }
        
        .stat-label {
            font-size: 12px;
            color: #999;
            text-transform: uppercase;
        }
        
        .stat-value {
            font-size: 24px;
            font-weight: bold;
            color: #e94560;
        }
        
        #node-info {
            background: #0f3460;
            padding: 15px;
            margin-top: 20px;
            border-radius: 8px;
            max-height: 300px;
            overflow-y: auto;
        }
        
        #node-info h3 {
            margin: 0 0 10px 0;
            color: #e94560;
        }
        
        .info-row {
            display: flex;
            justify-content: space-between;
            padding: 5px 0;
            border-bottom: 1px solid #16213e;
        }
        
        #value-chart {
            height: 200px;
            margin-top: 20px;
        }
        
        .connection-status {
            padding: 5px 10px;
            border-radius: 15px;
            font-size: 12px;
            background: #16a085;
            color: white;
        }
        
        .connection-status.disconnected {
            background: #e74c3c;
        }
    </style>
</head>
<body>
    <div id="sidebar">
        <h1>🎯 MCTS Explorer</h1>
        <div id="connection-status" class="connection-status">Connected</div>
        
        <div class="stat-box">
            <div class="stat-label">Simulation</div>
            <div class="stat-value">
                <span id="current-sim">0</span> / <span id="total-sim">0</span>
            </div>
        </div>
        
        <div class="stat-box">
            <div class="stat-label">Total Nodes</div>
            <div class="stat-value" id="total-nodes">0</div>
        </div>
        
        <div class="stat-box">
            <div class="stat-label">Best Value</div>
            <div class="stat-value" id="best-value">0.00</div>
        </div>
        
        <div id="node-info">
            <h3>Node Details</h3>
            <p style="color: #999;">Click a node to see details</p>
        </div>
        
        <canvas id="value-chart"></canvas>
    </div>
    
    <div id="main">
        <div id="header">
            <div id="phases">
                <div class="phase" id="phase-selection">Selection</div>
                <div class="phase" id="phase-expansion">Expansion</div>
                <div class="phase" id="phase-rollout">Rollout</div>
                <div class="phase" id="phase-backprop">Backprop</div>
            </div>
            <div id="question" style="font-size: 14px; color: #aaa;"></div>
        </div>
        
        <div id="network"></div>
    </div>
    
    <script>
        let network = null;
        let nodes = new vis.DataSet();
        let edges = new vis.DataSet();
        let ws = null;
        let selectedNodeId = null;
        let valueHistory = [];
        let valueChart = null;
        
        function initNetwork() {
            const container = document.getElementById('network');
            const data = { nodes: nodes, edges: edges };
            
            const options = {
                nodes: {
                    shape: 'dot',
                    size: 20,
                    font: { 
                        size: 12, 
                        color: '#fff',
                        strokeWidth: 3,
                        strokeColor: '#0f3460'
                    },
                    borderWidth: 2,
                    shadow: true
                },
                edges: {
                    width: 2,
                    color: { color: '#16a085' },
                    arrows: { to: { enabled: true, scaleFactor: 0.5 } },
                    smooth: { type: 'cubicBezier' }
                },
                layout: {
                    hierarchical: {
                        direction: 'UD',
                        sortMethod: 'directed',
                        nodeSpacing: 100,
                        levelSeparation: 80
                    }
                },
                physics: {
                    enabled: true,
                    hierarchicalRepulsion: {
                        nodeDistance: 120
                    }
                }
            };
            
            network = new vis.Network(container, data, options);
            
            network.on('click', function(params) {
                if (params.nodes.length > 0) {
                    selectedNodeId = params.nodes[0];
                    showNodeDetails(selectedNodeId);
                }
            });
        }
        
        function initChart() {
            const ctx = document.getElementById('value-chart').getContext('2d');
            valueChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: [],
                    datasets: [{
                        label: 'Best Value',
                        data: [],
                        borderColor: '#e94560',
                        backgroundColor: 'rgba(233, 69, 96, 0.1)',
                        tension: 0.4
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {
                        legend: { display: false }
                    },
                    scales: {
                        y: {
                            beginAtZero: true,
                            max: 1,
                            grid: { color: '#16213e' },
                            ticks: { color: '#999' }
                        },
                        x: {
                            grid: { color: '#16213e' },
                            ticks: { color: '#999' }
                        }
                    }
                }
            });
        }
        
        function getNodeColor(value, visits) {
            if (visits === 0) return '#666';
            if (value >= 0.8) return '#27ae60';
            if (value >= 0.6) return '#f39c12';
            if (value >= 0.4) return '#e67e22';
            return '#e74c3c';
        }
        
        function updateTree(data) {
            // Update nodes
            for (const [nodeId, nodeData] of Object.entries(data.nodes)) {
                const color = getNodeColor(nodeData.value, nodeData.visits);
                const size = Math.min(50, 15 + nodeData.visits * 2);
                
                if (nodes.get(nodeId)) {
                    nodes.update({
                        id: nodeId,
                        label: nodeData.label,
                        color: color,
                        size: size,
                        title: `Value: ${nodeData.value.toFixed(3)}\\nVisits: ${nodeData.visits}`
                    });
                } else {
                    nodes.add({
                        id: nodeId,
                        label: nodeData.label,
                        color: color,
                        size: size,
                        title: `Value: ${nodeData.value.toFixed(3)}\\nVisits: ${nodeData.visits}`
                    });
                }
            }
            
            // Update edges
            for (const edge of data.edges) {
                const edgeId = `${edge.from}-${edge.to}`;
                if (!edges.get(edgeId)) {
                    edges.add({
                        id: edgeId,
                        from: edge.from,
                        to: edge.to
                    });
                }
            }
            
            // Update statistics
            document.getElementById('current-sim').textContent = 
                data.statistics.current_simulation;
            document.getElementById('total-sim').textContent = 
                data.statistics.total_simulations;
            document.getElementById('total-nodes').textContent = 
                data.statistics.total_nodes;
            document.getElementById('best-value').textContent = 
                data.statistics.best_value.toFixed(3);
            
            // Update value chart
            if (data.statistics.best_value > 0) {
                valueHistory.push(data.statistics.best_value);
                if (valueHistory.length > 50) valueHistory.shift();
                
                valueChart.data.labels = valueHistory.map((_, i) => i);
                valueChart.data.datasets[0].data = valueHistory;
                valueChart.update('none');
            }
        }
        
        function updatePhase(phase) {
            document.querySelectorAll('.phase').forEach(el => {
                el.classList.remove('active');
            });
            
            if (phase && phase !== 'idle') {
                const phaseEl = document.getElementById(`phase-${phase}`);
                if (phaseEl) phaseEl.classList.add('active');
            }
        }
        
        function showNodeDetails(nodeId) {
            const nodeData = nodes.get(nodeId);
            if (!nodeData) return;
            
            const info = document.getElementById('node-info');
            info.innerHTML = `
                <h3>Node ${nodeId}</h3>
                <div class="info-row">
                    <span>Action:</span>
                    <span>${nodeData.label || 'N/A'}</span>
                </div>
                <div class="info-row">
                    <span>Value:</span>
                    <span>${nodeData.value ? nodeData.value.toFixed(3) : '0.000'}</span>
                </div>
                <div class="info-row">
                    <span>Visits:</span>
                    <span>${nodeData.visits || 0}</span>
                </div>
            `;
        }
        
        function connect() {
            ws = new WebSocket('ws://' + window.location.host + '/ws');
            
            ws.onopen = () => {
                document.getElementById('connection-status').textContent = 'Connected';
                document.getElementById('connection-status').classList.remove('disconnected');
            };
            
            ws.onmessage = (event) => {
                const message = JSON.parse(event.data);
                
                if (message.type === 'full_update') {
                    updateTree(message.data);
                    updatePhase(message.data.current_phase);
                    if (message.data.question) {
                        document.getElementById('question').textContent = 
                            message.data.question.substring(0, 50) + '...';
                    }
                } else if (message.type === 'event') {
                    // Handle specific events
                    if (message.event.startsWith('phase.')) {
                        const phase = message.data.current_phase || 
                            message.event.replace('phase.', '').replace('_start', '');
                        updatePhase(phase);
                    }
                    
                    // Refresh tree
                    fetch('/api/tree')
                        .then(r => r.json())
                        .then(updateTree);
                }
            };
            
            ws.onclose = () => {
                document.getElementById('connection-status').textContent = 'Disconnected';
                document.getElementById('connection-status').classList.add('disconnected');
                setTimeout(connect, 2000);
            };
        }
        
        // Initialize
        initNetwork();
        initChart();
        connect();
    </script>
</body>
</html>
"""


async def process_event_queue():
    """Process queued events and broadcast updates"""
    while True:
        try:
            if not event_queue.empty():
                event = event_queue.get_nowait()
                await broadcast_update(event["type"], event)
        except:
            pass
        await asyncio.sleep(0.01)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run the visualization server"""
    # Start TCP listener in background
    tcp_thread = threading.Thread(
        target=run_tcp_listener,
        daemon=True
    )
    tcp_thread.start()
    
    # Start FastAPI server
    print(f"🚀 MCTS Live Viewer at http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="warning")


if __name__ == "__main__":
    run_server()