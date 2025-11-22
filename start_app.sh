#!/bin/bash

# Chess Opening Recommendations - Startup Script
# This script starts the Flask app with automatic port selection

echo "========================================="
echo "Chess Opening Recommendations v2.0"
echo "========================================="
echo ""

# Navigate to project directory
cd "$(dirname "$0")"

# Check if network file exists
if [ ! -f "data/relatedness_network.pkl" ]; then
    echo "❌ ERROR: Network file not found!"
    echo ""
    echo "Please run:"
    echo "  python3 scripts/reprocess_existing_network.py"
    echo ""
    exit 1
fi

echo "✅ Network file found"

# Check dependencies
echo "Checking dependencies..."
python3 -c "import flask, networkx, numpy, scipy, bicm, requests, chess, tqdm" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "❌ Missing dependencies!"
    echo ""
    echo "Installing dependencies..."
    pip3 install -r requirements.txt
    if [ $? -ne 0 ]; then
        echo "❌ Failed to install dependencies"
        exit 1
    fi
fi

echo "✅ Dependencies installed"
echo ""

# Try port 5000 first, then 8000
PORT=5000

# Check if port 5000 is in use
lsof -ti:5000 >/dev/null 2>&1
if [ $? -eq 0 ]; then
    echo "⚠️  Port 5000 is in use (probably AirPlay Receiver)"
    echo "   Using port 8000 instead"
    PORT=8000
fi

# Check if port 8000 is also in use
if [ $PORT -eq 8000 ]; then
    lsof -ti:8000 >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "❌ Port 8000 is also in use!"
        echo ""
        echo "Please free up port 5000 or 8000 and try again."
        echo "To disable AirPlay Receiver (frees port 5000):"
        echo "  System Preferences → General → AirDrop & Handoff → AirPlay Receiver (OFF)"
        exit 1
    fi
fi

echo "========================================="
echo "Starting Flask app on port $PORT..."
echo "========================================="
echo ""
echo "📊 Network Status:"
echo "   - Nodes: 144 openings"
echo "   - Edges: 1,654 connections"
echo "   - EFC: Calculated"
echo ""
echo "🌐 Access the app at:"
echo "   http://localhost:$PORT"
echo ""
echo "💡 Test usernames to try:"
echo "   - DrNykterstein (Magnus Carlsen)"
echo "   - penguingm1 (Andrew Tang)"
echo "   - GMWSO (Qiyu Zhou)"
echo ""
echo "Press Ctrl+C to stop the server"
echo "========================================="
echo ""

# Start the app
export PORT=$PORT
python3 app/app.py
