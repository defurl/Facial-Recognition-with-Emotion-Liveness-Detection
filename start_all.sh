#!/bin/bash

# Define cleanup function to kill background processes
cleanup() {
    echo "Stopping services..."
    kill $(jobs -p)
    exit
}

# Trap SIGINT (Ctrl+C) and SIGTERM
trap cleanup SIGINT SIGTERM

echo "Starting Face Recognition System..."

# Set environment variable for backend
export ENVIRONMENT=final-topic
export BACKEND_HOST="0.0.0.0"
export BACKEND_PORT="8000"

# Navigate to project root (relative to script location)
cd "$(dirname "$0")"

# Start Backend
echo "[1/2] Starting Backend (FastAPI)..."
python3 backend/server.py &
BACKEND_PID=$!

echo "Backend running with PID $BACKEND_PID (ENVIRONMENT=$ENVIRONMENT)"

# Wait a moment for backend to initialize
sleep 2

# Start Frontend
echo "[2/2] Starting Frontend (React)..."
cd frontend
yarn start &
FRONTEND_PID=$!

echo "Frontend running with PID $FRONTEND_PID"

# Wait for both processes
wait
