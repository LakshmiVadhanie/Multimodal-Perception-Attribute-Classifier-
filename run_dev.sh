#!/usr/bin/env bash
# ============================================================
# run_dev.sh — One-command launcher for PerceptionAI
#
# Starts:
#   1. Python venv + FastAPI backend on :8000
#   2. Vite frontend on :5173
#   3. Opens browser automatically
# ============================================================

set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=${PYTHON:-python3}

# Homebrew / system node path
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

echo ""
echo "  ⛽  PerceptionAI — dev launcher"
echo "  ================================"
echo ""

# ------------------------------------------------------------------
# Python backend
# ------------------------------------------------------------------
if [ ! -d "$DIR/venv" ]; then
  echo "  📦  Creating Python virtual environment…"
  $PYTHON -m venv "$DIR/venv"
fi
echo "  📦  Installing / verifying Python dependencies…"
"$DIR/venv/bin/pip" install -q -r "$DIR/requirements.txt" 2>&1 | tail -3

echo ""
echo "  🚀  Starting FastAPI backend on http://localhost:8000 …"
cd "$DIR"
"$DIR/venv/bin/uvicorn" serving.api_v2:app \
  --host 0.0.0.0 --port 8000 --reload \
  > /tmp/perception_backend.log 2>&1 &
BACKEND_PID=$!
echo "  ✅  Backend PID: $BACKEND_PID"

# Wait for backend to be ready
echo "  ⏳  Waiting for backend to start…"
for i in $(seq 1 30); do
  if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
    echo "  ✅  Backend is healthy!"
    break
  fi
  sleep 1
done

# ------------------------------------------------------------------
# Frontend
# ------------------------------------------------------------------
cd "$DIR/frontend"

if [ ! -d "node_modules" ]; then
  echo "  📦  Installing frontend dependencies…"
  npm install
fi

echo ""
echo "  🌐  Starting Vite frontend on http://localhost:5173 …"
npx vite --open &
FRONTEND_PID=$!
echo "    Frontend PID: $FRONTEND_PID"

# ------------------------------------------------------------------
# Cleanup on Ctrl+C
# ------------------------------------------------------------------
echo ""
echo "    Both services are running!"
echo "      Backend:  http://localhost:8000"
echo "      Frontend: http://localhost:5173"
echo "      API Docs: http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop everything."
echo ""

cleanup() {
  echo ""
  echo "   Shutting down…"
  kill $BACKEND_PID $FRONTEND_PID 2>/dev/null
  exit 0
}

trap cleanup SIGINT SIGTERM

# Keep running
wait
