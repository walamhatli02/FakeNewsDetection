#!/bin/bash
# ─────────────────────────────────────────────────────────
# Fake News Detection – Quick Start Script
# Run: bash run_all.sh
# ─────────────────────────────────────────────────────────

set -e
echo "============================================"
echo "  Fake News Detection – Full Pipeline"
echo "============================================"

# 1. Generate sample data (if not already present)
echo ""
echo "[1/4] Generating sample data..."
python src/generate_sample_data.py

# 2. Train model
echo ""
echo "[2/4] Training model..."
python -m src.train --data_dir data/ --output_dir data/

# 3. Start API (background)
echo ""
echo "[3/4] Starting FastAPI backend..."
uvicorn api.main:app --host 0.0.0.0 --port 8000 &
API_PID=$!
sleep 3

# Quick health check
curl -s http://localhost:8000/health && echo ""
echo "✅ API running at http://localhost:8000"
echo "   Docs: http://localhost:8000/docs"

echo ""
echo "[4/4] Instructions:"
echo "  - API:      http://localhost:8000"
echo "  - Docs:     http://localhost:8000/docs"
echo "  - Frontend: cd frontend && npm install && npm start"
echo "  - Docker:   docker-compose up --build"
echo ""
echo "Press Ctrl+C to stop the API server."
wait $API_PID
