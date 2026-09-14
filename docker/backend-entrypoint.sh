#!/bin/bash
set -e

echo "================================================================="
echo "Starting HITL-XGNN Fraud Detection Backend Service"
echo "================================================================="

# Wait for PostgreSQL to accept connections
echo "Checking database connectivity..."
python - << 'EOF'
import os
import sys
import time
import socket

host = os.environ.get("POSTGRES_HOST", "localhost")
port = int(os.environ.get("POSTGRES_PORT", 5432))

print(f"Waiting for database at {host}:{port}...")
max_retries = 30
for i in range(max_retries):
    try:
        with socket.create_connection((host, port), timeout=2):
            print("Database port is open and reachable!")
            sys.exit(0)
    except (socket.error, ConnectionRefusedError):
        time.sleep(1)

print(f"Database at {host}:{port} not reachable after {max_retries} attempts.")
# Continue anyway if database not configured (fallback mode)
sys.exit(0)
EOF

# Run Alembic migrations
echo "Executing database migrations (Alembic upgrade head)..."
alembic upgrade head || echo "Migration skipped or already at head."

# Seed model registry from artifacts if needed
echo "Verifying model registry seeding..."
python - << 'EOF'
import asyncio
import os
from pathlib import Path
from src.module8_database.connection import db_manager
from src.module8_database.service import DatabaseService

async def seed_if_needed():
    exp07_path = Path("data/processed/experiments/exp07_hitl_learning.json")
    if exp07_path.exists():
        service = DatabaseService()
        async with db_manager.session() as session:
            try:
                models = await service.model_repo.get_all_models(session)
                if len(models) == 0:
                    print("Seeding model registry from EXP-07 active learning artifact...")
                    await service.seed_model_registry_from_exp07(session, exp07_path)
                    await session.commit()
                else:
                    print(f"Model registry already contains {len(models)} models.")
            except Exception as e:
                print(f"Model seeding skipped: {e}")

asyncio.run(seed_if_needed())
EOF

echo "Starting Uvicorn ASGI Web Server on 0.0.0.0:8000..."
exec uvicorn src.module6_api.main:app --host 0.0.0.0 --port 8000 --workers 2
