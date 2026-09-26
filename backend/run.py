#!/usr/bin/env python3
"""
backend/run.py — Start the OceanEmbed FastAPI backend.

Usage:
    python backend/run.py

Or directly with uvicorn:
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload
"""
import subprocess
import sys

if __name__ == "__main__":
    subprocess.run(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "backend.app.main:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
            "--reload",
        ],
        check=True,
    )
