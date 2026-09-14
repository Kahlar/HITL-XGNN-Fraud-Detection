"""Alias runner for Phase 7 execution."""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.run_exp07_hitl import main

if __name__ == "__main__":
    sys.exit(main())
