"""Make `experiment/` importable as flat modules, the way the scripts run."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
