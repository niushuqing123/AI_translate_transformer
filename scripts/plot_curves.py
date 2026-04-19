from __future__ import annotations

import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.plotting import plot_training_curves

# Direct-run defaults. Edit this value, then run `python scripts/plot_curves.py`.
HISTORY_PATH = Path('outputs/baseline_tiny_en_de/history.csv')


if __name__ == '__main__':
    history_path = HISTORY_PATH
    if not history_path.exists():
        raise FileNotFoundError(
            f'Missing history file: {history_path}. Run scripts/train.py first, or edit HISTORY_PATH at the top of scripts/plot_curves.py.'
        )
    generated = plot_training_curves(history_path)
    print(f"[DONE] saved {', '.join(str(path) for path in generated)}")
