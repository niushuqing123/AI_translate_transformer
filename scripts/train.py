from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config
from src.trainer import run_training

# Direct-run defaults. Change to `configs/quick_demo_en_de.json` for a faster smoke run.
CONFIG_PATH = 'configs/full_tiny_en_de.json'


if __name__ == '__main__':
    config = load_config(CONFIG_PATH)
    run_training(config)
