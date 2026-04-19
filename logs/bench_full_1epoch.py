from pathlib import Path
from src.config_utils import load_config
from src.trainer import run_training
cfg = load_config('configs/full_tiny_en_de.json')
cfg['train']['epochs'] = 1
cfg['train']['patience'] = 99
cfg['runtime']['output_dir'] = 'outputs/bench_full_1epoch'
result = run_training(cfg)
print(result)
