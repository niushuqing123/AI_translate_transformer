from __future__ import annotations

import gzip
import json
import shutil
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.prepare_data import prepare_data
from src.trainer import run_training

TOY_TRAIN = [
    ('i like apples .', 'ich mag apfel .'),
    ('you like apples .', 'du magst apfel .'),
    ('i like dogs .', 'ich mag hunde .'),
    ('you like dogs .', 'du magst hunde .'),
    ('we see a dog .', 'wir sehen einen hund .'),
    ('we see a cat .', 'wir sehen eine katze .'),
    ('a dog runs .', 'ein hund rennt .'),
    ('a cat runs .', 'eine katze rennt .'),
] * 1

TOY_VALID = [
    ('i like apples .', 'ich mag apfel .'),
    ('a dog runs .', 'ein hund rennt .'),
]

TOY_TEST = [
    ('you like dogs .', 'du magst hunde .'),
    ('we see a cat .', 'wir sehen eine katze .'),
]


def write_gz_lines(path: Path, lines):
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, 'wt', encoding='utf-8') as f:
        for line in lines:
            f.write(line + '\n')


if __name__ == '__main__':
    tmp_root = Path(tempfile.mkdtemp(prefix='mt_smoke_test_'))
    try:
        raw_dir = tmp_root / 'data' / 'raw' / 'multi30k'
        processed_dir = tmp_root / 'data' / 'processed' / 'toy_en_de'
        output_dir = tmp_root / 'outputs' / 'toy_run'

        write_gz_lines(raw_dir / 'train.en.gz', [x for x, _ in TOY_TRAIN])
        write_gz_lines(raw_dir / 'train.de.gz', [y for _, y in TOY_TRAIN])
        write_gz_lines(raw_dir / 'val.en.gz', [x for x, _ in TOY_VALID])
        write_gz_lines(raw_dir / 'val.de.gz', [y for _, y in TOY_VALID])
        write_gz_lines(raw_dir / 'test_2016_flickr.en.gz', [x for x, _ in TOY_TEST])
        write_gz_lines(raw_dir / 'test_2016_flickr.de.gz', [y for _, y in TOY_TEST])

        config = {
            'project_name': 'smoke_test',
            'seed': 7,
            'device': 'cpu',
            'mixed_precision': False,
            'data': {
                'raw_dir': str(raw_dir),
                'processed_dir': str(processed_dir),
                'src_lang': 'en',
                'tgt_lang': 'de',
                'lowercase': True,
                'max_len': 8,
                'min_freq': 1,
                'max_src_vocab': 128,
                'max_tgt_vocab': 128,
                'train_subset_size': None,
                'valid_subset_size': None,
                'test_subset_size': None,
            },
            'model': {
                'd_model': 16,
                'nhead': 2,
                'num_encoder_layers': 1,
                'num_decoder_layers': 1,
                'dim_feedforward': 32,
                'dropout': 0.1,
                'activation': 'relu',
                'position_type': 'sinusoidal',
                'tie_weights': True,
                'max_position_embeddings': 16,
            },
            'train': {
                'epochs': 1,
                'batch_size': 4,
                'grad_accum_steps': 1,
                'warmup_steps': 5,
                'weight_decay': 0.0,
                'clip_grad_norm': 1.0,
                'label_smoothing': 0.0,
                'num_workers': 0,
                'patience': 4,
                'eval_every': 1,
                'save_every': 1,
                'log_interval': 10,
            },
            'decode': {'max_decode_len': 8},
            'runtime': {
                'output_dir': str(output_dir),
                'memory_budget_mb': 900,
            },
        }

        prepare_data(config)
        result = run_training(config)
        metrics_path = Path(result['test_metrics_path'])
        assert metrics_path.exists(), 'test metrics not generated'
        metrics = json.loads(metrics_path.read_text(encoding='utf-8'))
        print('[SMOKE TEST] metrics =', metrics)
        print('[SMOKE TEST] PASS')
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
