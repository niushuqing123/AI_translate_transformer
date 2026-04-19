from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.download_multi30k import download_multi30k
from scripts.prepare_data import prepare_data
from src.config_utils import load_config
from src.decode import translate_texts
from src.trainer import load_model_from_checkpoint, run_training
from src.utils import get_device

# Direct-run defaults. Edit these values, then run `python scripts/run_pipeline.py`.
CONFIG_PATH = 'configs/full_tiny_en_de.json'
SKIP_DOWNLOAD = False


if __name__ == '__main__':
    config = load_config(CONFIG_PATH)

    data_cfg = config['data']
    if not SKIP_DOWNLOAD:
        download_multi30k(
            raw_dir=data_cfg['raw_dir'],
            src_lang=data_cfg['src_lang'],
            tgt_lang=data_cfg['tgt_lang'],
            force=False,
        )
    prepare_data(config)
    result = run_training(config)

    device = get_device('auto')
    model, src_vocab, tgt_vocab, _, _ = load_model_from_checkpoint(result['best_checkpoint'], device)
    demo_texts = [
        'a man in a blue shirt is running .',
        'a child is playing with a ball .',
        'a woman is standing near a train .',
    ]
    translations = translate_texts(
        model=model,
        texts=demo_texts,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        max_len=config['decode'].get('max_decode_len', config['data']['max_len']),
        lowercase=config['data'].get('lowercase', True),
    )
    print('\n===== Demo translations =====')
    for src, tgt in zip(demo_texts, translations):
        print('SRC:', src)
        print('TGT:', tgt)
        print('-' * 50)
