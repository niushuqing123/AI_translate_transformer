from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

from src.decode import translate_texts
from src.trainer import load_model_from_checkpoint
from src.utils import get_device

DEFAULT_SAMPLES = [
    'a man is riding a bicycle on the street .',
    'two children are playing in the park .',
    'a dog is running through the snow .',
    'a woman is standing near the train .',
]

# Direct-run defaults. Edit these values, then run `python scripts/translate_demo.py`.
CHECKPOINT_PATH = 'outputs/baseline_tiny_en_de/checkpoints/best.pt'
SINGLE_TEXT = None
INPUT_FILE = None
OUTPUT_FILE = None


if __name__ == '__main__':
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f'Missing checkpoint: {checkpoint_path}. Run scripts/train.py first, or edit CHECKPOINT_PATH at the top of scripts/translate_demo.py.'
        )
    device = get_device('auto')
    model, src_vocab, tgt_vocab, config, _ = load_model_from_checkpoint(checkpoint_path, device)
    max_len = int(config['decode'].get('max_decode_len', config['data']['max_len']))
    lowercase = bool(config['data'].get('lowercase', True))

    if SINGLE_TEXT is not None:
        texts = [SINGLE_TEXT]
    elif INPUT_FILE is not None:
        texts = [line.rstrip('\n') for line in Path(INPUT_FILE).open('r', encoding='utf-8')]
    else:
        texts = DEFAULT_SAMPLES

    translations = translate_texts(
        model=model,
        texts=texts,
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        device=device,
        max_len=max_len,
        lowercase=lowercase,
    )

    for src, tgt in zip(texts, translations):
        print('SRC:', src)
        print('TGT:', tgt)
        print('-' * 60)

    if OUTPUT_FILE is None:
        out_path = checkpoint_path.resolve().parents[1] / 'demo_translations.txt'
    else:
        out_path = Path(OUTPUT_FILE)

    lines = []
    for src, tgt in zip(texts, translations):
        lines.extend([f'SRC: {src}', f'TGT: {tgt}', '-' * 60])
    out_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'[DONE] translations saved to {out_path}')
