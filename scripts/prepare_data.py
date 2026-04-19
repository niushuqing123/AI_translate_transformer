from __future__ import annotations

import gzip
from pathlib import Path
from typing import Dict, List, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import load_config, save_json
from src.tokenization import basic_tokenize
from src.utils import ensure_dir, write_text_lines
from src.vocab import build_vocab

RAW_NAME_MAP = {
    'train': 'train',
    'valid': 'val',
    'test': 'test_2016_flickr',
}

# Direct-run defaults. Edit this value, then run `python scripts/prepare_data.py`.
CONFIG_PATH = 'configs/full_tiny_en_de.json'


def read_gzip_lines(path: Path) -> List[str]:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


def load_parallel_split(raw_dir: Path, split: str, src_lang: str, tgt_lang: str) -> Tuple[List[str], List[str]]:
    prefix = RAW_NAME_MAP[split]
    src_path = raw_dir / f'{prefix}.{src_lang}.gz'
    tgt_path = raw_dir / f'{prefix}.{tgt_lang}.gz'
    if not src_path.exists() or not tgt_path.exists():
        raise FileNotFoundError(
            f'Missing raw files for split={split}. Expected {src_path} and {tgt_path}. '
            f'Please run scripts/download_multi30k.py first.'
        )
    src_lines = read_gzip_lines(src_path)
    tgt_lines = read_gzip_lines(tgt_path)
    if len(src_lines) != len(tgt_lines):
        raise ValueError(f'Line count mismatch in split={split}: {len(src_lines)} vs {len(tgt_lines)}')
    return src_lines, tgt_lines


def preprocess_pairs(src_lines: List[str], tgt_lines: List[str], lowercase: bool) -> Tuple[List[List[str]], List[List[str]]]:
    src_tok: List[List[str]] = []
    tgt_tok: List[List[str]] = []
    for src, tgt in zip(src_lines, tgt_lines):
        src_tokens = basic_tokenize(src, lowercase=lowercase)
        tgt_tokens = basic_tokenize(tgt, lowercase=lowercase)
        if not src_tokens or not tgt_tokens:
            continue
        src_tok.append(src_tokens)
        tgt_tok.append(tgt_tokens)
    return src_tok, tgt_tok


def prepare_data(config: Dict) -> Dict:
    data_cfg = config['data']
    raw_dir = Path(data_cfg['raw_dir'])
    processed_dir = ensure_dir(data_cfg['processed_dir'])
    src_lang = data_cfg['src_lang']
    tgt_lang = data_cfg['tgt_lang']
    lowercase = bool(data_cfg.get('lowercase', True))

    tokenized = {}
    for split in ('train', 'valid', 'test'):
        src_lines, tgt_lines = load_parallel_split(raw_dir, split, src_lang, tgt_lang)
        src_tok, tgt_tok = preprocess_pairs(src_lines, tgt_lines, lowercase=lowercase)
        tokenized[split] = {'src': src_tok, 'tgt': tgt_tok}
        print(f'[INFO] {split}: {len(src_tok)} sentence pairs')

    src_vocab = build_vocab(
        tokenized['train']['src'],
        min_freq=int(data_cfg.get('min_freq', 1)),
        max_size=data_cfg.get('max_src_vocab'),
    )
    tgt_vocab = build_vocab(
        tokenized['train']['tgt'],
        min_freq=int(data_cfg.get('min_freq', 1)),
        max_size=data_cfg.get('max_tgt_vocab'),
    )

    for split in ('train', 'valid', 'test'):
        src_lines = [' '.join(tokens) for tokens in tokenized[split]['src']]
        tgt_lines = [' '.join(tokens) for tokens in tokenized[split]['tgt']]
        write_text_lines(src_lines, processed_dir / f'{split}.src')
        write_text_lines(tgt_lines, processed_dir / f'{split}.tgt')

    save_json(src_vocab.to_dict(), processed_dir / 'src_vocab.json')
    save_json(tgt_vocab.to_dict(), processed_dir / 'tgt_vocab.json')

    meta = {
        'dataset': 'Multi30k',
        'src_lang': src_lang,
        'tgt_lang': tgt_lang,
        'lowercase': lowercase,
        'num_train': len(tokenized['train']['src']),
        'num_valid': len(tokenized['valid']['src']),
        'num_test': len(tokenized['test']['src']),
        'src_vocab_size': len(src_vocab),
        'tgt_vocab_size': len(tgt_vocab),
        'max_len_config': int(data_cfg['max_len']),
        'raw_dir': str(raw_dir),
        'processed_dir': str(processed_dir),
    }
    save_json(meta, processed_dir / 'meta.json')
    print(f'[DONE] processed data saved to {processed_dir}')
    print(f"[DONE] src_vocab={len(src_vocab)} tgt_vocab={len(tgt_vocab)}")
    return meta

if __name__ == '__main__':
    cfg = load_config(CONFIG_PATH)
    prepare_data(cfg)
