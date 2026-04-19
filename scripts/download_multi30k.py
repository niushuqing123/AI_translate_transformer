from __future__ import annotations

import gzip
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Dict

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils import ensure_dir

BASE_URL = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw'
EXPECTED_LINES = {
    'train': 29000,
    'valid': 1014,
    'test': 1000,
}
SPLIT_TO_FILENAME = {
    'train': 'train',
    'valid': 'val',
    'test': 'test_2016_flickr',
}

# Direct-run defaults. Edit these values, then run `python scripts/download_multi30k.py`.
RAW_DIR = Path('data/raw/multi30k')
SRC_LANG = 'en'
TGT_LANG = 'de'
FORCE_DOWNLOAD = False


def sha256_of_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()



def count_gzip_lines(path: Path) -> int:
    with gzip.open(path, 'rt', encoding='utf-8') as f:
        return sum(1 for _ in f)



def download_file(url: str, target_path: Path) -> None:
    print(f'[DOWNLOAD] {url}')
    target_path.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, target_path.open('wb') as out_f:
        out_f.write(response.read())



def download_multi30k(raw_dir: str | Path, src_lang: str = 'en', tgt_lang: str = 'de', force: bool = False) -> Dict:
    raw_dir = ensure_dir(raw_dir)
    manifest = {
        'dataset': 'Multi30k',
        'src_lang': src_lang,
        'tgt_lang': tgt_lang,
        'files': {},
    }

    for split, prefix in SPLIT_TO_FILENAME.items():
        for lang in (src_lang, tgt_lang):
            filename = f'{prefix}.{lang}.gz'
            url = f'{BASE_URL}/{filename}'
            target_path = raw_dir / filename
            if force or not target_path.exists():
                download_file(url, target_path)
            else:
                print(f'[SKIP] exists: {target_path}')

            num_lines = count_gzip_lines(target_path)
            expected = EXPECTED_LINES[split]
            if num_lines != expected:
                raise ValueError(
                    f'Validation failed for {target_path}: expected {expected} lines, got {num_lines}'
                )

            manifest['files'][filename] = {
                'url': url,
                'sha256': sha256_of_file(target_path),
                'size_bytes': target_path.stat().st_size,
                'num_lines': num_lines,
            }
            print(f'[OK] {filename}: {num_lines} lines')

    manifest_path = raw_dir / 'download_manifest.json'
    with manifest_path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f'[DONE] manifest saved to {manifest_path}')
    return manifest


if __name__ == '__main__':
    download_multi30k(RAW_DIR, SRC_LANG, TGT_LANG, FORCE_DOWNLOAD)
