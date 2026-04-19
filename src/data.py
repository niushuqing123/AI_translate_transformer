from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader, Dataset

from src.utils import read_json, load_text_lines
from src.vocab import Vocab


@dataclass
class DatasetBundle:
    train_loader: DataLoader
    valid_loader: DataLoader
    test_loader: DataLoader
    src_vocab: Vocab
    tgt_vocab: Vocab
    meta: Dict


class TranslationDataset(Dataset):
    def __init__(
        self,
        processed_dir: str | Path,
        split: str,
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        max_len: int,
        subset_size: Optional[int] = None,
    ) -> None:
        processed_dir = Path(processed_dir)
        src_path = processed_dir / f'{split}.src'
        tgt_path = processed_dir / f'{split}.tgt'
        src_lines = load_text_lines(src_path)
        tgt_lines = load_text_lines(tgt_path)
        if len(src_lines) != len(tgt_lines):
            raise ValueError(f'Line count mismatch: {src_path} vs {tgt_path}')

        if subset_size is not None:
            src_lines = src_lines[:subset_size]
            tgt_lines = tgt_lines[:subset_size]

        self.samples: List[Tuple[List[int], List[int]]] = []
        for src_line, tgt_line in zip(src_lines, tgt_lines):
            src_tokens = src_line.split()
            tgt_tokens = tgt_line.split()
            src_ids = src_vocab.encode(src_tokens, add_bos=True, add_eos=True, max_len=max_len)
            tgt_ids = tgt_vocab.encode(tgt_tokens, add_bos=True, add_eos=True, max_len=max_len)
            self.samples.append((src_ids, tgt_ids))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[List[int], List[int]]:
        return self.samples[idx]


class Collator:
    def __init__(self, src_pad_id: int, tgt_pad_id: int) -> None:
        self.src_pad_id = src_pad_id
        self.tgt_pad_id = tgt_pad_id

    def __call__(self, batch: Sequence[Tuple[List[int], List[int]]]) -> Dict[str, torch.Tensor]:
        src_batch = [torch.tensor(src, dtype=torch.long) for src, _ in batch]
        tgt_batch = [torch.tensor(tgt, dtype=torch.long) for _, tgt in batch]

        src = torch.nn.utils.rnn.pad_sequence(src_batch, batch_first=True, padding_value=self.src_pad_id)
        tgt = torch.nn.utils.rnn.pad_sequence(tgt_batch, batch_first=True, padding_value=self.tgt_pad_id)

        tgt_in = tgt[:, :-1]
        tgt_out = tgt[:, 1:]
        return {
            'src': src,
            'tgt_in': tgt_in,
            'tgt_out': tgt_out,
        }



def build_dataloader(
    dataset: TranslationDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    src_pad_id: int,
    tgt_pad_id: int,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=Collator(src_pad_id, tgt_pad_id),
        pin_memory=torch.cuda.is_available(),
    )



def load_dataset_bundle(config: Dict) -> DatasetBundle:
    data_cfg = config['data']
    train_cfg = config['train']
    processed_dir = Path(data_cfg['processed_dir'])

    src_vocab = Vocab.from_dict(read_json(processed_dir / 'src_vocab.json'))
    tgt_vocab = Vocab.from_dict(read_json(processed_dir / 'tgt_vocab.json'))
    meta = read_json(processed_dir / 'meta.json')

    train_ds = TranslationDataset(
        processed_dir=processed_dir,
        split='train',
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_len=data_cfg['max_len'],
        subset_size=data_cfg.get('train_subset_size'),
    )
    valid_ds = TranslationDataset(
        processed_dir=processed_dir,
        split='valid',
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_len=data_cfg['max_len'],
        subset_size=data_cfg.get('valid_subset_size'),
    )
    test_ds = TranslationDataset(
        processed_dir=processed_dir,
        split='test',
        src_vocab=src_vocab,
        tgt_vocab=tgt_vocab,
        max_len=data_cfg['max_len'],
        subset_size=data_cfg.get('test_subset_size'),
    )

    train_loader = build_dataloader(
        train_ds,
        batch_size=train_cfg['batch_size'],
        shuffle=True,
        num_workers=train_cfg.get('num_workers', 0),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
    )
    valid_loader = build_dataloader(
        valid_ds,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=train_cfg.get('num_workers', 0),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
    )
    test_loader = build_dataloader(
        test_ds,
        batch_size=train_cfg['batch_size'],
        shuffle=False,
        num_workers=train_cfg.get('num_workers', 0),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
    )
    return DatasetBundle(train_loader, valid_loader, test_loader, src_vocab, tgt_vocab, meta)
