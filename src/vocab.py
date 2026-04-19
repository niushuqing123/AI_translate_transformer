from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

PAD_TOKEN = '<pad>'
BOS_TOKEN = '<bos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
SPECIAL_TOKENS = [PAD_TOKEN, BOS_TOKEN, EOS_TOKEN, UNK_TOKEN]


@dataclass
class Vocab:
    itos: List[str]

    def __post_init__(self) -> None:
        self.stoi: Dict[str, int] = {token: idx for idx, token in enumerate(self.itos)}
        self.pad_id = self.stoi[PAD_TOKEN]
        self.bos_id = self.stoi[BOS_TOKEN]
        self.eos_id = self.stoi[EOS_TOKEN]
        self.unk_id = self.stoi[UNK_TOKEN]

    def __len__(self) -> int:
        return len(self.itos)

    def encode(self, tokens: Sequence[str], add_bos: bool = True, add_eos: bool = True, max_len: int | None = None) -> List[int]:
        ids = [self.stoi.get(token, self.unk_id) for token in tokens]
        if max_len is not None:
            budget = max_len - int(add_bos) - int(add_eos)
            if budget < 0:
                budget = 0
            ids = ids[:budget]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: Sequence[int], remove_special: bool = True, stop_at_eos: bool = True) -> List[str]:
        tokens: List[str] = []
        for idx in ids:
            token = self.itos[int(idx)] if 0 <= int(idx) < len(self.itos) else UNK_TOKEN
            if stop_at_eos and token == EOS_TOKEN:
                break
            if remove_special and token in SPECIAL_TOKENS:
                continue
            tokens.append(token)
        return tokens

    def to_dict(self) -> Dict[str, List[str]]:
        return {'itos': self.itos}

    @classmethod
    def from_dict(cls, data: Dict[str, List[str]]) -> 'Vocab':
        return cls(itos=list(data['itos']))



def build_vocab(token_sequences: Iterable[Sequence[str]], min_freq: int = 1, max_size: int | None = None) -> Vocab:
    counter: Counter[str] = Counter()
    for tokens in token_sequences:
        counter.update(tokens)

    candidates = [(token, freq) for token, freq in counter.items() if freq >= min_freq and token not in SPECIAL_TOKENS]
    candidates.sort(key=lambda x: (-x[1], x[0]))

    if max_size is not None:
        available = max(max_size - len(SPECIAL_TOKENS), 0)
        candidates = candidates[:available]

    vocab_tokens = SPECIAL_TOKENS + [token for token, _ in candidates]
    return Vocab(vocab_tokens)
