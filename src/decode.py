from __future__ import annotations

from typing import Iterable, List, Sequence

import torch

from src.tokenization import basic_tokenize, detokenize


@torch.no_grad()
def greedy_decode(
    model,
    src: torch.Tensor,
    bos_id: int,
    eos_id: int,
    max_len: int,
    device: torch.device,
) -> torch.Tensor:
    model.eval()
    if src.dim() == 1:
        src = src.unsqueeze(0)
    src = src.to(device)

    memory, src_key_padding_mask = model.encode(src)
    batch_size = src.size(0)
    generated = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

    for _ in range(max_len - 1):
        logits = model.decode(generated, memory, src_key_padding_mask)
        next_token = logits[:, -1, :].argmax(dim=-1)
        next_token = torch.where(finished, torch.full_like(next_token, eos_id), next_token)
        generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)
        finished = finished | next_token.eq(eos_id)
        if bool(finished.all()):
            break
    return generated


@torch.no_grad()
def translate_texts(
    model,
    texts: Sequence[str],
    src_vocab,
    tgt_vocab,
    device: torch.device,
    max_len: int,
    lowercase: bool = True,
) -> List[str]:
    model.eval()
    results: List[str] = []
    for text in texts:
        tokens = basic_tokenize(text, lowercase=lowercase)
        src_ids = src_vocab.encode(tokens, add_bos=True, add_eos=True, max_len=max_len)
        src_tensor = torch.tensor(src_ids, dtype=torch.long).unsqueeze(0)
        decoded = greedy_decode(
            model=model,
            src=src_tensor,
            bos_id=tgt_vocab.bos_id,
            eos_id=tgt_vocab.eos_id,
            max_len=max_len,
            device=device,
        )
        pred_tokens = tgt_vocab.decode(decoded[0].tolist(), remove_special=True, stop_at_eos=True)
        results.append(detokenize(pred_tokens))
    return results
