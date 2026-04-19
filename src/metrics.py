from __future__ import annotations

import math
from collections import Counter
from typing import Dict, Iterable, List, Sequence, Tuple

try:
    import sacrebleu  # type: ignore
except Exception:
    sacrebleu = None


def _extract_ngrams(tokens: Sequence[str], n: int) -> Counter:
    if len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1))



def _internal_bleu(references: Sequence[str], hypotheses: Sequence[str], max_order: int = 4, smooth: bool = True) -> float:
    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    ref_length = 0
    hyp_length = 0

    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        ref_length += len(ref_tokens)
        hyp_length += len(hyp_tokens)
        for n in range(1, max_order + 1):
            ref_counts = _extract_ngrams(ref_tokens, n)
            hyp_counts = _extract_ngrams(hyp_tokens, n)
            overlap = ref_counts & hyp_counts
            matches_by_order[n - 1] += sum(overlap.values())
            possible_matches_by_order[n - 1] += max(len(hyp_tokens) - n + 1, 0)

    precisions: List[float] = []
    for i in range(max_order):
        if smooth:
            precisions.append((matches_by_order[i] + 1.0) / (possible_matches_by_order[i] + 1.0))
        else:
            if possible_matches_by_order[i] == 0:
                precisions.append(0.0)
            else:
                precisions.append(matches_by_order[i] / possible_matches_by_order[i])

    if min(precisions) > 0:
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / max_order)
    else:
        geo_mean = 0.0

    if hyp_length == 0:
        return 0.0
    brevity_penalty = 1.0 if hyp_length > ref_length else math.exp(1.0 - ref_length / hyp_length)
    return 100.0 * brevity_penalty * geo_mean



def _char_ngrams(text: str, n: int) -> Counter:
    compact = text.replace(' ', '')
    if len(compact) < n:
        return Counter()
    return Counter(compact[i : i + n] for i in range(len(compact) - n + 1))



def _internal_chrf(references: Sequence[str], hypotheses: Sequence[str], char_order: int = 6, beta: float = 2.0) -> float:
    beta2 = beta ** 2
    precision_scores: List[float] = []
    recall_scores: List[float] = []

    for n in range(1, char_order + 1):
        overlap_total = 0
        hyp_total = 0
        ref_total = 0
        for ref, hyp in zip(references, hypotheses):
            ref_counts = _char_ngrams(ref, n)
            hyp_counts = _char_ngrams(hyp, n)
            overlap = ref_counts & hyp_counts
            overlap_total += sum(overlap.values())
            hyp_total += sum(hyp_counts.values())
            ref_total += sum(ref_counts.values())
        precision_scores.append(overlap_total / hyp_total if hyp_total > 0 else 0.0)
        recall_scores.append(overlap_total / ref_total if ref_total > 0 else 0.0)

    precision = sum(precision_scores) / len(precision_scores)
    recall = sum(recall_scores) / len(recall_scores)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    return 100.0 * ((1 + beta2) * precision * recall) / (beta2 * precision + recall)



def token_accuracy(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    correct = 0
    total = 0
    for ref, hyp in zip(references, hypotheses):
        ref_tokens = ref.split()
        hyp_tokens = hyp.split()
        for ref_tok, hyp_tok in zip(ref_tokens, hyp_tokens):
            if ref_tok == hyp_tok:
                correct += 1
            total += 1
    return 100.0 * correct / total if total > 0 else 0.0



def exact_match_rate(references: Sequence[str], hypotheses: Sequence[str]) -> float:
    if not references:
        return 0.0
    exact = sum(int(ref.strip() == hyp.strip()) for ref, hyp in zip(references, hypotheses))
    return 100.0 * exact / len(references)



def compute_mt_metrics(references: Sequence[str], hypotheses: Sequence[str]) -> Dict[str, float | str]:
    if len(references) != len(hypotheses):
        raise ValueError('references and hypotheses must have the same length')

    if sacrebleu is not None:
        bleu = sacrebleu.corpus_bleu(list(hypotheses), [list(references)])
        chrf = sacrebleu.corpus_chrf(list(hypotheses), [list(references)])
        metrics: Dict[str, float | str] = {
            'bleu': float(bleu.score),
            'chrf': float(chrf.score),
            'metrics_backend': f'sacrebleu {sacrebleu.__version__}',
        }
    else:
        metrics = {
            'bleu': _internal_bleu(references, hypotheses),
            'chrf': _internal_chrf(references, hypotheses),
            'metrics_backend': 'internal',
        }

    metrics['token_accuracy'] = token_accuracy(references, hypotheses)
    metrics['exact_match'] = exact_match_rate(references, hypotheses)
    return metrics
