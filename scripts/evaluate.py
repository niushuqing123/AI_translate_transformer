from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_dataset_bundle
from src.trainer import evaluate_loader, load_model_from_checkpoint
from src.utils import get_device, write_json

# Direct-run defaults. Edit these values, then run `python scripts/evaluate.py`.
CHECKPOINT_PATH = 'outputs/baseline_tiny_en_de/checkpoints/best.pt'
OUTPUT_PATH = None


if __name__ == '__main__':
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f'Missing checkpoint: {checkpoint_path}. Run scripts/train.py first, or edit CHECKPOINT_PATH at the top of scripts/evaluate.py.'
        )
    device = get_device('auto')
    model, src_vocab, tgt_vocab, config, checkpoint = load_model_from_checkpoint(checkpoint_path, device)
    bundle = load_dataset_bundle(config)
    criterion = __import__('torch').nn.CrossEntropyLoss(ignore_index=tgt_vocab.pad_id, label_smoothing=config['train'].get('label_smoothing', 0.0))
    metrics = evaluate_loader(
        model=model,
        loader=bundle.test_loader,
        criterion=criterion,
        tgt_vocab=tgt_vocab,
        device=device,
        max_decode_len=config['decode'].get('max_decode_len', config['data']['max_len']),
    )
    output = {
        'test_loss': metrics['loss'],
        'test_perplexity': metrics['perplexity'],
        'test_bleu': metrics['bleu'],
        'test_chrf': metrics['chrf'],
        'test_token_accuracy': metrics['token_accuracy'],
        'test_exact_match': metrics['exact_match'],
        'metrics_backend': metrics['metrics_backend'],
    }
    if OUTPUT_PATH is None:
        out_path = checkpoint_path.resolve().parents[1] / 're_evaluated_test_metrics.json'
    else:
        out_path = Path(OUTPUT_PATH)
    write_json(output, out_path)
    print(f'[DONE] evaluation written to {out_path}')
    for key, value in output.items():
        print(f'{key}: {value}')
