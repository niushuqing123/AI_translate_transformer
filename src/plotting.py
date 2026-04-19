from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt


def load_history_rows(path: str | Path) -> List[Dict[str, float]]:
    path = Path(path)
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        rows = []
        for row in reader:
            rows.append({key: float(value) if key != 'epoch' else int(value) for key, value in row.items()})
    if not rows:
        raise ValueError(f'History file is empty: {path}')
    return rows


def plot_training_curves(history_path: str | Path) -> List[Path]:
    history_path = Path(history_path)
    rows = load_history_rows(history_path)
    output_dir = history_path.parent

    epochs = [int(row['epoch']) for row in rows]
    train_loss = [float(row['train_loss']) for row in rows]
    val_loss = [float(row['val_loss']) for row in rows]
    train_ppl = [float(row['train_ppl']) for row in rows]
    val_ppl = [float(row['val_ppl']) for row in rows]
    val_bleu = [float(row['val_bleu']) for row in rows]
    lr = [float(row['lr']) for row in rows]
    epoch_seconds = [float(row['epoch_seconds']) for row in rows]
    peak_memory_mb = [float(row['peak_memory_mb']) for row in rows]

    generated: List[Path] = []

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_loss, marker='o', label='train_loss')
    plt.plot(epochs, val_loss, marker='o', label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    loss_path = output_dir / 'loss_curve.png'
    plt.tight_layout()
    plt.savefig(loss_path, dpi=150)
    plt.close()
    generated.append(loss_path)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_ppl, marker='o', label='train_ppl')
    plt.plot(epochs, val_ppl, marker='o', label='val_ppl')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title('Training and Validation Perplexity')
    plt.legend()
    plt.grid(True, alpha=0.3)
    ppl_path = output_dir / 'perplexity_curve.png'
    plt.tight_layout()
    plt.savefig(ppl_path, dpi=150)
    plt.close()
    generated.append(ppl_path)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, val_bleu, marker='o', label='val_bleu')
    plt.xlabel('Epoch')
    plt.ylabel('BLEU')
    plt.title('Validation BLEU')
    plt.legend()
    plt.grid(True, alpha=0.3)
    bleu_path = output_dir / 'bleu_curve.png'
    plt.tight_layout()
    plt.savefig(bleu_path, dpi=150)
    plt.close()
    generated.append(bleu_path)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, lr, marker='o', label='learning_rate')
    plt.xlabel('Epoch')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()
    plt.grid(True, alpha=0.3)
    lr_path = output_dir / 'lr_curve.png'
    plt.tight_layout()
    plt.savefig(lr_path, dpi=150)
    plt.close()
    generated.append(lr_path)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, epoch_seconds, marker='o', color='tab:blue', label='epoch_seconds')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Epoch Seconds', color='tab:blue')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(epochs, peak_memory_mb, marker='s', color='tab:red', label='peak_memory_mb')
    ax2.set_ylabel('Peak Memory (MB)', color='tab:red')
    ax2.tick_params(axis='y', labelcolor='tab:red')

    fig.suptitle('Epoch Runtime and Peak Memory')
    runtime_path = output_dir / 'runtime_memory_curve.png'
    fig.tight_layout()
    fig.savefig(runtime_path, dpi=150)
    plt.close(fig)
    generated.append(runtime_path)

    return generated


def plot_experiment_comparison(summary_rows: Iterable[Dict[str, float | str]], output_dir: str | Path) -> List[Path]:
    output_dir = Path(output_dir)
    rows = list(summary_rows)
    if not rows:
        return []

    names = [str(row['name']) for row in rows]
    bleu = [float(row['bleu']) for row in rows]
    chrf = [float(row['chrf']) for row in rows]
    generated: List[Path] = []

    plt.figure(figsize=(8, 5))
    plt.bar(names, bleu, color=['#4C72B0', '#55A868', '#C44E52'][: len(names)])
    for idx, value in enumerate(bleu):
        plt.text(idx, value + 0.1, f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    plt.ylabel('Test BLEU')
    plt.title('Experiment Comparison: BLEU')
    plt.grid(True, axis='y', alpha=0.3)
    bleu_path = output_dir / 'comparison_bleu.png'
    plt.tight_layout()
    plt.savefig(bleu_path, dpi=150)
    plt.close()
    generated.append(bleu_path)

    plt.figure(figsize=(8, 5))
    plt.bar(names, chrf, color=['#8172B3', '#64B5CD', '#CCB974'][: len(names)])
    for idx, value in enumerate(chrf):
        plt.text(idx, value + 0.1, f'{value:.2f}', ha='center', va='bottom', fontsize=9)
    plt.ylabel('Test chrF')
    plt.title('Experiment Comparison: chrF')
    plt.grid(True, axis='y', alpha=0.3)
    chrf_path = output_dir / 'comparison_chrf.png'
    plt.tight_layout()
    plt.savefig(chrf_path, dpi=150)
    plt.close()
    generated.append(chrf_path)

    return generated
