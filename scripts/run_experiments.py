from __future__ import annotations

import csv
import json
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
import sys
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config_utils import apply_overrides, load_json, load_config
from src.plotting import plot_experiment_comparison
from src.trainer import run_training
from src.utils import current_timestamp, ensure_dir, format_seconds, write_json

# Direct-run defaults. Edit this value, then run `python scripts/run_experiments.py`.
PLAN_PATH = 'configs/experiments_en_de.json'


def _summary_row_from_existing_output(name: str, output_dir: Path) -> dict:
    metrics_path = output_dir / 'test_metrics.json'
    if not metrics_path.exists():
        raise FileNotFoundError(
            f'Missing {metrics_path}. Run scripts/train.py first, or edit reuse_output_dir in scripts/run_experiments.py.'
        )
    metrics = load_json(metrics_path)
    resolved_config_path = output_dir / 'resolved_config.json'
    resolved_config = load_json(resolved_config_path) if resolved_config_path.exists() else {}
    return {
        'name': name,
        'bleu': metrics['test_bleu'],
        'chrf': metrics['test_chrf'],
        'token_accuracy': metrics['test_token_accuracy'],
        'exact_match': metrics['test_exact_match'],
        'best_epoch': metrics['best_epoch'],
        'best_val_bleu': metrics.get('best_val_bleu'),
        'test_loss': metrics.get('test_loss'),
        'train_seconds': metrics.get('total_training_seconds'),
        'parameter_count': metrics['parameter_count'],
        'reused': True,
        'overrides': '',
        'position_type': resolved_config.get('model', {}).get('position_type', ''),
        'label_smoothing': resolved_config.get('train', {}).get('label_smoothing', ''),
        'output_dir': str(output_dir),
    }


if __name__ == '__main__':
    plan = load_json(PLAN_PATH)
    base_config = load_config(plan['base_config'])
    output_root = ensure_dir(plan.get('output_root', 'outputs/experiments'))

    runner_start = time.time()
    summary_rows = []
    for experiment in plan['experiments']:
        name = experiment['name']
        reuse_output_dir = experiment.get('reuse_output_dir')
        if reuse_output_dir:
            output_dir = Path(reuse_output_dir)
            print(f'\n===== Reusing experiment: {name} =====')
            summary_rows.append(_summary_row_from_existing_output(name, output_dir))
            continue

        overrides = experiment.get('overrides', {})
        config = apply_overrides(base_config, overrides)
        config['runtime']['output_dir'] = str(Path(output_root) / name)
        print(f'\n===== Running experiment: {name} =====')
        result = run_training(config)
        summary_rows.append(
            {
                'name': name,
                'bleu': result['test_bleu'],
                'chrf': result['test_chrf'],
                'token_accuracy': result['test_token_accuracy'],
                'exact_match': result['test_exact_match'],
                'best_epoch': result['best_epoch'],
                'best_val_bleu': result['best_val_bleu'],
                'test_loss': result['test_loss'],
                'train_seconds': result['total_training_seconds'],
                'parameter_count': result['parameter_count'],
                'reused': False,
                'overrides': json.dumps(overrides, ensure_ascii=False, sort_keys=True),
                'position_type': config['model']['position_type'],
                'label_smoothing': config['train'].get('label_smoothing', 0.0),
                'output_dir': result['output_dir'],
            }
        )

    baseline_row = next((row for row in summary_rows if row['name'] == 'baseline'), summary_rows[0])
    for row in summary_rows:
        row['delta_bleu_vs_baseline'] = float(row['bleu']) - float(baseline_row['bleu'])
        row['delta_chrf_vs_baseline'] = float(row['chrf']) - float(baseline_row['chrf'])
        row['train_minutes'] = float(row['train_seconds']) / 60.0 if row.get('train_seconds') is not None else None

    summary_csv = Path(output_root) / 'summary.csv'
    summary_json = Path(output_root) / 'summary.json'
    summary_md = Path(output_root) / 'summary.md'
    fieldnames = [
        'name',
        'bleu',
        'chrf',
        'token_accuracy',
        'exact_match',
        'best_epoch',
        'best_val_bleu',
        'test_loss',
        'train_seconds',
        'train_minutes',
        'delta_bleu_vs_baseline',
        'delta_chrf_vs_baseline',
        'parameter_count',
        'reused',
        'position_type',
        'label_smoothing',
        'overrides',
        'output_dir',
    ]

    with summary_csv.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(summary_rows)

    comparison_plots = plot_experiment_comparison(summary_rows, output_root)
    runner_seconds = time.time() - runner_start
    write_json(
        {
            'generated_at': current_timestamp(),
            'plan_path': PLAN_PATH,
            'base_config': plan['base_config'],
            'output_root': str(output_root),
            'runner_total_seconds': runner_seconds,
            'comparison_plots': [str(path) for path in comparison_plots],
            'experiments': summary_rows,
        },
        summary_json,
    )

    md_lines = [
        '# Experiment Summary',
        '',
        f'- generated_at: {current_timestamp()}',
        f'- plan_path: {PLAN_PATH}',
        f'- base_config: {plan["base_config"]}',
        f'- output_root: {output_root}',
        f'- runner_total_seconds: {runner_seconds:.2f} ({format_seconds(runner_seconds)})',
        '',
        '| name | bleu | chrf | token_accuracy | exact_match | best_epoch | parameter_count | output_dir |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |',
    ]
    for row in summary_rows:
        md_lines.append(
            f"| {row['name']} | {row['bleu']:.2f} | {row['chrf']:.2f} | {row['token_accuracy']:.2f} | "
            f"{row['exact_match']:.2f} | {row['best_epoch']} | {row['parameter_count']} | {row['output_dir']} |"
        )
    md_lines.extend(
        [
            '',
            '## Deltas vs Baseline',
            '',
            '| name | delta_bleu | delta_chrf | train_minutes | reused | position_type | label_smoothing |',
            '| --- | ---: | ---: | ---: | --- | --- | ---: |',
        ]
    )
    for row in summary_rows:
        train_minutes = '-' if row['train_minutes'] is None else f"{row['train_minutes']:.2f}"
        md_lines.append(
            f"| {row['name']} | {row['delta_bleu_vs_baseline']:.2f} | {row['delta_chrf_vs_baseline']:.2f} | "
            f"{train_minutes} | {row['reused']} | {row['position_type']} | {row['label_smoothing']} |"
        )
    md_lines.extend(
        [
            '',
            '## Per-Experiment Artifacts',
            '',
        ]
    )
    for row in summary_rows:
        output_dir = Path(row['output_dir'])
        md_lines.extend(
            [
                f"### {row['name']}",
                '',
                f"- output_dir: {output_dir}",
                f"- run_summary: {output_dir / 'run_summary.md'}",
                f"- train_log: {output_dir / 'train.log'}",
                f"- history_csv: {output_dir / 'history.csv'}",
                f"- test_metrics: {output_dir / 'test_metrics.json'}",
                f"- test_samples: {output_dir / 'samples' / 'test_samples.txt'}",
                f"- overrides: {row['overrides'] or '{}'}",
                '',
            ]
        )
    if comparison_plots:
        md_lines.extend(
            [
                '## Comparison Plots',
                '',
            ]
        )
        for path in comparison_plots:
            md_lines.append(f'- {path}')
    summary_md.write_text('\n'.join(md_lines) + '\n', encoding='utf-8')
    print(f'[DONE] summary saved to {summary_csv}, {summary_json}, and {summary_md}')
