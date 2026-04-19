# Experiment Summary

- generated_at: 2026-04-19T12:14:19+08:00
- plan_path: configs/experiments_en_de.json
- base_config: configs/full_tiny_en_de.json
- output_root: outputs\experiments
- runner_total_seconds: 1406.65 (23:26)

| name | bleu | chrf | token_accuracy | exact_match | best_epoch | parameter_count | output_dir |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| baseline | 39.35 | 63.95 | 45.90 | 7.10 | 12 | 2195584 | outputs\baseline_tiny_en_de |
| learned_pos | 39.14 | 63.92 | 44.17 | 6.40 | 22 | 2228352 | outputs\experiments\learned_pos |
| label_smoothing | 39.58 | 63.95 | 45.71 | 5.80 | 25 | 2195584 | outputs\experiments\label_smoothing |

## Deltas vs Baseline

| name | delta_bleu | delta_chrf | train_minutes | reused | position_type | label_smoothing |
| --- | ---: | ---: | ---: | --- | --- | ---: |
| baseline | 0.00 | 0.00 | 8.23 | True | sinusoidal | 0.0 |
| learned_pos | -0.22 | -0.03 | 12.19 | False | learned | 0.0 |
| label_smoothing | 0.23 | 0.00 | 10.86 | False | sinusoidal | 0.1 |

## Per-Experiment Artifacts

### baseline

- output_dir: outputs\baseline_tiny_en_de
- run_summary: outputs\baseline_tiny_en_de\run_summary.md
- train_log: outputs\baseline_tiny_en_de\train.log
- history_csv: outputs\baseline_tiny_en_de\history.csv
- test_metrics: outputs\baseline_tiny_en_de\test_metrics.json
- test_samples: outputs\baseline_tiny_en_de\samples\test_samples.txt
- overrides: {}

### learned_pos

- output_dir: outputs\experiments\learned_pos
- run_summary: outputs\experiments\learned_pos\run_summary.md
- train_log: outputs\experiments\learned_pos\train.log
- history_csv: outputs\experiments\learned_pos\history.csv
- test_metrics: outputs\experiments\learned_pos\test_metrics.json
- test_samples: outputs\experiments\learned_pos\samples\test_samples.txt
- overrides: {"model.position_type": "learned"}

### label_smoothing

- output_dir: outputs\experiments\label_smoothing
- run_summary: outputs\experiments\label_smoothing\run_summary.md
- train_log: outputs\experiments\label_smoothing\train.log
- history_csv: outputs\experiments\label_smoothing\history.csv
- test_metrics: outputs\experiments\label_smoothing\test_metrics.json
- test_samples: outputs\experiments\label_smoothing\samples\test_samples.txt
- overrides: {"train.label_smoothing": 0.1}

## Comparison Plots

- outputs\experiments\comparison_bleu.png
- outputs\experiments\comparison_chrf.png
