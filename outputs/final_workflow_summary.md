# Final Workflow Summary

## Execution

- generated_at: 2026-04-19T12:14:19+08:00
- base_config: configs/full_tiny_en_de.json
- experiments_plan: configs/experiments_en_de.json
- total_training_seconds_across_runs: 1876.56
- total_training_minutes_across_runs: 31.28

## Environment

- python_version: 3.12.10
- platform: Windows-11-10.0.26200-SP0
- torch_version: 2.10.0+cu130
- torch_cuda_version: 13.0
- device: NVIDIA GeForce RTX 2050
- device_total_memory_mb: 4095.56

## Dataset

- train_size: 29000
- valid_size: 1014
- test_size: 1000
- src_vocab_size: 5977
- tgt_vocab_size: 6000

## Baseline

- best_epoch: 12
- best_val_bleu: 40.7667
- test_bleu: 39.3538
- test_chrf: 63.9467
- test_token_accuracy: 45.9007
- test_exact_match: 7.1000
- test_loss: 1.4011
- test_perplexity: 4.0595
- re_eval_bleu: 39.3538
- re_eval_chrf: 63.9467
- baseline_training_seconds: 494.03

## Experiments

| name | bleu | chrf | delta_bleu | delta_chrf | best_epoch | train_minutes | position_type | label_smoothing |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: |
| baseline | 39.3538 | 63.9467 | 0.0000 | 0.0000 | 12 | 8.23 | sinusoidal | 0.0 |
| learned_pos | 39.1386 | 63.9180 | -0.2152 | -0.0287 | 22 | 12.19 | learned | 0.0 |
| label_smoothing | 39.5848 | 63.9511 | 0.2310 | 0.0044 | 25 | 10.86 | sinusoidal | 0.1 |

## Demo Translations

```text
SRC: a man is riding a bicycle on the street .
TGT: ein mann fährt auf der straße fahrrad.
------------------------------------------------------------
SRC: two children are playing in the park .
TGT: zwei kinder spielen im park.
------------------------------------------------------------
SRC: a dog is running through the snow .
TGT: ein hund rennt durch den schnee.
------------------------------------------------------------
SRC: a woman is standing near the train .
TGT: eine frau steht in der nähe des.
------------------------------------------------------------
```

## Key Artifacts

- baseline_run_summary: outputs/baseline_tiny_en_de/run_summary.md
- baseline_train_log: outputs/baseline_tiny_en_de/train.log
- baseline_step_log: outputs/baseline_tiny_en_de/train_steps.csv
- baseline_epoch_log: outputs/baseline_tiny_en_de/epoch_metrics.jsonl
- baseline_test_samples: outputs/baseline_tiny_en_de/samples/test_samples.txt
- baseline_curves: outputs/baseline_tiny_en_de/loss_curve.png, outputs/baseline_tiny_en_de/perplexity_curve.png, outputs/baseline_tiny_en_de/bleu_curve.png, outputs/baseline_tiny_en_de/lr_curve.png, outputs/baseline_tiny_en_de/runtime_memory_curve.png
- experiments_summary_md: outputs/experiments/summary.md
- experiments_summary_csv: outputs/experiments/summary.csv
- experiments_comparison_bleu: outputs/experiments/comparison_bleu.png
- experiments_comparison_chrf: outputs/experiments/comparison_chrf.png

## Notes

- Baseline stopped early at epoch 17 because patience reached 5, but the best checkpoint came from epoch 12.
- In the 3-group comparison, label smoothing slightly improved BLEU over baseline, while learned positional embeddings were slightly worse on this setup.
- download_multi30k.py and prepare_data.py were re-run in this final pass and both completed successfully using the existing dataset files.
