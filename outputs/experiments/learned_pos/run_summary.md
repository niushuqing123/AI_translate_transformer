# Run Summary

## Config

- output_dir: outputs\experiments\learned_pos
- device: cuda
- mixed_precision: True
- d_model: 128
- nhead: 4
- encoder_layers: 2
- decoder_layers: 2
- dim_feedforward: 256
- position_type: learned
- batch_size: 48
- grad_accum_steps: 1
- epochs: 25
- label_smoothing: 0.0

## Dataset

- train_size: 29000
- valid_size: 1014
- test_size: 1000
- src_vocab_size: 5977
- tgt_vocab_size: 6000

## Results

- best_epoch: 22
- best_val_bleu: 40.48
- test_bleu: 39.14
- test_chrf: 63.92
- test_token_accuracy: 44.17
- test_exact_match: 6.40
- total_training_seconds: 731.12
- parameter_count: 2228352

## Artifacts

- train_log: train.log
- history_csv: history.csv
- step_log_csv: train_steps.csv
- epoch_log_jsonl: epoch_metrics.jsonl
- run_metadata: run_metadata.json
- test_metrics: test_metrics.json
- test_samples: samples\test_samples.txt
- plots: loss_curve.png, perplexity_curve.png, bleu_curve.png, lr_curve.png, runtime_memory_curve.png
