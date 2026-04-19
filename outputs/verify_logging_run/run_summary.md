# Run Summary

## Config

- output_dir: outputs\verify_logging_run
- device: cuda
- mixed_precision: True
- d_model: 128
- nhead: 4
- encoder_layers: 2
- decoder_layers: 2
- dim_feedforward: 256
- position_type: sinusoidal
- batch_size: 48
- grad_accum_steps: 1
- epochs: 1
- label_smoothing: 0.0

## Dataset

- train_size: 29000
- valid_size: 1014
- test_size: 1000
- src_vocab_size: 5977
- tgt_vocab_size: 6000

## Results

- best_epoch: 1
- best_val_bleu: 20.57
- test_bleu: 20.14
- test_chrf: 43.55
- test_token_accuracy: 35.86
- test_exact_match: 1.10
- total_training_seconds: 38.27
- parameter_count: 2195584

## Artifacts

- train_log: train.log
- history_csv: history.csv
- step_log_csv: train_steps.csv
- epoch_log_jsonl: epoch_metrics.jsonl
- run_metadata: run_metadata.json
- test_metrics: test_metrics.json
- test_samples: samples\test_samples.txt
- plots: loss_curve.png, perplexity_curve.png, bleu_curve.png, lr_curve.png, runtime_memory_curve.png
