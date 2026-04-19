from __future__ import annotations

import copy
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from src.data import DatasetBundle, load_dataset_bundle
from src.decode import greedy_decode
from src.metrics import compute_mt_metrics
from src.model import TinyTransformerNMT
from src.plotting import plot_training_curves
from src.utils import (
    append_csv_row,
    append_jsonl_row,
    autocast_if_available,
    count_parameters,
    current_timestamp,
    create_grad_scaler,
    ensure_dir,
    format_seconds,
    get_cuda_allocated_memory_mb,
    get_environment_info,
    get_cuda_peak_memory_mb,
    get_device,
    reset_cuda_peak_memory,
    set_seed,
    write_json,
)
from src.vocab import Vocab


class WarmupInverseSqrtScheduler:
    def __init__(self, optimizer: torch.optim.Optimizer, d_model: int, warmup_steps: int = 400, factor: float = 1.0) -> None:
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = max(1, warmup_steps)
        self.factor = factor
        self.step_num = 0

    def get_lr(self, step_num: int | None = None) -> float:
        step = self.step_num if step_num is None else max(1, step_num)
        return self.factor * (self.d_model ** -0.5) * min(step ** -0.5, step * (self.warmup_steps ** -1.5))

    def step(self) -> float:
        self.step_num += 1
        lr = self.get_lr(self.step_num)
        for group in self.optimizer.param_groups:
            group['lr'] = lr
        return lr

    def state_dict(self) -> Dict[str, Any]:
        return {
            'd_model': self.d_model,
            'warmup_steps': self.warmup_steps,
            'factor': self.factor,
            'step_num': self.step_num,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.d_model = int(state_dict['d_model'])
        self.warmup_steps = int(state_dict['warmup_steps'])
        self.factor = float(state_dict['factor'])
        self.step_num = int(state_dict['step_num'])
        lr = self.get_lr(max(self.step_num, 1))
        for group in self.optimizer.param_groups:
            group['lr'] = lr



def build_model(config: Dict[str, Any], src_vocab: Vocab, tgt_vocab: Vocab) -> TinyTransformerNMT:
    model_cfg = config['model']
    return TinyTransformerNMT(
        src_vocab_size=len(src_vocab),
        tgt_vocab_size=len(tgt_vocab),
        src_pad_id=src_vocab.pad_id,
        tgt_pad_id=tgt_vocab.pad_id,
        d_model=model_cfg['d_model'],
        nhead=model_cfg['nhead'],
        num_encoder_layers=model_cfg['num_encoder_layers'],
        num_decoder_layers=model_cfg['num_decoder_layers'],
        dim_feedforward=model_cfg['dim_feedforward'],
        dropout=model_cfg['dropout'],
        activation=model_cfg.get('activation', 'relu'),
        position_type=model_cfg.get('position_type', 'sinusoidal'),
        tie_weights=model_cfg.get('tie_weights', True),
        max_position_embeddings=model_cfg.get('max_position_embeddings', 128),
    )



def build_optimizer_and_scheduler(config: Dict[str, Any], model: nn.Module) -> Tuple[torch.optim.Optimizer, WarmupInverseSqrtScheduler]:
    model_cfg = config['model']
    train_cfg = config['train']
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=0.0,
        betas=(0.9, 0.98),
        eps=1e-9,
        weight_decay=train_cfg.get('weight_decay', 0.0),
    )
    scheduler = WarmupInverseSqrtScheduler(
        optimizer=optimizer,
        d_model=model_cfg['d_model'],
        warmup_steps=train_cfg.get('warmup_steps', 400),
        factor=1.0,
    )
    scheduler.step()
    return optimizer, scheduler



def _move_batch_to_device(batch: Dict[str, torch.Tensor], device: torch.device) -> Dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}



def _build_criterion(tgt_pad_id: int, label_smoothing: float) -> nn.Module:
    return nn.CrossEntropyLoss(ignore_index=tgt_pad_id, label_smoothing=label_smoothing)



def _loss_to_perplexity(loss: float) -> float:
    return math.exp(min(loss, 10.0))


@torch.no_grad()
def collect_translation_samples(
    model: nn.Module,
    loader,
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    device: torch.device,
    max_decode_len: int,
    sample_count: int,
) -> List[Dict[str, str]]:
    model.eval()
    records: List[Dict[str, str]] = []
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        decoded = greedy_decode(
            model=model,
            src=batch['src'],
            bos_id=tgt_vocab.bos_id,
            eos_id=tgt_vocab.eos_id,
            max_len=max_decode_len,
            device=device,
        )

        src_ids = batch['src'].detach().cpu().tolist()
        tgt_out = batch['tgt_out'].detach().cpu().tolist()
        decoded_ids = decoded.detach().cpu().tolist()

        for src_item, ref_item, hyp_item in zip(src_ids, tgt_out, decoded_ids):
            records.append(
                {
                    'source': ' '.join(src_vocab.decode(src_item, remove_special=True, stop_at_eos=True)),
                    'reference': ' '.join(tgt_vocab.decode(ref_item, remove_special=True, stop_at_eos=True)),
                    'hypothesis': ' '.join(tgt_vocab.decode(hyp_item, remove_special=True, stop_at_eos=True)),
                }
            )
            if len(records) >= sample_count:
                return records
    return records


def _write_sample_records(path: str | Path, records: List[Dict[str, str]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []
    for idx, record in enumerate(records, start=1):
        lines.extend(
            [
                f'[{idx}]',
                f"SRC: {record['source']}",
                f"REF: {record['reference']}",
                f"HYP: {record['hypothesis']}",
                '',
            ]
        )
    path.write_text('\n'.join(lines).strip() + '\n', encoding='utf-8')


@torch.no_grad()
def generate_predictions(
    model: nn.Module,
    loader,
    tgt_vocab: Vocab,
    device: torch.device,
    max_decode_len: int,
) -> Tuple[List[str], List[str]]:
    model.eval()
    references: List[str] = []
    hypotheses: List[str] = []
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        decoded = greedy_decode(
            model=model,
            src=batch['src'],
            bos_id=tgt_vocab.bos_id,
            eos_id=tgt_vocab.eos_id,
            max_len=max_decode_len,
            device=device,
        )
        tgt_out = batch['tgt_out'].detach().cpu().tolist()
        decoded_ids = decoded.detach().cpu().tolist()

        for ref_ids, hyp_ids in zip(tgt_out, decoded_ids):
            ref_tokens = tgt_vocab.decode(ref_ids, remove_special=True, stop_at_eos=True)
            hyp_tokens = tgt_vocab.decode(hyp_ids, remove_special=True, stop_at_eos=True)
            references.append(' '.join(ref_tokens))
            hypotheses.append(' '.join(hyp_tokens))
    return references, hypotheses


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader,
    criterion: nn.Module,
    tgt_vocab: Vocab,
    device: torch.device,
    max_decode_len: int,
) -> Dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_batches = 0
    for batch in loader:
        batch = _move_batch_to_device(batch, device)
        logits = model(batch['src'], batch['tgt_in'])
        vocab_size = logits.size(-1)
        loss = criterion(logits.reshape(-1, vocab_size), batch['tgt_out'].reshape(-1))
        total_loss += float(loss.item())
        total_batches += 1

    references, hypotheses = generate_predictions(
        model=model,
        loader=loader,
        tgt_vocab=tgt_vocab,
        device=device,
        max_decode_len=max_decode_len,
    )
    metrics = compute_mt_metrics(references, hypotheses)
    avg_loss = total_loss / max(total_batches, 1)
    metrics.update(
        {
            'loss': avg_loss,
            'perplexity': _loss_to_perplexity(avg_loss),
            'num_samples': len(references),
            'references': references,
            'hypotheses': hypotheses,
        }
    )
    return metrics



def save_checkpoint(
    checkpoint_path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: WarmupInverseSqrtScheduler,
    scaler,
    config: Dict[str, Any],
    src_vocab: Vocab,
    tgt_vocab: Vocab,
    epoch: int,
    best_score: float,
) -> None:
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            'model_state': model.state_dict(),
            'optimizer_state': optimizer.state_dict(),
            'scheduler_state': scheduler.state_dict(),
            'scaler_state': scaler.state_dict() if scaler is not None else None,
            'config': config,
            'src_vocab': src_vocab.to_dict(),
            'tgt_vocab': tgt_vocab.to_dict(),
            'epoch': epoch,
            'best_score': best_score,
        },
        checkpoint_path,
    )



def load_model_from_checkpoint(checkpoint_path: str | Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint['config']
    src_vocab = Vocab.from_dict(checkpoint['src_vocab'])
    tgt_vocab = Vocab.from_dict(checkpoint['tgt_vocab'])
    model = build_model(config, src_vocab, tgt_vocab)
    model.load_state_dict(checkpoint['model_state'])
    model.to(device)
    model.eval()
    return model, src_vocab, tgt_vocab, config, checkpoint



def run_training(config: Dict[str, Any]) -> Dict[str, Any]:
    config = copy.deepcopy(config)
    output_dir = ensure_dir(config['runtime']['output_dir'])
    checkpoints_dir = ensure_dir(output_dir / 'checkpoints')
    samples_dir = ensure_dir(output_dir / 'samples')
    cleanup_targets = [
        output_dir / 'train.log',
        output_dir / 'history.csv',
        output_dir / 'train_steps.csv',
        output_dir / 'epoch_metrics.jsonl',
        output_dir / 'resolved_config.json',
        output_dir / 'run_metadata.json',
        output_dir / 'run_summary.md',
        output_dir / 'test_metrics.json',
        output_dir / 'predictions_test.txt',
        output_dir / 'references_test.txt',
        output_dir / 'loss_curve.png',
        output_dir / 'bleu_curve.png',
        output_dir / 'perplexity_curve.png',
        output_dir / 'lr_curve.png',
        output_dir / 'runtime_memory_curve.png',
        checkpoints_dir / 'best.pt',
        checkpoints_dir / 'last.pt',
    ]
    for target in cleanup_targets:
        if target.exists():
            target.unlink()
    for sample_file in samples_dir.glob('*'):
        if sample_file.is_file():
            sample_file.unlink()
    run_log_path = output_dir / 'train.log'
    run_log_file = run_log_path.open('w', encoding='utf-8')

    def log(message: str) -> None:
        print(message)
        run_log_file.write(message + '\n')
        run_log_file.flush()

    set_seed(int(config.get('seed', 42)))
    try:
        device = get_device(config.get('device', 'auto'))
        amp_enabled = bool(config.get('mixed_precision', True) and device.type == 'cuda')
        env_info = get_environment_info(device)
        log(f'[INFO] device = {device}')
        log(f'[INFO] mixed_precision = {amp_enabled}')
        if env_info.get('device_name'):
            log(
                f"[INFO] gpu = {env_info['device_name']} "
                f"(total_mem={float(env_info['device_total_memory_mb']):.1f}MB, capability={env_info['device_capability']})"
            )

        bundle: DatasetBundle = load_dataset_bundle(config)
        model = build_model(config, bundle.src_vocab, bundle.tgt_vocab).to(device)
        param_count = count_parameters(model)
        log(f'[INFO] trainable parameters = {param_count:,}')
        log(
            '[INFO] dataset sizes = '
            f"train:{len(bundle.train_loader.dataset)} valid:{len(bundle.valid_loader.dataset)} test:{len(bundle.test_loader.dataset)}"
        )
        log(
            '[INFO] vocab sizes = '
            f"src:{len(bundle.src_vocab)} tgt:{len(bundle.tgt_vocab)} max_len:{config['data']['max_len']}"
        )

        optimizer, scheduler = build_optimizer_and_scheduler(config, model)
        scaler = create_grad_scaler(device, amp_enabled)
        criterion = _build_criterion(bundle.tgt_vocab.pad_id, config['train'].get('label_smoothing', 0.0))

        resolved_config = copy.deepcopy(config)
        resolved_config['runtime']['device_resolved'] = str(device)
        resolved_config['runtime']['parameter_count'] = int(param_count)
        write_json(resolved_config, output_dir / 'resolved_config.json')

        run_metadata = {
            'start_time': current_timestamp(),
            'environment': env_info,
            'dataset': {
                'train_size': len(bundle.train_loader.dataset),
                'valid_size': len(bundle.valid_loader.dataset),
                'test_size': len(bundle.test_loader.dataset),
                'train_batches': len(bundle.train_loader),
                'valid_batches': len(bundle.valid_loader),
                'test_batches': len(bundle.test_loader),
                'src_vocab_size': len(bundle.src_vocab),
                'tgt_vocab_size': len(bundle.tgt_vocab),
            },
            'artifacts': {
                'resolved_config': str(output_dir / 'resolved_config.json'),
                'history_csv': str(output_dir / 'history.csv'),
                'step_log_csv': str(output_dir / 'train_steps.csv'),
                'epoch_log_jsonl': str(output_dir / 'epoch_metrics.jsonl'),
                'train_log': str(run_log_path),
            },
        }
        write_json(run_metadata, output_dir / 'run_metadata.json')

        history_path = output_dir / 'history.csv'
        step_history_path = output_dir / 'train_steps.csv'
        epoch_log_path = output_dir / 'epoch_metrics.jsonl'
        history_fields = [
            'epoch',
            'train_loss',
            'train_ppl',
            'val_loss',
            'val_ppl',
            'val_bleu',
            'val_chrf',
            'lr',
            'epoch_seconds',
            'peak_memory_mb',
        ]
        step_history_fields = [
            'epoch',
            'step',
            'global_step',
            'avg_train_loss',
            'lr',
            'grad_norm',
            'batch_seconds',
            'src_tokens',
            'tgt_tokens',
            'tokens_per_second',
            'current_memory_mb',
            'peak_memory_mb',
        ]

        best_bleu = float('-inf')
        best_epoch = 0
        patience = int(config['train'].get('patience', 5))
        patience_counter = 0
        grad_accum_steps = int(config['train'].get('grad_accum_steps', 1))
        max_decode_len = int(config['decode'].get('max_decode_len', config['data']['max_len']))
        sample_count = int(config['runtime'].get('sample_count', 8))
        step_log_interval = max(1, int(config['train'].get('log_interval', 50)))
        global_step = 0

        overall_start = time.time()
        for epoch in range(1, int(config['train']['epochs']) + 1):
            epoch_start = time.time()
            model.train()
            total_loss = 0.0
            total_batches = 0
            optimizer.zero_grad(set_to_none=True)
            reset_cuda_peak_memory()

            progress = tqdm(bundle.train_loader, desc=f'Epoch {epoch:02d}', leave=False, disable=not sys.stdout.isatty())
            for step_idx, batch in enumerate(progress, start=1):
                step_start = time.time()
                batch = _move_batch_to_device(batch, device)
                src_tokens = int(batch['src'].ne(bundle.src_vocab.pad_id).sum().item())
                tgt_tokens = int(batch['tgt_out'].ne(bundle.tgt_vocab.pad_id).sum().item())

                with autocast_if_available(device, amp_enabled):
                    logits = model(batch['src'], batch['tgt_in'])
                    vocab_size = logits.size(-1)
                    loss = criterion(logits.reshape(-1, vocab_size), batch['tgt_out'].reshape(-1))
                    loss = loss / grad_accum_steps

                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

                grad_norm = float('nan')
                did_step = step_idx % grad_accum_steps == 0 or step_idx == len(bundle.train_loader)
                if did_step:
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    grad_norm = float(
                        torch.nn.utils.clip_grad_norm_(model.parameters(), config['train'].get('clip_grad_norm', 1.0))
                    )

                    if scaler is not None:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    current_lr = scheduler.step()
                    global_step += 1
                else:
                    current_lr = scheduler.get_lr()

                total_loss += float(loss.item()) * grad_accum_steps
                total_batches += 1
                avg_train_loss = total_loss / max(total_batches, 1)
                progress.set_postfix(loss=f'{avg_train_loss:.4f}', lr=f'{current_lr:.6f}')

                if did_step and (global_step == 1 or global_step % step_log_interval == 0 or step_idx == len(bundle.train_loader)):
                    batch_seconds = time.time() - step_start
                    total_tokens = src_tokens + tgt_tokens
                    tokens_per_second = total_tokens / max(batch_seconds, 1e-9)
                    current_memory_mb = get_cuda_allocated_memory_mb()
                    peak_memory_mb = get_cuda_peak_memory_mb()
                    step_row = {
                        'epoch': epoch,
                        'step': step_idx,
                        'global_step': global_step,
                        'avg_train_loss': avg_train_loss,
                        'lr': current_lr,
                        'grad_norm': grad_norm,
                        'batch_seconds': batch_seconds,
                        'src_tokens': src_tokens,
                        'tgt_tokens': tgt_tokens,
                        'tokens_per_second': tokens_per_second,
                        'current_memory_mb': current_memory_mb,
                        'peak_memory_mb': peak_memory_mb,
                    }
                    append_csv_row(step_history_path, step_history_fields, step_row)
                    log(
                        f"[STEP e{epoch:02d} s{step_idx:04d}/{len(bundle.train_loader):04d}] "
                        f"avg_loss={avg_train_loss:.4f} lr={current_lr:.6f} grad_norm={grad_norm:.3f} "
                        f"tok/s={tokens_per_second:.1f} src_tok={src_tokens} tgt_tok={tgt_tokens} "
                        f"mem={current_memory_mb:.1f}/{peak_memory_mb:.1f}MB"
                    )

            train_loss = total_loss / max(total_batches, 1)
            train_ppl = _loss_to_perplexity(train_loss)
            val_metrics = evaluate_loader(
                model=model,
                loader=bundle.valid_loader,
                criterion=criterion,
                tgt_vocab=bundle.tgt_vocab,
                device=device,
                max_decode_len=max_decode_len,
            )
            val_bleu = float(val_metrics['bleu'])
            epoch_seconds = time.time() - epoch_start
            peak_memory_mb = get_cuda_peak_memory_mb()

            epoch_row = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_ppl': train_ppl,
                'val_loss': val_metrics['loss'],
                'val_ppl': val_metrics['perplexity'],
                'val_bleu': val_bleu,
                'val_chrf': val_metrics['chrf'],
                'lr': scheduler.get_lr(),
                'epoch_seconds': epoch_seconds,
                'peak_memory_mb': peak_memory_mb,
            }
            append_csv_row(history_path, history_fields, epoch_row)
            append_jsonl_row(
                epoch_log_path,
                {
                    **epoch_row,
                    'timestamp': current_timestamp(),
                    'val_token_accuracy': float(val_metrics['token_accuracy']),
                    'val_exact_match': float(val_metrics['exact_match']),
                },
            )

            valid_samples = collect_translation_samples(
                model=model,
                loader=bundle.valid_loader,
                src_vocab=bundle.src_vocab,
                tgt_vocab=bundle.tgt_vocab,
                device=device,
                max_decode_len=max_decode_len,
                sample_count=sample_count,
            )
            epoch_sample_path = samples_dir / f'valid_epoch_{epoch:02d}.txt'
            _write_sample_records(epoch_sample_path, valid_samples)

            log(
                f"[EPOCH {epoch:02d}] train_loss={train_loss:.4f} train_ppl={train_ppl:.2f} "
                f"val_loss={val_metrics['loss']:.4f} val_bleu={val_bleu:.2f} val_chrf={float(val_metrics['chrf']):.2f} "
                f"val_token_acc={float(val_metrics['token_accuracy']):.2f} peak_mem={peak_memory_mb:.1f}MB "
                f"time={format_seconds(epoch_seconds)} samples={epoch_sample_path.name}"
            )

            save_checkpoint(
                checkpoints_dir / 'last.pt',
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                bundle.src_vocab,
                bundle.tgt_vocab,
                epoch,
                best_bleu,
            )

            if val_bleu > best_bleu:
                best_bleu = val_bleu
                best_epoch = epoch
                patience_counter = 0
                save_checkpoint(
                    checkpoints_dir / 'best.pt',
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    bundle.src_vocab,
                    bundle.tgt_vocab,
                    epoch,
                    best_bleu,
                )
                best_samples_path = samples_dir / 'best_valid_samples.txt'
                _write_sample_records(best_samples_path, valid_samples)
                log(f'[INFO] new best checkpoint saved at epoch {epoch} (val_bleu={val_bleu:.2f})')
            else:
                patience_counter += 1
                log(f'[INFO] patience {patience_counter}/{patience}')
                if patience_counter >= patience:
                    log(f'[INFO] Early stopping triggered at epoch {epoch}.')
                    break

        train_seconds = time.time() - overall_start
        best_checkpoint = checkpoints_dir / 'best.pt'
        best_model, _, tgt_vocab, _, _ = load_model_from_checkpoint(best_checkpoint, device)
        test_metrics = evaluate_loader(
            model=best_model,
            loader=bundle.test_loader,
            criterion=criterion,
            tgt_vocab=tgt_vocab,
            device=device,
            max_decode_len=max_decode_len,
        )

        pred_path = output_dir / 'predictions_test.txt'
        ref_path = output_dir / 'references_test.txt'
        pred_path.write_text('\n'.join(test_metrics['hypotheses']) + '\n', encoding='utf-8')
        ref_path.write_text('\n'.join(test_metrics['references']) + '\n', encoding='utf-8')

        test_samples = collect_translation_samples(
            model=best_model,
            loader=bundle.test_loader,
            src_vocab=bundle.src_vocab,
            tgt_vocab=tgt_vocab,
            device=device,
            max_decode_len=max_decode_len,
            sample_count=sample_count,
        )
        test_samples_path = samples_dir / 'test_samples.txt'
        _write_sample_records(test_samples_path, test_samples)

        plot_paths = [str(path) for path in plot_training_curves(history_path)]

        metrics_to_save = {
            'best_epoch': best_epoch,
            'best_val_bleu': best_bleu,
            'test_loss': test_metrics['loss'],
            'test_perplexity': test_metrics['perplexity'],
            'test_bleu': test_metrics['bleu'],
            'test_chrf': test_metrics['chrf'],
            'test_token_accuracy': test_metrics['token_accuracy'],
            'test_exact_match': test_metrics['exact_match'],
            'metrics_backend': test_metrics['metrics_backend'],
            'total_training_seconds': train_seconds,
            'parameter_count': param_count,
            'device': str(device),
        }
        metrics_path = output_dir / 'test_metrics.json'
        write_json(metrics_to_save, metrics_path)

        run_metadata.update(
            {
                'end_time': current_timestamp(),
                'best_epoch': best_epoch,
                'best_val_bleu': best_bleu,
                'total_training_seconds': train_seconds,
                'test_metrics': metrics_to_save,
                'artifacts': {
                    **run_metadata['artifacts'],
                    'test_metrics': str(metrics_path),
                    'predictions_test': str(pred_path),
                    'references_test': str(ref_path),
                    'test_samples': str(test_samples_path),
                    'plot_paths': plot_paths,
                },
            }
        )
        write_json(run_metadata, output_dir / 'run_metadata.json')

        summary_lines = [
            '# Run Summary',
            '',
            '## Config',
            '',
            f"- output_dir: {output_dir}",
            f"- device: {device}",
            f"- mixed_precision: {amp_enabled}",
            f"- d_model: {config['model']['d_model']}",
            f"- nhead: {config['model']['nhead']}",
            f"- encoder_layers: {config['model']['num_encoder_layers']}",
            f"- decoder_layers: {config['model']['num_decoder_layers']}",
            f"- dim_feedforward: {config['model']['dim_feedforward']}",
            f"- position_type: {config['model']['position_type']}",
            f"- batch_size: {config['train']['batch_size']}",
            f"- grad_accum_steps: {config['train']['grad_accum_steps']}",
            f"- epochs: {config['train']['epochs']}",
            f"- label_smoothing: {config['train'].get('label_smoothing', 0.0)}",
            '',
            '## Dataset',
            '',
            f"- train_size: {len(bundle.train_loader.dataset)}",
            f"- valid_size: {len(bundle.valid_loader.dataset)}",
            f"- test_size: {len(bundle.test_loader.dataset)}",
            f"- src_vocab_size: {len(bundle.src_vocab)}",
            f"- tgt_vocab_size: {len(bundle.tgt_vocab)}",
            '',
            '## Results',
            '',
            f"- best_epoch: {best_epoch}",
            f"- best_val_bleu: {best_bleu:.2f}",
            f"- test_bleu: {float(test_metrics['bleu']):.2f}",
            f"- test_chrf: {float(test_metrics['chrf']):.2f}",
            f"- test_token_accuracy: {float(test_metrics['token_accuracy']):.2f}",
            f"- test_exact_match: {float(test_metrics['exact_match']):.2f}",
            f"- total_training_seconds: {train_seconds:.2f}",
            f"- parameter_count: {param_count}",
            '',
            '## Artifacts',
            '',
            f"- train_log: {run_log_path.name}",
            f"- history_csv: {history_path.name}",
            f"- step_log_csv: {step_history_path.name}",
            f"- epoch_log_jsonl: {epoch_log_path.name}",
            f"- run_metadata: run_metadata.json",
            f"- test_metrics: {metrics_path.name}",
            f"- test_samples: {test_samples_path.relative_to(output_dir)}",
            f"- plots: {', '.join(Path(path).name for path in plot_paths)}",
        ]
        run_summary_path = output_dir / 'run_summary.md'
        run_summary_path.write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')

        log(
            f"[TEST] bleu={float(test_metrics['bleu']):.2f} chrf={float(test_metrics['chrf']):.2f} "
            f"token_acc={float(test_metrics['token_accuracy']):.2f} exact={float(test_metrics['exact_match']):.2f}"
        )
        log(f'[DONE] run summary written to {run_summary_path}')

        return {
            'output_dir': str(output_dir),
            'best_checkpoint': str(best_checkpoint),
            'test_metrics_path': str(metrics_path),
            'run_summary_path': str(run_summary_path),
            'train_log_path': str(run_log_path),
            **metrics_to_save,
        }
    finally:
        run_log_file.close()
