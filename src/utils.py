from __future__ import annotations

import contextlib
import csv
import json
import os
import platform
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List

import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(obj: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def read_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def append_csv_row(path: str | Path, fieldnames: List[str], row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open('a', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def append_jsonl_row(path: str | Path, row: Dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(row, ensure_ascii=False) + '\n')


def read_csv_rows(path: str | Path) -> List[Dict[str, str]]:
    path = Path(path)
    with path.open('r', encoding='utf-8', newline='') as f:
        reader = csv.DictReader(f)
        return list(reader)


def get_device(device_name: str = 'auto') -> torch.device:
    if device_name != 'auto':
        return torch.device(device_name)
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def format_seconds(seconds: float) -> str:
    seconds = int(seconds)
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f'{h:d}:{m:02d}:{s:02d}'
    return f'{m:02d}:{s:02d}'


def count_parameters(model: torch.nn.Module) -> int:
    seen = set()
    total = 0
    for param in model.parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen:
            continue
        seen.add(id(param))
        total += param.numel()
    return total


def get_cuda_peak_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def get_cuda_allocated_memory_mb() -> float:
    if not torch.cuda.is_available():
        return 0.0
    return torch.cuda.memory_allocated() / (1024 ** 2)


def reset_cuda_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


@contextlib.contextmanager
def autocast_if_available(device: torch.device, enabled: bool) -> Iterator[None]:
    if enabled and device.type == 'cuda':
        try:
            with torch.amp.autocast('cuda'):
                yield
        except AttributeError:
            with torch.cuda.amp.autocast():
                yield
    else:
        yield


def create_grad_scaler(device: torch.device, enabled: bool):
    if not enabled or device.type != 'cuda':
        return None
    try:
        return torch.amp.GradScaler('cuda', enabled=True)
    except AttributeError:
        return torch.cuda.amp.GradScaler(enabled=True)
    except TypeError:
        return torch.cuda.amp.GradScaler(enabled=True)


def load_text_lines(path: str | Path) -> List[str]:
    with Path(path).open('r', encoding='utf-8') as f:
        return [line.rstrip('\n') for line in f]


def write_text_lines(lines: Iterable[str], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        for line in lines:
            f.write(str(line).rstrip('\n') + '\n')


def pretty_float(value: float, digits: int = 4) -> str:
    return f'{value:.{digits}f}'


def current_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec='seconds')


def get_environment_info(device: torch.device) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        'python_version': sys.version.split()[0],
        'platform': platform.platform(),
        'torch_version': torch.__version__,
        'torch_cuda_version': torch.version.cuda,
        'cuda_available': torch.cuda.is_available(),
        'device_resolved': str(device),
    }
    try:
        info['cudnn_version'] = torch.backends.cudnn.version()
    except Exception:
        info['cudnn_version'] = None

    if device.type == 'cuda' and torch.cuda.is_available():
        device_index = device.index if device.index is not None else torch.cuda.current_device()
        props = torch.cuda.get_device_properties(device_index)
        info.update(
            {
                'device_name': props.name,
                'device_total_memory_mb': round(props.total_memory / (1024 ** 2), 2),
                'device_capability': f'{props.major}.{props.minor}',
            }
        )
    return info
