from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional


def load_json(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _set_by_dot_path(cfg: MutableMapping[str, Any], dot_path: str, value: Any) -> None:
    keys = dot_path.split('.')
    current: MutableMapping[str, Any] = cfg
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value


def apply_overrides(config: Dict[str, Any], overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    resolved = copy.deepcopy(config)
    if not overrides:
        return resolved
    for dot_path, value in overrides.items():
        _set_by_dot_path(resolved, dot_path, value)
    return resolved


def load_config(config_path: str | Path, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    config = load_json(config_path)
    return apply_overrides(config, overrides)
