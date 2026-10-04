import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, Iterable

import numpy as np
import torch


def setup_logging(output_dir: str) -> logging.Logger:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("clin2pro")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(path / "pipeline.log", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(value)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError("Not JSON serializable: %r" % (value,))


def atomic_torch_save(payload: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


class EarlyStopping:
    def __init__(self, patience: int) -> None:
        self.patience = patience
        self.best = float("inf")
        self.bad_epochs = 0

    def update(self, value: float) -> bool:
        if value < self.best - 1e-8:
            self.best = value
            self.bad_epochs = 0
            return True
        self.bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.bad_epochs >= self.patience


def parameter_groups(
    modules: Iterable[torch.nn.Module],
    base_lr: float,
    encoder: torch.nn.Module,
    encoder_lr: float,
    weight_decay: float,
):
    encoder_ids = {id(p) for p in encoder.parameters()}
    encoder_params = list(encoder.parameters())
    other_params = []
    seen = set(encoder_ids)
    for module in modules:
        for parameter in module.parameters():
            if id(parameter) not in seen:
                other_params.append(parameter)
                seen.add(id(parameter))
    return [
        {"params": encoder_params, "lr": encoder_lr, "weight_decay": weight_decay},
        {"params": other_params, "lr": base_lr, "weight_decay": weight_decay},
    ]
