"""Seed-политика: один глобальный seed, производные на этап через sha256.
Смена seed одного этапа не перетасовывает остальные."""
from __future__ import annotations

import hashlib
import random

import numpy as np


def set_all_seeds(seed: int) -> np.random.Generator:
    """Первая строка каждого раннера. Возвращает Generator для нового кода;
    глобальные RandomState тоже фиксируются — часть библиотек читает их."""
    random.seed(seed)
    np.random.seed(seed % (2**32))
    return np.random.default_rng(seed)


def stage_seed(seed: int, stage: str) -> int:
    """Производный seed этапа: sha256("{seed}:{stage}")[:8] → int."""
    h = hashlib.sha256(f"{seed}:{stage}".encode()).hexdigest()
    return int(h[:8], 16)
