"""Общие фикстуры. Мини-панель генерируется синтетикой PP-Dir —
CI не тянет сырые данные; requires_raw помечает тесты, которым сырые нужны."""
import os

import pytest

RAW_PRESENT = os.path.exists("data/raw/hackathon/hackathonlicence/consumption.parquet")

requires_raw = pytest.mark.skipif(not RAW_PRESENT, reason="нет сырых данных (CI-режим)")


@pytest.fixture(scope="session")
def mini_panel() -> dict:
    """PP-Dir мини-панель: 60 МО × 24 мес, K=3, seed=0 — лёгкая панель
    с известной истиной для остальных тестов проекта."""
    from ecotypes.synthetic import make_ppdataset

    return make_ppdataset(seed=0, K=3, n=60, T=24)
