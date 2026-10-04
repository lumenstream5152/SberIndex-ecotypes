"""Общие фикстуры. Мини-панель генерируется синтетикой PP-Dir —
CI не тянет сырые данные (фикстура регистрируется в test_synthetic.py-сессии;
здесь — только пути и пропуски)."""
import os

import pytest

RAW_PRESENT = os.path.exists("data/raw/hackathon/hackathonlicence/consumption.parquet")

requires_raw = pytest.mark.skipif(not RAW_PRESENT, reason="нет сырых данных (CI-режим)")
