"""Тесты модуля драйверов переходов (research/31): (а) rolling-origin фолды не
пересекаются с будущим (эмбарго h по индексам), (б) на синтетике с известным
драйвером модель находит его в топ-3 SHAP, (в) isotonic-калибровка монотонна,
(г) детерминизм по seed, (д) частота таргета соответствует events.parquet (±20%)."""
import json
import os

import numpy as np
import pandas as pd
import pytest

from ecotypes import drivers as drv

DYN_PRESENT = os.path.exists("data/processed/dynamics/labels.parquet") and \
    os.path.exists("data/processed/dynamics/events.parquet")
requires_dynamics = pytest.mark.skipif(not DYN_PRESENT,
                                       reason="нет processed-динамики (CI-режим)")

MONTHS = [str(p) for p in pd.period_range("2023-01", "2024-12", freq="M")]


def _params() -> drv.DriversParams:
    p = drv.DriversParams()
    p.lgbm = dict(p.lgbm, n_estimators=120)
    return p


# (а) rolling-origin: train строго ≤ v−h−1, val не в train, счёт фолдов 12/10
def test_folds_embargo_and_counts():
    for h, usable, n_folds in ((1, list(range(6, 23)), 12),
                               (3, list(range(6, 21)), 10)):
        folds = drv.rolling_origin_folds(usable, h, first_val_idx=11)
        assert len(folds) == n_folds
        for train, v in folds:
            assert v not in train
            assert max(train) <= v - h - 1          # эмбарго h месяцев
            assert min(train) == usable[0]          # expanding window
        vals = [v for _, v in folds]
        assert vals == sorted(vals) and len(set(vals)) == len(vals)


# (а2) регрессионный тест: mirror-рёбра несут чужой rank → слоты коллидировали,
# узлы теряли ближайших соседей; knn_slots обязан дать k ближайших без потерь
def test_knn_slots_no_rank_collision():
    edges = pd.DataFrame([
        # own-рёбра узла 1: ранги 1..3
        (1, 2, 10.0, 1), (1, 3, 20.0, 2), (1, 4, 30.0, 3),
        # mirror-рёбра с теми же rank, но бóльшими dist — коллизия слотов
        (1, 5, 40.0, 1), (1, 6, 50.0, 2),
        # own узла 3 + mirror от (1,3) с унаследованным rank=2
        (3, 2, 5.0, 1), (3, 1, 20.0, 2),
    ], columns=["tid_x", "tid_y", "dist_km", "rank"])
    tids = np.array([1, 2, 3, 4, 5, 6])
    nbr = drv.knn_slots(edges, tids, k=3)
    # у узла 1 три ближайших: 2 (10 км), 3 (20 км), 4 (30 км) — не 5/6
    assert set(nbr[0]) == {1, 2, 3}          # позиции tid 2,3,4
    assert (nbr[0] >= 0).all()               # ни один слот не потерян
    # у узла 3 два соседа: 2 (5 км) и 1 (20 км); третий слот = −1
    assert set(nbr[2][nbr[2] >= 0]) == {0, 1}


def _synthetic_xy(seed: int = 0, n: int = 4000):
    """Известный драйвер: logit(y) = 2.5·sig_driver + 0.3·noise_f1."""
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 8)),
                     columns=["sig_driver", "noise_f1"] +
                             [f"noise_f{i}" for i in range(2, 8)])
    p = 1.0 / (1.0 + np.exp(-(2.5 * X["sig_driver"] + 0.3 * X["noise_f1"])))
    y = rng.binomial(1, p).astype(float)
    return X, y


# (б) синтетика с известным драйвером → он в топ-3 по mean|SHAP|
def test_shap_finds_known_driver():
    X, y = _synthetic_xy()
    par = _params()
    m = drv.fit_lgbm(X.iloc[:3000], y[:3000], seed=7, par=par)
    sv = drv.shap_values(m, X.iloc[3000:])
    prof = drv.shap_profile(sv, list(X.columns))
    assert "sig_driver" in prof.head(3).feature.tolist()


# (в) isotonic-калибровка монотонна
def test_isotonic_monotone():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 2000)
    y = rng.binomial(1, 1 / (1 + np.exp(-4 * (p - 0.5)))).astype(float)
    iso = drv.fit_isotonic(p, y)
    grid = np.linspace(0, 1, 200)
    pred = iso.predict(grid)
    assert np.all(np.diff(pred) >= -1e-12)


# (г) детерминизм: два запуска fit/predict с одним seed → идентичные вероятности
def test_determinism_same_seed():
    X, y = _synthetic_xy()
    par = _params()
    p1 = drv.fit_lgbm(X, y, seed=42, par=par).predict_proba(X)[:, 1]
    p2 = drv.fit_lgbm(X, y, seed=42, par=par).predict_proba(X)[:, 1]
    assert np.array_equal(p1, p2)
    f1 = drv.rolling_origin_folds(list(range(6, 23)), 1, 11)
    f2 = drv.rolling_origin_folds(list(range(6, 23)), 1, 11)
    assert f1 == f2


# (д) частота y1 соответствует events.parquet (±20%) на реальных метках
@requires_dynamics
def test_target_frequency_matches_events():
    root = "data/processed"
    labels = pd.read_parquet(f"{root}/dynamics/labels.parquet")
    events = pd.read_parquet(f"{root}/dynamics/events.parquet")
    registry = pd.read_parquet(f"{root}/dynamics/type_registry.parquet")
    with open(f"{root}/dynamics/stability.json", encoding="utf-8") as f:
        stability = json.load(f)
    flagged = drv.flagged_pair_months(stability)
    months = sorted(labels.month.unique())
    targets, _ = drv.make_targets(labels, registry, flagged, months,
                                  first_usable_idx=6)
    y1 = int(targets.y1.sum())
    midx = {m: i for i, m in enumerate(months)}
    ev_months = [months[i + 1] for i in
                 targets[targets.y1.notna()].month_t.map(midx).unique()]
    n_ev = int(events[events.month.isin(ev_months)].shape[0])
    assert abs(y1 - n_ev) <= 0.2 * n_ev, f"y1={y1} vs events={n_ev}"
