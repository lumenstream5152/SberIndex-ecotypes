"""Тесты движка описаний (research/33): инвариант Миркина B+W=T на mini_panel;
топ-признаки = истинные драйверы на чистой гауссовой синтетике; fidelity
суррогатного дерева; согласие трёх описателей; полнота паспорта (10 полей);
bootstrap recall@5; детерминизм; smoke permutation-fallback SHAP."""
import numpy as np
import pandas as pd
import pytest

from ecotypes.benchmark import modal_labels
from ecotypes.interpret import (PASSPORT_FIELDS, agreement_table,
                                bootstrap_stability, build_passports,
                                mirkin_decomposition, mirkin_profiles,
                                shap_profiles, surrogate_tree,
                                _permutation_profiles)

FEAT = [f"f{i}" for i in range(6)]


@pytest.fixture(scope="module")
def gauss() -> dict:
    """Чистая синтетика с известными драйверами: K=3, p=6, n=150 на тип.
    У типа k два сильных драйвера (3.4σ и 2.6σ), остальные признаки — слабые
    несимметричные сдвиги (ранжирование устойчиво к перевыборке → recall@5).
    Топ-2 драйвера: тип 0 → {f0,f1}, тип 1 → {f2,f3}, тип 2 → {f4,f5}."""
    rng = np.random.default_rng(7)
    mus = [
        [3.4, 2.6, 0.00, 0.10, 0.20, 0.30],
        [0.30, 0.20, 3.4, 2.6, 0.00, 0.10],
        [0.10, 0.00, 0.20, 0.30, 3.4, 2.6],
    ]
    Xs, ls = [], []
    for k, mu in enumerate(mus):
        Xs.append(rng.normal(np.array(mu), 1.0, size=(150, 6)))
        ls += [k] * 150
    X_raw = np.vstack(Xs)
    labels = np.array(ls)
    mu, sd = X_raw.mean(0), X_raw.std(0)
    return dict(X=(X_raw - mu) / sd, X_raw=X_raw, labels=labels, feat=FEAT,
                means=mu, stds=sd,
                drivers={0: {"f0", "f1"}, 1: {"f2", "f3"}, 2: {"f4", "f5"}})


# (а) инвариант Миркина B + W = T на mini_panel (синтетика PP-Dir из conftest)
def test_mirkin_scatter_identity(mini_panel):
    Xc = mini_panel["X"].mean(axis=0)  # (n, 6) средние доли
    Xc = np.log(np.clip(Xc, 1e-9, None))
    X = (Xc - Xc.mean(0)) / Xc.std(0)
    labels = modal_labels(mini_panel["z"])
    B, rel, sign, W, T = mirkin_decomposition(X, labels)
    assert abs(B.sum() + W - T) < 1e-6 * T
    assert np.isfinite(rel).all()


# (б) топ-признаки по Миркину = истинные драйверы генератора
def test_mirkin_top_features_are_drivers(gauss):
    mk = mirkin_profiles(gauss["X"], gauss["labels"], gauss["feat"],
                         X_raw=gauss["X_raw"])
    for k, drivers in gauss["drivers"].items():
        top2 = set(mk[(mk.type_id == k) & (mk.rank_in_type <= 2)].feature)
        assert top2 == drivers, f"тип {k}: топ-2 {top2} ≠ истинные {drivers}"


# (в) суррогатное дерево: fidelity > 0.9 на чистой синтетике (и per class)
def test_surrogate_tree_fidelity(gauss):
    tr = surrogate_tree(gauss["X"], gauss["labels"], gauss["feat"],
                        max_depth=4, min_samples_leaf=20, seed=0,
                        feature_means=gauss["means"],
                        feature_stds=gauss["stds"])
    assert tr["fidelity"] > 0.9
    assert all(v > 0.9 for v in tr["fidelity_per_class"].values())
    assert all(r["rule"] for r in tr["rules"].values())


# (г) согласие трёх описателей высокое на чистой синтетике
def test_agreement_high(gauss):
    mk = mirkin_profiles(gauss["X"], gauss["labels"], gauss["feat"])
    tr = surrogate_tree(gauss["X"], gauss["labels"], gauss["feat"], seed=0)
    sh = shap_profiles(gauss["X"], gauss["labels"], gauss["feat"], seed=0)
    agr = agreement_table(mk.pivot(index="type_id", columns="feature",
                                   values="rel")[gauss["feat"]],
                          tr["class_importances"], sh["matrix"], gauss["feat"])
    assert (agr["summary"].agreement_score >= 0.6).all(), agr["summary"]


# (д) паспорт содержит все 10 полей спеки 33 §5
def test_passport_fields(gauss):
    rng = np.random.default_rng(3)
    n = len(gauss["labels"])
    nodes = pd.DataFrame({
        "territory_id": np.arange(1, n + 1, dtype=np.int32),
        "name": [f"МО {i}" for i in range(n)],
        "region_name": rng.choice(["Татарстан", "Краснодарский край",
                                   "ЯНАО"], n),
        "mun_type": rng.choice(["городской округ", "муниципальный район"], n),
        "pop_2024": rng.integers(2_000, 500_000, n),
    })
    pp = build_passports(gauss["X"], gauss["X_raw"], gauss["feat"],
                         gauss["labels"], nodes, layer="test")
    assert set(PASSPORT_FIELDS) <= set(pp.columns)
    for f in PASSPORT_FIELDS:
        assert pp[f].notna().all(), f"поле {f} пусто"
    assert len(pp) == 3


# (е) bootstrap-устойчивость описаний: recall@5 ≥ 0.9 на чистой синтетике
def test_bootstrap_recall(gauss):
    bt = bootstrap_stability(gauss["X"], gauss["labels"], gauss["feat"],
                             B=30, top=5, seed=0)
    mir = bt[bt.descriptor == "mirkin"]
    assert len(mir) == 3
    assert (mir.recall_at_top >= 0.9).all(), mir


# (ж) детерминизм: два прогона описателей → идентичные выходы
def test_determinism(gauss):
    a = mirkin_profiles(gauss["X"], gauss["labels"], gauss["feat"])
    b = mirkin_profiles(gauss["X"], gauss["labels"], gauss["feat"])
    pd.testing.assert_frame_equal(a, b)
    t1 = surrogate_tree(gauss["X"], gauss["labels"], gauss["feat"], seed=0)
    t2 = surrogate_tree(gauss["X"], gauss["labels"], gauss["feat"], seed=0)
    assert t1["text"] == t2["text"]
    pd.testing.assert_frame_equal(t1["class_importances"],
                                  t2["class_importances"])
    s1 = shap_profiles(gauss["X"], gauss["labels"], gauss["feat"], seed=0)
    s2 = shap_profiles(gauss["X"], gauss["labels"], gauss["feat"], seed=0)
    assert s1["fidelity"] == s2["fidelity"]
    pd.testing.assert_frame_equal(s1["matrix"], s2["matrix"])


# smoke: запасной путь §3.4 (permutation one-vs-rest) — форма и конечность
def test_permutation_fallback_smoke(gauss):
    sl = slice(0, 300)  # типы 0 и 1
    classes = np.unique(gauss["labels"][sl])
    prof = _permutation_profiles(gauss["X"][sl], gauss["labels"][sl],
                                 classes, gauss["feat"], seed=0, n_repeats=3)
    assert prof.shape == (len(classes), 6)
    assert np.isfinite(prof).all()
