"""Тесты модуля ICVI (research/32: формулы сверены с Pattern до 1e-16,
регрессионный эталон — ручной пример 6 узлов, зашит в спеку)."""
from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse
from sklearn.metrics import calinski_harabasz_score, silhouette_score

from ecotypes import icvi

ATOL = 1e-12  # допуск на float-округление в суммах при перестановках


# ---------------------------------------------------------------- helpers

def random_instance(rng: np.random.Generator, n: int = 24, d: int = 4, k: int = 3):
    """Случайные малые X/A/labels: X ~ N(0,1); A симметричная взвешенная,
    diag(A) = 0; все k кластеров представлены (S_Dbw требует k ≥ 2)."""
    X = rng.normal(size=(n, d))
    labels = rng.integers(0, k, n)
    labels[:k] = np.arange(k)
    W = rng.random((n, n)) * (rng.random((n, n)) < 0.35)
    A = np.triu(W, 1)
    A = A + A.T
    return X, A, labels


def attribute_panel(X: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "SW": icvi.sw(X, labels),
        "CH_over_N": icvi.ch_over_n(X, labels),
        "S_Dbw": icvi.sdbw(X, labels),
    }


def network_panel(A: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return {
        "AVI": icvi.avi(A, labels),
        "AVU": icvi.avu(A, labels),
        "ANUI": icvi.anui(A, labels),
        "Q": icvi.q_modularity(A, labels),
        "MQ_turbomq": icvi.mq(A, labels, variant="turbomq"),
        "MQ_mancoridis": icvi.mq(A, labels, variant="mancoridis"),
    }


def assert_panels_close(base: dict[str, float], other: dict[str, float]) -> None:
    assert base.keys() == other.keys()
    for name, v in base.items():
        assert np.isclose(v, other[name], atol=ATOL, rtol=1e-9), (
            f"{name}: {v!r} vs {other[name]!r}"
        )


def manual_graph():
    """Два треугольника K3 + мост (1,3); labels = [0,0,0,1,1,1]; S = [[6,1],[1,6]]."""
    A = np.zeros((6, 6))
    for i, j in [(0, 1), (1, 2), (0, 2), (3, 4), (4, 5), (3, 5), (1, 3)]:
        A[i, j] = A[j, i] = 1.0
    labels = np.array([0, 0, 0, 1, 1, 1])
    return A, labels


# ---------------------------------------------------------------- (а) инвариантности

@pytest.mark.parametrize("run", range(20))
def test_label_renaming_invariance(run: int):
    rng = np.random.default_rng(10_000 + run)
    X, A, labels = random_instance(rng)
    k = len(np.unique(labels))
    labels_renamed = rng.permutation(k)[labels]
    assert_panels_close(
        {**attribute_panel(X, labels), **network_panel(A, labels)},
        {**attribute_panel(X, labels_renamed), **network_panel(A, labels_renamed)},
    )


@pytest.mark.parametrize("run", range(20))
def test_node_permutation_invariance(run: int):
    rng = np.random.default_rng(20_000 + run)
    X, A, labels = random_instance(rng)
    p = rng.permutation(labels.shape[0])
    X_p, labels_p, A_p = X[p], labels[p], A[np.ix_(p, p)]
    assert_panels_close(
        {**attribute_panel(X, labels), **network_panel(A, labels)},
        {**attribute_panel(X_p, labels_p), **network_panel(A_p, labels_p)},
    )


# ---------------------------------------------------------------- (б) sklearn-эталон

def test_sw_ch_match_sklearn_to_the_last_digit():
    rng = np.random.default_rng(7)
    X, _, labels = random_instance(rng, n=60, d=5, k=4)
    assert icvi.sw(X, labels) == silhouette_score(X, labels)
    assert icvi.ch_over_n(X, labels) == calinski_harabasz_score(X, labels) / X.shape[0]


# ---------------------------------------------------------------- (в) ручной пример, спека 32

def test_manual_six_nodes_matches_spec32():
    A, labels = manual_graph()
    assert icvi.avi(A, labels) == pytest.approx(6 / 7, abs=1e-15)
    assert icvi.avu(A, labels) == pytest.approx(1.0, abs=1e-15)
    assert icvi.anui(A, labels) == pytest.approx(6 / 13, abs=1e-15)
    assert icvi.q_modularity(A, labels) == pytest.approx(5 / 14, abs=1e-15)
    assert icvi.mq(A, labels, variant="mancoridis") == pytest.approx(5 / 18, abs=1e-15)
    assert icvi.mq(A, labels, variant="turbomq") == pytest.approx(12 / 7, abs=1e-15)


def test_mq_mancoridis_single_cluster_is_a1():
    A, _ = manual_graph()
    labels = np.zeros(6, dtype=int)  # k = 1: MQ = μ₁/N₁² = 7/36
    assert icvi.mq(A, labels, variant="mancoridis") == pytest.approx(7 / 36, abs=1e-15)


def test_anui_zero_avi_gives_zero_like_pattern():
    # Граф без внутренних рёбер (двудольный): AVI = 0 → ANUI = 0 (дословно Pattern).
    A = np.zeros((4, 4))
    for i, j in [(0, 2), (1, 3)]:
        A[i, j] = A[j, i] = 1.0
    labels = np.array([0, 0, 1, 1])
    assert icvi.avi(A, labels) == 0.0
    assert icvi.anui(A, labels) == 0.0


def test_sparse_adjacency_matches_dense():
    A, labels = manual_graph()
    A_sp = sparse.csr_matrix(A)
    dense = network_panel(A, labels)
    sparse_panel = network_panel(A_sp, labels)
    for name, v in dense.items():
        assert v == pytest.approx(sparse_panel[name], abs=1e-15), name


# ---------------------------------------------------------------- (г) границы

@pytest.mark.parametrize("run", range(10))
def test_bounds(run: int):
    rng = np.random.default_rng(30_000 + run)
    X, A, labels = random_instance(rng, n=30, d=4, k=int(rng.integers(2, 5)))
    k = len(np.unique(labels))
    assert -1.0 - ATOL <= icvi.sw(X, labels) <= 1.0 + ATOL
    assert 0.0 - ATOL <= icvi.mq(A, labels, variant="turbomq") <= k + ATOL
    assert 0.0 - ATOL <= icvi.avu(A, labels) <= (k - 1) + ATOL


# ---------------------------------------------------------------- (д) permutation_baseline

def test_permutation_baseline_ideal_structure_z_gt_2():
    rng = np.random.default_rng(42)
    k, n_per = 4, 50
    centers = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]])
    labels = np.repeat(np.arange(k), n_per)
    X = centers[labels] + 0.1 * rng.normal(size=(k * n_per, 2))
    res = icvi.permutation_baseline(icvi.sw, X, labels=labels, runs=100, rng=rng)
    assert set(res) == {"z", "percentile", "null_mean", "null_std"}
    assert res["z"] > 2
    assert res["percentile"] == 1.0
    assert res["null_std"] > 0
    assert res["null_mean"] < icvi.sw(X, labels)


def test_permutation_baseline_random_labels_abs_z_lt_2():
    rng = np.random.default_rng(43)
    n = 200
    X = rng.normal(size=(n, 4))
    labels = rng.integers(0, 4, n)
    res = icvi.permutation_baseline(icvi.sw, X, labels=labels, runs=100, rng=rng)
    assert abs(res["z"]) < 2
    assert 0.0 <= res["percentile"] <= 1.0


# ---------------------------------------------------------------- (е) sdbw — обёртка пакета

def test_sdbw_matches_package_direct_call():
    from s_dbw import S_Dbw

    rng = np.random.default_rng(5)
    X, _, labels = random_instance(rng, n=40, d=3, k=3)
    direct = S_Dbw(X, labels, alg_noise="bind")
    assert np.isfinite(direct)
    assert icvi.sdbw(X, labels) == direct


# ---------------------------------------------------------------- compute_all

def test_compute_all_defaults_and_cfg_override():
    rng = np.random.default_rng(11)
    X, A, labels = random_instance(rng, n=30, d=3, k=3)
    full = icvi.compute_all(X, A, labels)
    assert set(full) == {"SW", "CH_over_N", "S_Dbw", "AVI", "AVU", "MQ"}
    assert all(np.isfinite(v) for v in full.values())
    assert full["MQ"] == icvi.mq(A, labels, variant="mancoridis")  # дефолт prereg.yaml

    part = icvi.compute_all(
        None, A, labels, cfg_icvi={"panel": ["AVI", "MQ", "ANUI"], "mq_variant": "mancoridis"}
    )
    assert set(part) == {"AVI", "MQ", "ANUI"}
    assert part["MQ"] == icvi.mq(A, labels, variant="mancoridis")

    with pytest.raises(ValueError, match="неизвестный индекс"):
        icvi.compute_all(X, A, labels, cfg_icvi={"panel": ["NOPE"]})
