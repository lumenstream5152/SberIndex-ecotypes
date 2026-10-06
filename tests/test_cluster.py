"""Тесты кластерного этапа (research/25/26/27): Leiden-ядро на planted partition,
CSPA-консенсус (детерминизм + доминирование), плато-правило, X_static (форма,
без NaN, без B5), детерминизм kmeans при n_init=10, KEFRiN > kmeans на
сетевой синтетике с шумными признаками."""
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from ecotypes.cluster import (B5_FORBIDDEN, FEATURE_COLS, feature_matrix,
                              leiden_consensus_cspa, leiden_plateau,
                              run_feature_zoo, run_kefrin, run_leiden)
from ecotypes.config import load_config
from ecotypes.synthetic import ari, make_ppdataset, nmi

# γ внутри плато PP-Dir K=4 (проверено свипом: k=4, seed-ARI=1.0 при γ∈[0.6,0.8];
# γ=1.0 уже дробит k=6 — дефолт конфига не подставлять в тесты вслепую).
GAMMA_PP = 0.7


@pytest.fixture(scope="module")
def pp4() -> dict:
    """PP-Dir K=4, δ=0, n=300 — planted partition для Leiden-тестов."""
    return make_ppdataset(seed=1, K=4, n=300, T=24, delta=0.0)


def _nodes_from_mini(mini: dict) -> pd.DataFrame:
    """nodes_static-подобный DataFrame из PP-Dir мини-панели: все FEATURE_COLS
    + запрещённый B5-блок (проверяем, что он НЕ попадает в X)."""
    X, V = mini["X"], mini["V"]  # (T,n,6) доли, (T,n) объёмы
    n = X.shape[1]
    sh = np.clip(X.mean(axis=0), 1e-9, None)
    sh /= sh.sum(1, keepdims=True)
    clr = np.log(sh) - np.log(sh).mean(1, keepdims=True)
    lv = np.log(V)
    dec = np.arange(X.shape[0]) % 12 == 11
    nodes = pd.DataFrame({
        "territory_id": np.arange(n, dtype=np.int32) + 1,
        "clr_mean_prod": clr[:, 0], "clr_mean_health": clr[:, 1],
        "clr_mean_market": clr[:, 2], "clr_mean_food": clr[:, 3],
        "clr_mean_transp": clr[:, 4], "clr_mean_proch": clr[:, 5],
        "level_mean": np.median(lv, axis=0),
        "yoy_all_med": np.median(lv[12:] - lv[:12], axis=0),
        "yoy_market_med": np.median(np.log(X[12:, :, 2] / X[:12, :, 2]), axis=0),
        "dclr_market": clr[:, 2] * 0.1,
        "dec_amp_all": lv[dec].mean(0) - lv.mean(0),
        "season_std": np.std([lv[m::12].mean(0) for m in range(12)], axis=0),
        # B5-контекст: обязан остаться снаружи X_static
        "urban_share": np.linspace(0, 1, n), "log_wage": np.full(n, 10.0),
        "empl_pc": np.full(n, 0.2), "log_ma": np.log(V.mean(0)),
    })
    return nodes


# (а) Leiden восстанавливает planted partition
def test_run_leiden_recovers_planted(pp4):
    lab = run_leiden(pp4["A"], gamma=GAMMA_PP, seed=42)
    assert lab.dtype == np.int64 and lab.shape == (300,)
    assert ari(pp4["z"][0], lab) > 0.8


# (б) CSPA: детерминизм и доминирование над одиночными прогонами
def test_cspa_deterministic_and_dominant():
    d = make_ppdataset(seed=3, K=4, n=300, T=24, delta=0.0, mu_edge=0.4)
    A, seeds = d["A"], list(range(10))
    c1 = leiden_consensus_cspa(A, gamma=GAMMA_PP, seeds=seeds)
    c2 = leiden_consensus_cspa(A, gamma=GAMMA_PP, seeds=seeds)
    assert ari(c1, c2) == 1.0
    singles = [run_leiden(A, gamma=GAMMA_PP, seed=s) for s in seeds]
    cross_single = np.mean([ari(singles[i], singles[j])
                            for i, j in combinations(range(len(seeds)), 2)])
    cross_cons = np.mean([ari(c1, s) for s in singles])
    assert cross_cons >= cross_single


# (в) плато-правило находит γ с k ∈ [4, 10]
def test_plateau_finds_gamma(pp4):
    gammas = [0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
    res = leiden_plateau(pp4["A"], gammas, seeds=[0, 1, 2],
                         k_min=4, k_max=10, ari_min=0.9)
    assert res["gamma_star"] in res["plateau"] and not res["fallback"]
    assert 4 <= res["k_med"] <= 10
    assert len(res["table"]) == len(gammas)
    lab = run_leiden(pp4["A"], gamma=res["gamma_star"], seed=42)
    assert ari(pp4["z"][0], lab) > 0.8  # плато-точка — качественная


# (г) feature_matrix: форма, без NaN, без B5
def test_feature_matrix(mini_panel):
    cfg = load_config("configs/default.yaml")
    nodes = _nodes_from_mini(mini_panel)
    X, names = feature_matrix(nodes, cfg)
    assert X.shape == (len(nodes), len(FEATURE_COLS)) and not np.isnan(X).any()
    assert names == FEATURE_COLS
    assert not (set(names) & set(B5_FORBIDDEN))
    bad = nodes.copy()
    bad.loc[0, "level_mean"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        feature_matrix(bad, cfg)


# (д) kmeans n_init=10 детерминизм
def test_kmeans_n_init_determinism(mini_panel):
    cfg = load_config("configs/default.yaml")
    assert cfg.cluster.kmeans.n_init == 10  # ловушка sklearn 1.9: auto == 1
    X, _ = feature_matrix(_nodes_from_mini(mini_panel), cfg)
    z1 = run_feature_zoo(X, cfg)
    z2 = run_feature_zoo(X, cfg)
    for method, dd in z1.items():
        for k, lab in dd.items():
            assert np.array_equal(lab, z2[method][k]), f"{method}@{k}"


# (е) KEFRiN бьёт K-means на сильной сетевой структуре с шумными признаками
def test_kefrin_beats_kmeans():
    d = make_ppdataset(seed=7, K=4, n=400, T=24, delta=0.0, mu_edge=0.15)
    A, z = d["A"], d["z"][0]
    sh = np.clip(d["X"].mean(axis=0), 1e-9, None)
    sh /= sh.sum(1, keepdims=True)
    clr = np.log(sh) - np.log(sh).mean(1, keepdims=True)
    rng = np.random.default_rng(0)
    X = StandardScaler().fit_transform(clr + rng.normal(0, 1.5, clr.shape))
    km = KMeans(n_clusters=4, n_init=10, random_state=0).fit_predict(X)
    kf = run_kefrin(A, X, K=4, rho=0.5, xi=0.5, seed=0)
    assert nmi(z, kf) > nmi(z, km)
