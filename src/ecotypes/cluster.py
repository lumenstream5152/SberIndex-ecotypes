"""Кластерный этап (research/25/26/27, 05.10.2026).

Ядро — Leiden-RB на статическом взвешенном графе (similarity ∪ geo на поддержке
skeleton); разброс по seed'ам гасится CSPA-консенсусом (25 §TL;DR — одиночный
прогон недопустим: seed-разброс ARI 0.44–0.73). Рабочая точка γ* — плато-правилом
(25 §3.2), не ICVI-knee. Фича-парадигма (27): X_static 2016×12 + зоопарк
KMeans(n_init=10 — НЕ auto, ловушка sklearn 1.9)/Ward/GMM(diag+full, BIC)/
Spectral-kNN. KEFRiN — своя реализация по Shalileh & Mirkin, Entropy 2022
(у репо автора нет лицензии, 26 §2.3); численно сверена с референсом на
синтетике (NMI картина та же). Louvain/Infomap/EVA — строки сравнения (25 §4,
26 §3); DMoN — опциональный try-import, в lock не входит (tensorflow).
"""
from __future__ import annotations

import logging
from itertools import combinations

import igraph as ig
import leidenalg as la
import networkx as nx
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.cluster import AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

from .config import Config
from .synthetic import ari

log = logging.getLogger(__name__)

__all__ = [
    "FEATURE_COLS",
    "run_leiden",
    "leiden_consensus_cspa",
    "leiden_plateau",
    "feature_matrix",
    "run_feature_zoo",
    "gmm_bic",
    "run_kefrin",
    "run_louvain",
    "run_infomap",
    "run_eva",
    "run_dmon",
    "q_norm",
]

KEFRIN_N_INIT = 10  # n_init референса (KEFRiNConfig) и нашего протокола k-means

# X_static: 6 CLR-компонентов + уровень/рост/оцифровка/сезонность (27 §1).
# Отступление от спеки 27 (там 10 признаков): в блок идут ОБЕ версии роста
# маркетплейсов (yoy_market_med и dclr_market) и ОБЕ сезонные амплитуды
# (dec_amp_all и season_std) — 12 признаков; зафиксировано в PREREG_DEVIATIONS.
# Росстат-контекст B5 (urban_share, log_wage, empl_pc, log_ma) НЕ входит —
# только валидация/интерпретация (ловушка Frolovskiy, 20_BUILD_FOUNDATION).
FEATURE_COLS = [
    "clr_mean_prod", "clr_mean_health", "clr_mean_market", "clr_mean_food",
    "clr_mean_transp", "clr_mean_proch",
    "level_mean", "yoy_all_med", "yoy_market_med", "dclr_market",
    "dec_amp_all", "season_std",
]

# B5-колонки, запрещённые в X_static (проверяется тестом).
B5_FORBIDDEN = ("urban_share", "log_wage", "empl_pc", "log_ma")

_PARTITIONS = {
    "RBConfiguration": la.RBConfigurationVertexPartition,
    "CPM": la.CPMVertexPartition,
}


# ---------------------------------------------------------------- Leiden-ядро

def _to_igraph(A: sp.csr_matrix) -> ig.Graph:
    """CSR (симметричная, нулевая диагональ) → igraph со списком рёбер i<j.

    Веса — из A.data верхнего треугольника; диагональ и возможная асимметрия
    отбрасываются (контракт входа — симметрия, diag=0).
    """
    A = sp.csr_matrix(A)
    n = A.shape[0]
    up = sp.triu(A, k=1, format="coo")
    g = ig.Graph(n=n, edges=list(zip(up.row.tolist(), up.col.tolist())),
                 directed=False)
    g.es["weight"] = np.asarray(up.data, dtype=np.float64).tolist()
    return g


def _leiden_on(g: ig.Graph, gamma: float, seed: int, partition: str) -> np.ndarray:
    part = la.find_partition(
        g, _PARTITIONS[partition],
        weights="weight", resolution_parameter=gamma,
        seed=seed, n_iterations=-1,  # -1 = до сходимости (25 §2)
    )
    return np.asarray(part.membership, dtype=np.int64)


def run_leiden(A: sp.csr_matrix, *, gamma: float, seed: int,
               partition: str = "RBConfiguration") -> np.ndarray:
    """Один прогон Leiden (igraph + leidenalg) → метки int64 (n,). Контракт
    соседнего модуля dynamics — сигнатуру не менять."""
    return _leiden_on(_to_igraph(A), gamma, seed, partition)


def q_norm(A: sp.csr_matrix, labels: np.ndarray) -> float:
    """Нормированная модульность igraph ([−0.5, 1]). part.quality() у RB/CPM
    НЕнормирован — для отчётности только эта (грабля 25 §2)."""
    g = _to_igraph(A)
    return float(g.modularity(np.asarray(labels).tolist(), weights="weight"))


def leiden_consensus_cspa(A: sp.csr_matrix, *, gamma: float,
                          seeds: list[int]) -> np.ndarray:
    """CSPA-консенсус: co-association C = mean_s 1[z_i == z_j] по seeds →
    финальный Leiden на C (детерминирован: seed = seeds[0]). Одиночный прогон
    как прод-разбиение недопустим (25 TL;DR)."""
    labs = [run_leiden(A, gamma=gamma, seed=s) for s in seeds]
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for lab in labs:
        onehot = np.zeros((n, lab.max() + 1))
        onehot[np.arange(n), lab] = 1.0
        C += onehot @ onehot.T
    C /= len(seeds)
    np.fill_diagonal(C, 0.0)
    return run_leiden(sp.csr_matrix(C), gamma=gamma, seed=seeds[0])


def _grid_icvi_composite(g: ig.Graph, A: sp.csr_matrix, X: np.ndarray,
                         labs_per_gamma: list[list[np.ndarray]]) -> pd.DataFrame:
    """ICVI-композит на каждом γ сетки — для выбора ВНУТРИ плато (prereg
    fixed_decisions.gamma_rule: «ICVI — только внутри плато»). Веса — prereg
    method.icvi_subweights {SW .12, CH_over_N .06, neg_S_Dbw .06, MQ .06};
    MQ — mancoridis (fixed_decisions.mq_variant). Репрезентативное разбиение
    на γ — прогон с max Q_norm («vs лучший из найденных», 25 §6.1). Индексы
    min-max-нормируются по сетке (относительный скоринг, не абсолют)."""
    from . import icvi

    n = A.shape[0]
    rows = []
    for labs in labs_per_gamma:
        qs = [g.modularity(l.tolist(), weights="weight") for l in labs]
        lab = labs[int(np.argmax(qs))]
        k = len(np.unique(lab))
        if 1 < k < n:
            rows.append(dict(icvi_sw=icvi.sw(X, lab),
                             icvi_ch=icvi.ch_over_n(X, lab),
                             icvi_sdbw=icvi.sdbw(X, lab),
                             icvi_mq=icvi.mq(A, lab, variant="mancoridis")))
        else:  # вырожденное разбиение — композит 0, индексы NaN
            rows.append(dict(icvi_sw=np.nan, icvi_ch=np.nan,
                             icvi_sdbw=np.nan, icvi_mq=np.nan))
    df = pd.DataFrame(rows)

    def _norm(s: pd.Series, higher_better: bool = True) -> pd.Series:
        lo, hi = s.min(), s.max()
        z = (s - lo) / (hi - lo) if hi > lo else pd.Series(0.5, index=s.index)
        return z if higher_better else 1.0 - z

    df["icvi_comp"] = (0.12 * _norm(df.icvi_sw) + 0.06 * _norm(df.icvi_ch)
                       + 0.06 * _norm(df.icvi_sdbw, higher_better=False)
                       + 0.06 * _norm(df.icvi_mq)).fillna(0.0)
    return df


def leiden_plateau(A: sp.csr_matrix, gammas: list[float], seeds: list[int],
                   k_min: int = 4, k_max: int = 10,
                   ari_min: float = 0.9, *,
                   X: np.ndarray | None = None) -> dict:
    """Плато-правило выбора γ (25 §3.2): на каждом γ — len(seeds) прогонов;
    плато = максимальный подряд идущий участок сетки с seed-ARI(медиана) ≥ ari_min
    и разбросом медианного k ≤ ±1; среди плато с k ∈ [k_min, k_max] —
    МАКСИМУМ ICVI-КОМПОЗИТА (требует X; 25 §3.2 п.5, prereg gamma_rule),
    тай-брейк → более широкое плато, затем выше средний seed-ARI. Без X —
    легаси-ранжирование по ширине (совместимость тестов/синтетики без признаков).
    γ* = средняя точка плато (индекс сетки; на лог-сетке ≈ лог-середина).
    Плато с k в бюджете нет → деградация 25 §3.2: ближайшее по k плато,
    помечено fallback=True.

    Возвращает {gamma_star, plateau (список γ), table (DataFrame по γ),
    fallback, k_med}.
    """
    g = _to_igraph(A)
    n = A.shape[0]
    rows, labs_grid = [], []
    for gamma in gammas:
        labs = [_leiden_on(g, gamma, s, "RBConfiguration") for s in seeds]
        labs_grid.append(labs)
        ks = [int(l.max()) + 1 for l in labs]
        aris = [ari(labs[i], labs[j]) for i, j in combinations(range(len(labs)), 2)]
        # красные флаги (25 §6) — худший случай по seed'ам
        sing, share = [], []
        for l in labs:
            s = np.bincount(l)
            sing.append((s == 1).mean())
            share.append(s.max() / n)
        rows.append(dict(
            gamma=float(gamma), k_med=float(np.median(ks)),
            k_lo=min(ks), k_hi=max(ks),
            ari_med=float(np.median(aris)) if aris else 1.0,
            ari_min=float(np.min(aris)) if aris else 1.0,
            q_med=float(np.median([g.modularity(l.tolist(), weights="weight")
                                   for l in labs])),
            singleton_share=float(max(sing)),
            max_share=float(max(share)),
        ))
    table = pd.DataFrame(rows)
    if X is not None:
        table = pd.concat(
            [table, _grid_icvi_composite(g, A, X, labs_grid)], axis=1)
    valid = (table.ari_med >= ari_min).to_numpy()
    kmed = table.k_med.to_numpy()

    # кандидаты: (i0, i1) — включительные границы максимальных валидных окон
    # с разбросом k_med ≤ 1
    cands: list[tuple[int, int]] = []
    i = 0
    while i < len(gammas):
        if not valid[i]:
            i += 1
            continue
        j = i
        while j + 1 < len(gammas) and valid[j + 1]:
            j += 1
        for a in range(i, j + 1):
            b = a
            while b + 1 <= j and kmed[a:b + 2].max() - kmed[a:b + 2].min() <= 1:
                b += 1
            cands.append((a, b))
        i = j + 1
    # схлопываем вложенные: оставляем только максимальные по длине на старт
    cands = [(a, b) for a, b in cands
             if not any(a2 <= a and b <= b2 and (a2, b2) != (a, b)
                        for a2, b2 in cands)]

    def _score(c: tuple[int, int]) -> tuple:
        a, b = c
        kmid = 0.5 * (k_min + k_max)
        width, ari_m = b - a, table.ari_med.iloc[a:b + 1].mean()
        kdist = -abs(table.k_med.iloc[a:b + 1].mean() - kmid)
        if X is not None:  # 25 §3.2 п.5: композит первичен, ширина — тай-брейк
            return (table.icvi_comp.iloc[a:b + 1].mean(), width, ari_m, kdist)
        return (width, ari_m, kdist)

    in_budget = [c for c in cands
                 if k_min <= table.k_med.iloc[c[0]:c[1] + 1].mean() <= k_max]
    fallback = not in_budget
    pool = in_budget or sorted(
        cands,
        key=lambda c: (max(0.0, k_min - table.k_med.iloc[c[0]:c[1] + 1].max(),
                           table.k_med.iloc[c[0]:c[1] + 1].min() - k_max),
                       -table.ari_med.iloc[c[0]:c[1] + 1].mean()))
    chosen = max(pool, key=_score) if in_budget else (pool[0] if pool else None)
    if chosen is None:
        return {"gamma_star": None, "plateau": [], "table": table,
                "fallback": True}
    a, b = chosen
    mid = a + (b - a) // 2
    return {"gamma_star": float(gammas[mid]),
            "plateau": [float(x) for x in gammas[a:b + 1]],
            "table": table, "fallback": fallback,
            "k_med": float(table.k_med.iloc[a:b + 1].mean())}


# ---------------------------------------------------------------- фича-парадигма

def feature_matrix(nodes: pd.DataFrame, cfg: Config) -> tuple[np.ndarray, list[str]]:
    """X_static (27 §1): FEATURE_COLS, StandardScaler. B5-контекст исключён.
    NaN — ошибка, не импутация (неполные ряды отсеяны этапом 02)."""
    missing = [c for c in FEATURE_COLS if c not in nodes.columns]
    if missing:
        raise KeyError(f"в nodes_static нет колонок признаков: {missing}")
    X = nodes[FEATURE_COLS].to_numpy(np.float64)
    if np.isnan(X).any():
        bad = [c for c, m in zip(FEATURE_COLS, np.isnan(X).any(0)) if m]
        raise ValueError(f"NaN в признаках X_static: {bad} — "
                         "импутация запрещена (27 §1)")
    return StandardScaler().fit_transform(X), list(FEATURE_COLS)


def run_feature_zoo(X: np.ndarray, cfg: Config) -> dict[str, dict[int, np.ndarray]]:
    """Зоопарк парадигмы «а» по K из cfg.cluster.kmeans.ks (27 §3):
    kmeans (n_init ИЗ КОНФИГА, явно — ловушка sklearn 1.9), ward,
    gmm_full/gmm_diag (reg_covar=1e-6: CLR-блок вырожден, 27 §1), spectral_knn
    (полный affinity при n=2016 терпим; nystroem не нужен, 27 §5)."""
    ks = cfg.cluster.kmeans.ks
    seed = cfg.seed
    out: dict[str, dict[int, np.ndarray]] = {
        "kmeans": {}, "ward": {}, "gmm_full": {}, "gmm_diag": {}, "spectral_knn": {}}
    for k in ks:
        out["kmeans"][k] = KMeans(n_clusters=k, n_init=cfg.cluster.kmeans.n_init,
                                  random_state=seed).fit_predict(X)
        out["ward"][k] = AgglomerativeClustering(
            n_clusters=k, linkage="ward").fit_predict(X)
        out["gmm_full"][k] = GaussianMixture(
            n_components=k, covariance_type="full", n_init=3, reg_covar=1e-6,
            random_state=seed).fit_predict(X)
        out["gmm_diag"][k] = GaussianMixture(
            n_components=k, covariance_type="diag", n_init=3, reg_covar=1e-6,
            random_state=seed).fit_predict(X)
        out["spectral_knn"][k] = SpectralClustering(
            n_clusters=k, affinity="nearest_neighbors", n_neighbors=15,
            random_state=seed).fit_predict(X)
    return out


def gmm_bic(X: np.ndarray, ks: list[int], seed: int) -> dict[str, dict[int, float]]:
    """BIC обеих ковариаций GMM (выбор K: минимум у full, 27 §4)."""
    out: dict[str, dict[int, float]] = {"gmm_full": {}, "gmm_diag": {}}
    for k in ks:
        for cov in ("full", "diag"):
            gm = GaussianMixture(n_components=k, covariance_type=cov, n_init=3,
                                 reg_covar=1e-6, random_state=seed).fit(X)
            out[f"gmm_{cov}"][k] = float(gm.bic(X) / X.shape[0])
    return out


# ---------------------------------------------------------------- KEFRiN (своя)

def _sq_euclid(X: np.ndarray, C: np.ndarray) -> np.ndarray:
    """(n, K) квадраты евклидовых дистояний строк X до центроидов C."""
    return (np.maximum((X * X).sum(1)[:, None] + (C * C).sum(1)[None, :]
                       - 2.0 * (X @ C.T), 0.0))


def run_kefrin(A: sp.csr_matrix, X: np.ndarray, *, K: int, rho: float,
               xi: float, seed: int) -> np.ndarray:
    """KEFRiN (Shalileh & Mirkin, Entropy 24(5):626, 2022, open access) —
    своя реализация: у репо автора нет лицензии (26 §2.3), код не заимствуется.

    Extended k-means в объединённом пространстве «признаки ⊕ строки сети»:
    критерий Σ_k Σ_{i∈S_k} [ρ‖y_i−c_k^y‖² + ξ‖p_i−c_k^p‖²], p_i — i-я строка
    плотной матрицы смежности. Шаг назначения — argmin по сумме квадратов
    (центроид = среднее — точный минимизатор), инициализация k-means++ с
    вероятностями ∝ D², KEFRIN_N_INIT рестартов, лучший по инерции.
    Препроцессинг по статье: y — z-score (идемпотентен на нашем X), p — none;
    шкалу сети гасит ручка ξ. Пустой кластер — перевыброс центроида в самую
    удалённую точку (детерминировано).
    """
    Y = np.asarray(X, dtype=np.float64)
    Y = (Y - Y.mean(0)) / np.where(Y.std(0) == 0, 1.0, Y.std(0))
    P = (A.toarray() if sp.issparse(A) else np.asarray(A)).astype(np.float64)
    n = Y.shape[0]
    rng = np.random.default_rng(seed)
    best_inertia, best_labels = np.inf, None
    for child in rng.integers(0, 2**32 - 1, KEFRIN_N_INIT):
        rr = np.random.default_rng(int(child))
        idx = [int(rr.integers(n))]
        while len(idx) < K:
            d = (rho * _sq_euclid(Y, Y[idx]).min(1)
                 + xi * _sq_euclid(P, P[idx]).min(1))
            wsum = d.sum()
            idx.append(int(rr.choice(n, p=d / wsum)) if wsum > 0
                       else int(rr.integers(n)))
        cy, cp = Y[idx].copy(), P[idx].copy()
        labels = np.zeros(n, dtype=np.int64)
        for _ in range(300):
            d = rho * _sq_euclid(Y, cy) + xi * _sq_euclid(P, cp)
            new = d.argmin(1)
            inertia = float(d[np.arange(n), new].sum())
            if (new == labels).all():
                labels = new
                break
            labels = new
            for k in range(K):
                m = labels == k
                if m.any():
                    cy[k], cp[k] = Y[m].mean(0), P[m].mean(0)
                else:  # пустой кластер → самая дальняя точка
                    far = int(d.min(1).argmax())
                    cy[k], cp[k] = Y[far], P[far]
        if inertia < best_inertia:
            best_inertia, best_labels = inertia, labels.copy()
    return best_labels


# ---------------------------------------------------------------- строки зоопарка

def _to_nx(A: sp.csr_matrix) -> nx.Graph:
    return nx.from_scipy_sparse_array(sp.csr_matrix(A))


def run_louvain(A: sp.csr_matrix, *, seed: int) -> np.ndarray:
    """Louvain (python-louvain, BSD) — индустриальный baseline и permissive-
    запаска под GPL leidenalg (25 §4)."""
    import community as community_louvain

    part = community_louvain.best_partition(_to_nx(A), weight="weight",
                                            random_state=seed)
    return np.array([part[i] for i in range(A.shape[0])], dtype=np.int64)


def run_infomap(A: sp.csr_matrix, *, seed: int) -> np.ndarray:
    """Infomap (--two-level, API 2.15: add_links/get_modules, 25 §4)."""
    import infomap

    up = sp.triu(sp.csr_matrix(A), k=1, format="coo")
    im = infomap.Infomap(f"--two-level --silent --seed {seed}")
    im.add_links(list(zip(up.row.tolist(), up.col.tolist(),
                          np.asarray(up.data, float).tolist())))
    im.run()
    modules = im.get_modules()
    return np.array([modules[i] for i in range(A.shape[0])], dtype=np.int64)


def run_eva(A: sp.csr_matrix, attrs: np.ndarray, *, alpha: float) -> np.ndarray:
    """EVA (Citraro & Rossetti 2019) через CDlib 0.4.1 (BSD-2): модульность ×
    чистота меток, только КАТЕГОРИАЛЬНЫЕ атрибуты (26 §3 — наши CLR-профили
    дискретизируются argmax'ом снаружи, потеря оговорена в таблице методов)."""
    from cdlib import algorithms

    attrs = np.asarray(attrs)
    labels = {i: {"profile": str(attrs[i])} for i in range(A.shape[0])}
    clu = algorithms.eva(_to_nx(A), labels=labels, alpha=alpha)
    out = np.full(A.shape[0], -1, dtype=np.int64)
    for cid, comm in enumerate(clu.communities):
        out[list(comm)] = cid
    return out


def run_dmon(A: sp.csr_matrix, X: np.ndarray, *, K: int,
             seed: int) -> np.ndarray | None:
    """DMoN (Tsitsulin et al., Google) — НЕ в lock (tensorflow), только
    опциональный путь (26 §5). Без TF/вендор-кода — честный skip: None."""
    try:
        import tensorflow  # noqa: F401
        from .vendor.dmon import fit_dmon  # вендорится отдельной волной
    except ImportError:
        log.warning("DMoN пропущен: tensorflow/вендор dmon недоступны "
                    "(research/26 §5 — строка переносится в обзор или в отдельный "
                    "бюджет тюнинга; lock не трогаем)")
        return None
    return fit_dmon(A, X, K=K, seed=seed)
