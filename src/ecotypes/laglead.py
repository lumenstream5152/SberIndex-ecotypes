"""Лаг-лидерство (research/22): направленный граф «лидер → последователь».

Метод (контракт research/22 §8):
- Рабочие ряды — десезонированные рыночно-остаточные приросты агрегата
  (= measures.residual_returns, база M2) и 5 категорий (measures.category_residuals).
- Направленный скан лаговых кросс-корреляций k=1..4 с Pearson-перенормировкой
  на перекрытии; статистика пары = max signed r по 2k направленным лагам.
- ЗНАК ЛАГА (совпадает с исправленным measures.sim_M4): lag_map[i, j] = ℓ
  означает «j ЛИДИРУЕТ i на ℓ мес» при ℓ>0 (максимум на corr(x_i[t], x_j[t−ℓ]));
  ℓ<0 — i лидирует j на |ℓ|. В edge list направление разворачивается в колонки
  leader/follower, lag ребра = |ℓ| ∈ 1..max_lag (всегда положителен).
- Значимость: phase-randomization суррогаты (Theiler: сохраняют амплитуды
  спектра, фазы случайны → убивают выравнивание, сохраняют автокорреляцию).
  Пулled-нуль r_max по B суррогатам × случайному подмножеству пар → эмпирические
  p для ВСЕХ пар → BH-FDR. Пулled-дизайн снимает пол p_min = 1/(B+1)
  per-edge схемы (22 §3, круг 3): разрешение p = 1/(B·n_pairs + 1).
- Circular-shift перестановки НЕ используются [22 §3, checked]: допустимых
  сдвигов T−2 ≈ 22 → p_min ≈ 0.043 (структурно бессилен под FDR), а малые
  сдвиги сохраняют истинное выравнивание → нуль загрязнён сигналом.
- Пороги-ориентиры со спеки (пересчитываются на собранных данных): ядро
  r>0.80 ≈ 1.4k рёбер, слой r>0.70 ≈ 11k; эмпирический нуль q99.9 ≈ 0.785.

DTW как оценщик лага провален и здесь не используется (22 §2, §6): exact-recovery
≤0.15 на синтетике; DTW остаётся ненаправленной метрикой в measures (M6/M7).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import measures as ms

MAX_LAG = 4            # k=1..4 (22 §8.2); лаг >4 при T=24 не обоснован
N_SURROGATES = 200     # B фазовых суррогатов (22 §3: 200–499; 200 × 50k пар = 10M нуля)
NULL_PAIRS = 50_000    # случайное подмножество пар для пулled-нуля (≥50k по контракту)
FDR_Q = 0.05           # BH поверх эмпирических p всех пар
CORE_R = 0.80          # ядро (спека: ≈1.4k рёбер)
LAYER_R = 0.70         # расширенный слой (спека: ≈11k)
NEAR_KM = 150.0        # «близкие» рёбра для географического согласия


# --- скан лагов ----------------------------------------------------------------

def lag_scan(X: np.ndarray, max_lag: int = MAX_LAG
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Направленный скан кросс-корреляций рядов X (n, T) с перенормировкой
    на перекрытии. Возвращает (r_max, lag_map, direction), все (n, n):

    - r_max[i, j] = max signed r по лагам ±1..±max_lag (диагональ = NaN);
    - lag_map[i, j] = ℓ: ℓ>0 ⟺ j лидирует i на ℓ мес; ℓ<0 ⟺ i лидирует j на |ℓ|
      (конвенция measures.sim_M4 после исправления знака);
    - direction[i, j] = sign(ℓ) ∈ {+1, −1, 0}: +1 — лидер j, −1 — лидер i,
      0 — диагональ. Антисимметрична: direction[j, i] = −direction[i, j].
    """
    X = np.asarray(X, dtype=np.float64)
    n, T = X.shape
    r_max = np.full((n, n), -np.inf)
    lag_map = np.zeros((n, n), dtype=np.int8)
    for k in range(1, min(max_lag, T - 1) + 1):
        # R[i,j] = corr(x_i[t], x_j[t+k]): будущее j ≈ настоящее i → i лидирует
        # j на k мес → по конвенции lag_map[i,j] = −k; транспонирование даёт +k.
        R = ms._corr2(X[:, :T - k], X[:, k:])
        upd = R > r_max
        r_max[upd] = R[upd]
        lag_map[upd] = -k
        Rt = R.T
        upd = Rt > r_max
        r_max[upd] = Rt[upd]
        lag_map[upd] = k
    np.fill_diagonal(r_max, np.nan)
    np.fill_diagonal(lag_map, 0)
    return r_max, lag_map, np.sign(lag_map).astype(np.int8)


# --- суррогаты и нуль ------------------------------------------------------------

def phase_randomize(X: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Theiler phase-randomization: IFFT с теми же амплитудами спектра и
    случайными фазами (периодограмма сохраняется точно, выравнивание между
    рядами уничтожается). DC- и Nyquist-бины остаются вещественными."""
    X = np.asarray(X, dtype=np.float64)
    F = np.fft.rfft(X, axis=1)
    ph = rng.uniform(0.0, 2.0 * np.pi, F.shape)
    ph[:, 0] = 0.0
    if X.shape[1] % 2 == 0:
        ph[:, -1] = 0.0
    return np.fft.irfft(F * np.exp(1j * ph), n=X.shape[1], axis=1)


def null_pair_index(n: int, n_pairs: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Фиксированное (по seed) случайное подмножество пар i<j для нуля."""
    rng = np.random.default_rng(seed)
    iu = np.triu_indices(n, 1)
    total = iu[0].size
    sel = rng.choice(total, size=min(n_pairs, total), replace=False)
    return iu[0][sel].astype(np.int32), iu[1][sel].astype(np.int32)


def surrogate_null(X: np.ndarray, B: int = N_SURROGATES, n_pairs: int = NULL_PAIRS,
                   seed: int = 0, max_lag: int = MAX_LAG
                   ) -> tuple[np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray]]:
    """Пулled-нуль r_max: B фазовых суррогатов панели × фиксированное
    подмножество пар. Возвращает (null (B·m,), null_lags (B·m,) int8, (pi, pj)).
    Одна и та же пара даёт B независимых нулевых значений — это и есть пулинг."""
    rng = np.random.default_rng(seed)
    pi, pj = null_pair_index(X.shape[0], n_pairs, seed ^ 0x5EED)
    null = np.empty((B, pi.size), dtype=np.float64)
    lags = np.empty((B, pi.size), dtype=np.int8)
    for b in range(B):
        r_b, lag_b, _ = lag_scan(phase_randomize(X, rng), max_lag=max_lag)
        null[b] = r_b[pi, pj]
        lags[b] = lag_b[pi, pj]
    return null.ravel(), lags.ravel(), (pi, pj)


def empirical_pvalues(r_obs: np.ndarray, null_sorted: np.ndarray) -> np.ndarray:
    """p = (1 + #{null ≥ r}) / (1 + N_null) — односторонние, консервативные
    на связях (side='left': равные наблюдению нули считаются «хвостом»)."""
    ns = np.asarray(null_sorted)
    cnt = ns.size - np.searchsorted(ns, r_obs, side="left")
    return (1.0 + cnt) / (1.0 + ns.size)


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q-values (statsmodels в зависимостях нет)."""
    p = np.asarray(p, dtype=np.float64)
    m = p.size
    order = np.argsort(p, kind="stable")
    ranked = p[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m)
    q[order] = np.minimum(ranked, 1.0)
    return q


# --- граф ----------------------------------------------------------------------

def lead_edges_from_scan(r_max: np.ndarray, lag_map: np.ndarray,
                         sig_mask: np.ndarray) -> pd.DataFrame:
    """Разворот (i<j, ℓ) в ориентированные рёбра leader→follower.
    ℓ = lag_map[i,j] > 0 → лидер j; ℓ < 0 → лидер i. lag ребра = |ℓ| > 0."""
    ii, jj = np.triu_indices(r_max.shape[0], 1)
    m = sig_mask[ii, jj]
    i, j, lag = ii[m], jj[m], lag_map[ii[m], jj[m]]
    leader = np.where(lag > 0, j, i)
    follower = np.where(lag > 0, i, j)
    return pd.DataFrame({"leader_row": leader.astype(np.int32),
                         "follower_row": follower.astype(np.int32),
                         "lag": np.abs(lag).astype(np.int8),
                         "r": r_max[i, j]})


def lead_breadth(edges: pd.DataFrame, cat_lag_maps: list[np.ndarray]) -> np.ndarray:
    """Число категорий (0..5), согласных с направлением агрегатного ребра и его
    лагом ±1 мес: cat_lag[follower, leader] > 0 ⟺ leader лидирует и в категории."""
    f = edges.follower_row.to_numpy()
    l = edges.leader_row.to_numpy()
    k = edges.lag.to_numpy()
    breadth = np.zeros(len(edges), dtype=np.int8)
    for lm in cat_lag_maps:
        lc = lm[f, l]
        breadth += ((lc > 0) & (np.abs(lc - k) <= 1)).astype(np.int8)
    return breadth


def _top_nodes(deg: np.ndarray, tids: np.ndarray, names: pd.DataFrame,
               top: int = 15) -> list[dict]:
    idx = np.argsort(-deg, kind="stable")[:top]
    nm = names.set_index("territory_id")
    return [{"territory_id": int(tids[i]), "deg": int(deg[i]),
             "name": str(nm.loc[tids[i], "name"]),
             "region_name": str(nm.loc[tids[i], "region_name"])} for i in idx]


def graph_aggregates(edges: pd.DataFrame, data: ms.MeasureData,
                     labels: pd.DataFrame | None, nodes: pd.DataFrame,
                     null_pairs: tuple[np.ndarray, np.ndarray]) -> dict:
    """Агрегаты продукта: степени (маяки/последователи), согласие с типами
    (leiden_consensus) и с географией (highway kNN + haversine < 150 км)."""
    n, tids = data.n, data.tids
    lr, fr = edges.leader_row.to_numpy(), edges.follower_row.to_numpy()
    out_deg = np.bincount(lr, minlength=n)
    in_deg = np.bincount(fr, minlength=n)
    out: dict = {"top_beacons": _top_nodes(out_deg, tids, nodes),
                 "top_followers": _top_nodes(in_deg, tids, nodes),
                 "out_deg_mean": float(out_deg.mean()),
                 "in_deg_max": int(in_deg.max()), "out_deg_max": int(out_deg.max())}

    if labels is not None and len(edges):
        lab = labels.set_index("territory_id")["leiden_consensus"]
        cl = lab.loc[tids].to_numpy()
        same = cl[lr] == cl[fr]
        expected = float(((np.bincount(cl) / n) ** 2).sum())  # Σ_c (n_c/n)²
        out["types"] = {"within_share": float(same.mean()),
                        "expected_share_random": expected,
                        "enrichment": float(same.mean() / max(expected, 1e-12)),
                        "mean_out_deg_by_type": {str(c): float(out_deg[cl == c].mean())
                                                 for c in np.unique(cl)}}

    if data.hw is not None and len(edges):
        ei, ej, dist = data.hw
        key2d: dict[int, float] = {}
        for a, b, d in zip(ei, ej, dist):
            key2d[a * n + b] = min(d, key2d.get(a * n + b, np.inf))
        d_hw = np.array([key2d.get(a * n + b, np.nan) for a, b in zip(lr, fr)])
        matched = ~np.isnan(d_hw)
        r = 6371.0088
        la1, lo1 = np.radians(nodes.lat.to_numpy()), np.radians(nodes.lon.to_numpy())
        a = (np.sin((la1[fr] - la1[lr]) / 2) ** 2
             + np.cos(la1[lr]) * np.cos(la1[fr]) * np.sin((lo1[fr] - lo1[lr]) / 2) ** 2)
        d_hav = 2 * r * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
        pi, pj = null_pairs
        a0 = (np.sin((la1[pj] - la1[pi]) / 2) ** 2
              + np.cos(la1[pi]) * np.cos(la1[pj]) * np.sin((lo1[pj] - lo1[pi]) / 2) ** 2)
        d_base = 2 * r * np.arcsin(np.sqrt(np.clip(a0, 0, 1)))
        out["geo"] = {
            "hw_matched_share": float(matched.mean()),
            "hw_near_share_matched": float((d_hw[matched] < NEAR_KM).mean()),
            "near150_share_edges": float((d_hav < NEAR_KM).mean()),
            "near150_share_random_pairs": float((d_base < NEAR_KM).mean()),
        }
    return out


# --- синтетическая верификация ----------------------------------------------------

def synthetic_verification(seed: int = 0, n_pairs: int = 100, k: int = 2,
                           snr: float = 4.0, T: int = 23, max_lag: int = MAX_LAG,
                           null_n: int = 150, null_B: int = 40) -> dict:
    """(а) восстановление известного лага: follower = leader, сдвинутый на k,
    + шум SNR≥3 → доля точных попаданий; (в) нуль: независимые AR(1) → доля
    открытий после FDR (ожидание ≤ 5%). Лёгкая версия — для metrics.json."""
    rng = np.random.default_rng(seed)
    X = np.empty((2 * n_pairs, T))
    for p in range(n_pairs):
        e = np.empty(T + k)
        e[0] = rng.normal()
        for t in range(1, T + k):
            e[t] = 0.5 * e[t - 1] + rng.normal()
        lead = e[k:]
        X[2 * p] = lead
        X[2 * p + 1] = e[:T] + rng.normal(0.0, lead.std() / np.sqrt(snr), T)
    _, lag_map, _ = lag_scan(X, max_lag=max_lag)
    li = np.arange(0, 2 * n_pairs, 2)
    recovery = float((lag_map[li, li + 1] == -k).mean())

    z = np.empty((null_n, T))
    for i in range(null_n):
        z[i, 0] = rng.normal()
        for t in range(1, T):
            z[i, t] = 0.5 * z[i, t - 1] + rng.normal()
    null, _, _ = surrogate_null(z, B=null_B, n_pairs=5000, seed=seed + 1,
                                max_lag=max_lag)
    r_obs, _, _ = lag_scan(z, max_lag=max_lag)
    iu = np.triu_indices(null_n, 1)
    q = bh_qvalues(empirical_pvalues(r_obs[iu], np.sort(null)))
    return {"planted": {"n_pairs": n_pairs, "k": k, "snr": snr,
                        "exact_recovery": recovery},
            "null": {"n": null_n, "B": null_B,
                     "fdr_discovery_rate": float((q <= FDR_Q).mean())}}


# --- полный прогон -----------------------------------------------------------------

def run_laglead(out_dir: str | Path, labels_path: str | Path | None = None,
                seed: int = 0, B: int = N_SURROGATES, n_null_pairs: int = NULL_PAIRS,
                fdr_q: float = FDR_Q, max_lag: int = MAX_LAG,
                log=lambda m: None) -> tuple[pd.DataFrame, dict]:
    """Полный пайплайн: данные → скан (агрегат + 5 категорий) → суррогатный
    нуль → BH-FDR → edge list + агрегаты → запись в <out_dir>/laglead/."""
    import time
    t0 = time.time()
    out_dir = Path(out_dir)
    data = ms.from_processed(out_dir)
    log(f"laglead: n={data.n}, T={data.T}")

    X = ms.residual_returns(data)
    r_max, lag_map, _ = lag_scan(X, max_lag=max_lag)
    iu = np.triu_indices(data.n, 1)
    r_obs = r_max[iu]
    qts = (0.5, 0.9, 0.99, 0.999, 0.9999)
    emp_null = {f"q{int(q_ * 10000) / 100}": float(v)
                for q_, v in zip(qts, np.quantile(r_obs, qts))}
    log(f"скан агрегата: эмпирический нуль { {k: round(v, 3) for k, v in emp_null.items()} }")

    null, null_lags, null_pairs = surrogate_null(X, B=B, n_pairs=n_null_pairs,
                                                 seed=seed, max_lag=max_lag)
    null_sorted = np.sort(null)
    p = empirical_pvalues(r_obs, null_sorted)
    q = bh_qvalues(p)
    sig = q <= fdr_q
    log(f"нуль {null.size:,} значений ({time.time() - t0:.0f}s); "
        f"FDR q≤{fdr_q}: {sig.sum():,} рёбер из {p.size:,} пар")

    sig_mask = np.zeros((data.n, data.n), dtype=bool)
    sig_mask[iu[0][sig], iu[1][sig]] = True
    edges = lead_edges_from_scan(r_max, lag_map, sig_mask)
    edges["p"] = p[sig]
    edges["q"] = q[sig]
    cat_res = ms.category_residuals(data)  # (n, T−1, 5) — один расчёт на все категории
    cat_lags = [lag_scan(cat_res[:, :, c], max_lag=max_lag)[1] for c in range(5)]
    edges["lead_breadth"] = lead_breadth(edges, cat_lags)
    edges["leader"] = data.tids[edges.leader_row.to_numpy()]
    edges["follower"] = data.tids[edges.follower_row.to_numpy()]
    edges = edges[["leader", "follower", "lag", "r", "p", "q", "lead_breadth",
                   "leader_row", "follower_row"]].sort_values(
        ["r", "leader", "follower"], ascending=[False, True, True],
        ignore_index=True)

    labels = None
    if labels_path is not None and Path(labels_path).exists():
        labels = pd.read_parquet(labels_path, columns=["territory_id",
                                                       "leiden_consensus"])
    nodes = pd.read_parquet(out_dir / "nodes_static.parquet",
                            columns=["territory_id", "name", "region_name",
                                     "lat", "lon"])
    agg = graph_aggregates(edges, data, labels, nodes, null_pairs)

    summary = {
        "n_nodes": int(data.n), "n_pairs": int(p.size), "max_lag": max_lag,
        "surrogates": {"B": B, "null_pairs": int(null_pairs[0].size),
                       "null_size": int(null.size),
                       "null_q95": float(np.quantile(null, 0.95)),
                       "null_q99": float(np.quantile(null, 0.99)),
                       "null_lag_mix": {str(k_): float((np.abs(null_lags) == k_).mean())
                                        for k_ in range(1, max_lag + 1)}},
        "empirical_null_observed": emp_null,
        "fdr": {"q": fdr_q, "n_significant": int(sig.sum()),
                "r_at_fdr_cutoff": float(edges.r.min()) if len(edges) else None},
        "edges_by_threshold": {
            "r>0.80_core": int((r_obs > CORE_R).sum()),
            "r>0.70_layer": int((r_obs > LAYER_R).sum()),
            "spec_expectation": {"r>0.80": 1370, "r>0.70": 11434}},
        "edge_lag_mix": {str(k_): float((edges.lag == k_).mean())
                         for k_ in range(1, max_lag + 1)} if len(edges) else {},
        "lead_breadth_mean": float(edges.lead_breadth.mean()) if len(edges) else None,
        **agg,
        "seconds": round(time.time() - t0, 1),
    }

    dest = out_dir / "laglead"
    dest.mkdir(parents=True, exist_ok=True)
    edges.to_parquet(dest / "lead_graph.parquet", index=False)
    with open(dest / "lead_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    log(f"→ {dest}/ ({time.time() - t0:.0f}s)")
    return edges, summary
