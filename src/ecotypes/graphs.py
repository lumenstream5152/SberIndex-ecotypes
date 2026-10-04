"""Графовый этап (research/23 «Инженерная спека графа», §РЕШЕНИЯ ДЛЯ СБОРКИ).

Протокол:
- сходство: log-приросты агрегата «Все категории» с кросс-секционным демеанингом
  по месяцу (lr_dm[i,t] = lr[i,t] − mean_i lr[·,t]). Сырые приросты вырождены
  (med r = 0.906, FDR пуст [checked 23 §0]) — медианы обоих пишем в metrics как
  анти-факты.
- каркас E* = (kNN(10, union) по W = clip(R, 0) ∩ BH-FDR q=0.05, t-тест, df=T−3)
  ∪ топ-3 safety net (union). Ожидание [checked 23 §1.3]: |E*| ≈ 12 900
  (12 759 после FDR + ~141 safety), 0 изолятов, 1 компонента, mean r ≈ 0.737.
  Union, не mutual (mutual: 102 изолята).
- гео-слой: kNN(10) по highway-дистанции (edges_highway_knn.parquet этапа 02),
  w = exp(−d/λ), λ = cfg.graph.geo.lambda_km (150). Население в веса НЕ идёт.
- динамика (23 §3.2): один edge set E_total = E* ∪ G на все 24 мес; месячный вес
  = CLR-косинус профилей долей месяца t на рёбрах каркаса; geo-only рёбра —
  константа exp(−d/λ) (якорь). Оконных корреляций нет (Jaccard половин 0.031).
  Ожидание: corr векторов весов соседних месяцев ≈ 0.949. Веса клипятся в
  [1e-6, 1] (контракт 23 §4.2: веса ∈(0,1]; floor-факт пишем в metrics).
- хранение (23 §4.1): edge_index.npy (E×2 int32, узлы = позиции в отсортированном
  списке territory_id), weights.npy (E×24 float32), layer_mask.npy (бит0
  similarity, бит1 geo), snap_t01..24.npz (CSR float32, симметрия, нулевая
  диагональ), skeleton/geo_layer/similarity_layer.npz, node_index.parquet,
  edge_stats.parquet, meta.yaml (гиперпараметры, версии, sha256 входов).

Smoke: sample_n задаётся этапом панели — панель на диске уже сэмплирована,
здесь логика идентична; sample_n лишь фиксируется в meta/metrics.
"""
from __future__ import annotations

import hashlib
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import scipy.sparse as sp
import scipy.stats as st
import yaml
from scipy.sparse.csgraph import connected_components

from .config import Config
from .panel import PARTS
from .runctx import RunContext

WEIGHT_FLOOR = 1e-6  # нижний пол весов (контракт: строго > 0)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _corr(X: np.ndarray) -> np.ndarray:
    """Корреляционная матрица строк X (n,T): стандартизация + один matmul."""
    Z = X - X.mean(axis=1, keepdims=True)
    Z /= Z.std(axis=1, keepdims=True)
    R = Z @ Z.T / Z.shape[1]
    np.fill_diagonal(R, 1.0)
    return R


def _topk_idx(W: np.ndarray, k: int) -> np.ndarray:
    """(n,k) индексы топ-k соседей по строке W, упорядочены по убыванию W
    (позиция 0 → ранг 1). Детерминировано: argpartition introselect + stable sort."""
    n = W.shape[0]
    k = min(k, n - 1)
    part = np.argpartition(-W, kth=k - 1, axis=1)[:, :k]
    w = np.take_along_axis(W, part, axis=1)
    order = np.argsort(-w, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def _pack(a: np.ndarray, b: np.ndarray, n: int) -> np.ndarray:
    """Неориентированный ключ пары: min*n + max (int64, уникален при n < 2^31)."""
    lo, hi = np.minimum(a, b), np.maximum(a, b)
    return lo.astype(np.int64) * n + hi.astype(np.int64)


def _corr_pvalues(r: np.ndarray, df: int) -> np.ndarray:
    """Двусторонний p-value t-теста корреляции: t = |r|·sqrt(df/(1−r²))."""
    rc = np.clip(np.abs(r), 0.0, 1.0 - 1e-12)
    t = rc * np.sqrt(df / (1.0 - rc**2))
    return 2.0 * st.t.sf(t, df)


def _bh_mask(p: np.ndarray, q: float) -> np.ndarray:
    """Benjamini–Hochberg: pass = p ≤ max{p_(i): p_(i) ≤ q·i/m}."""
    m = p.size
    sp_ = np.sort(p)
    below = sp_ <= q * (np.arange(1, m + 1) / m)
    if not below.any():
        return np.zeros(m, bool)
    return p <= sp_[np.nonzero(below)[0].max()]


def _sym_csr(rows: np.ndarray, cols: np.ndarray, data: np.ndarray,
             n: int) -> sp.csr_matrix:
    """Симметричный CSR float32 из неориентированного edge list (i<j, i≠j)."""
    d = np.asarray(data, dtype=np.float32)
    A = sp.csr_matrix(
        (np.concatenate([d, d]),
         (np.concatenate([rows, cols]), np.concatenate([cols, rows]))),
        shape=(n, n))
    A.sum_duplicates()
    return A


def build_graphs(cfg: Config, processed_dir: str | Path = "data/processed",
                 out_dir: str | Path = "data/processed/graphs",
                 ctx: RunContext | None = None) -> dict[str, Path]:
    processed_dir, out_dir = Path(processed_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gcfg = cfg.graph

    panel = pd.read_parquet(processed_dir / "panel_monthly.parquet")
    nodes = pd.read_parquet(processed_dir / "nodes_static.parquet")
    geo_edges = pd.read_parquet(processed_dir / "edges_highway_knn.parquet")

    tids = np.sort(panel.territory_id.unique()).astype(np.int32)
    n = len(tids)
    assert set(nodes.territory_id) == set(tids), "nodes_static расходится с панелью"
    months = np.sort(panel.month_idx.unique())
    T = len(months)

    def _piv(col: str) -> np.ndarray:
        p = panel.pivot(index="territory_id", columns="month_idx", values=col)
        assert list(p.columns) == list(months)
        return p.loc[tids].to_numpy(np.float64)

    log_all = _piv("log_all")                                   # (n, T)
    shares = np.stack([_piv(f"share_{s}") for s in PARTS], axis=2)  # (n, T, 6)

    # --- слой сходства: демеаненные log-приросты -------------------------------
    lr = np.diff(log_all, axis=1)                               # (n, T−1)
    lr_dm = lr - lr.mean(axis=0, keepdims=True)
    R = _corr(lr_dm)
    R_raw = _corr(lr)  # анти-факт: сырые приросты вырождены (23 §0)

    ii, jj = np.triu_indices(n, k=1)
    r_all = R[ii, jj]
    df = (T - 1) - 2                                            # 21 при T=24
    p_all = _corr_pvalues(r_all, df)
    bh = _bh_mask(p_all, gcfg.fdr_q)
    E2 = ii[bh].astype(np.int64) * n + jj[bh].astype(np.int64)

    W = np.clip(R, 0.0, None)
    np.fill_diagonal(W, 0.0)
    k = min(gcfg.knn_k, n - 1)
    nbr = _topk_idx(W, k)
    src = np.repeat(np.arange(n), k)
    rank_map = dict(zip((src * n + nbr.ravel()).tolist(),
                        np.tile(np.arange(1, k + 1), n).tolist()))
    E1 = np.unique(_pack(src, nbr.ravel(), n))

    s = min(gcfg.safety_top, k)
    E3 = np.unique(_pack(np.repeat(np.arange(n), s), nbr[:, :s].ravel(), n))

    E_kf = np.intersect1d(E1, E2, assume_unique=True)
    E_star = np.union1d(E_kf, E3)

    # --- гео-слой: highway kNN, w = exp(−d/λ) ----------------------------------
    lam = gcfg.geo.lambda_km
    tid2row = pd.Series(np.arange(n), index=tids)
    gx = tid2row.loc[geo_edges.tid_x.to_numpy()].to_numpy()
    gy = tid2row.loc[geo_edges.tid_y.to_numpy()].to_numpy()
    gdf = pd.DataFrame({"p": _pack(gx, gy, n),
                        "d": geo_edges.dist_km.to_numpy(np.float64),
                        "fb": geo_edges.is_island_fallback.to_numpy(bool)})
    gdf = gdf.groupby("p", as_index=False).agg(d=("d", "min"), fb=("fb", "max"))
    G = gdf.p.to_numpy()
    dist_map = dict(zip(G.tolist(), gdf.d.tolist()))
    fb_set = set(gdf.p[gdf.fb].tolist())

    # --- объединение слоёв ------------------------------------------------------
    E_total = np.union1d(E_star, G)
    ei = (E_total // n).astype(np.int32)
    ej = (E_total % n).astype(np.int32)
    E = len(E_total)
    in_star = np.isin(E_total, E_star)
    in_geo = np.isin(E_total, G)
    layer_mask = in_star.astype(np.uint8) | (in_geo.astype(np.uint8) << 1)
    d_e = np.array([dist_map.get(p, np.nan) for p in E_total.tolist()])

    # --- месячные веса: CLR-косинус профилей долей ------------------------------
    logp = np.log(np.maximum(shares, cfg.panel.clr_clip))
    clr = logp - logp.mean(axis=2, keepdims=True)
    U = clr / np.linalg.norm(clr, axis=2, keepdims=True)        # (n, T, 6)
    cos = (U[ei] * U[ej]).sum(axis=2)                           # (E, T)
    geo_w = np.exp(-d_e / lam)
    wmat = np.where(in_star[:, None], cos, geo_w[:, None])
    floor_hits = int((wmat < WEIGHT_FLOOR).sum())
    wmat = np.clip(wmat, WEIGHT_FLOOR, 1.0).astype(np.float32)

    # --- топологический самоконтроль --------------------------------------------
    skeleton = _sym_csr(ei, ej, np.ones(E), n)
    deg = np.diff(skeleton.indptr)
    n_comp, _ = connected_components(skeleton, directed=False)
    # согласованность соседних снимков: на E*-рёбрах (сравнимо со спекой 0.949)
    # и на полном E_total (константный geo-якорь инфлирует последний к 1)
    adj_corr = [float(np.corrcoef(wmat[in_star, t], wmat[in_star, t + 1])[0, 1])
                for t in range(T - 1)]
    adj_corr_all = [float(np.corrcoef(wmat[:, t], wmat[:, t + 1])[0, 1])
                    for t in range(T - 1)]
    r_star = R[(E_star // n).astype(np.int64), (E_star % n).astype(np.int64)]

    # --- запись артефактов (23 §4.1) --------------------------------------------
    out: dict[str, Path] = {}

    def _put(name: str, path: Path) -> None:
        out[name] = path

    p = out_dir / "edge_index.npy"
    np.save(p, np.column_stack([ei, ej]).astype(np.int32))
    _put("edge_index", p)
    p = out_dir / "weights.npy"
    np.save(p, wmat)
    _put("weights", p)
    p = out_dir / "layer_mask.npy"
    np.save(p, layer_mask)
    _put("layer_mask", p)

    p = out_dir / "skeleton.npz"
    sp.save_npz(p, skeleton)
    _put("skeleton", p)
    si = (E_star // n).astype(np.int64)
    sj = (E_star % n).astype(np.int64)
    p = out_dir / "similarity_layer.npz"
    sp.save_npz(p, _sym_csr(si, sj, W[si, sj], n))
    _put("similarity_layer", p)
    gi = (G // n).astype(np.int64)
    gj = (G % n).astype(np.int64)
    p = out_dir / "geo_layer.npz"
    sp.save_npz(p, _sym_csr(gi, gj, np.exp(-gdf.d.to_numpy() / lam), n))
    _put("geo_layer", p)
    for t in range(T):
        p = out_dir / f"snap_t{t + 1:02d}.npz"
        sp.save_npz(p, _sym_csr(ei, ej, wmat[:, t], n))
        _put(f"snap_t{t + 1:02d}", p)

    p = out_dir / "node_index.parquet"
    pd.DataFrame({"row_idx": np.arange(n, dtype=np.int32),
                  "territory_id": tids}).to_parquet(p, index=False)
    _put("node_index", p)

    e_list = E_total.tolist()
    r_e = R[ei.astype(np.int64), ej.astype(np.int64)]
    stats = pd.DataFrame({
        "row_i": ei, "row_j": ej,
        "tid_x": tids[ei], "tid_y": tids[ej],
        "r": r_e.astype(np.float32),
        "p": _corr_pvalues(r_e, df).astype(np.float32),
        "bh_pass": np.isin(E_total, E2),
        "in_knn_e1": np.isin(E_total, E1),
        "in_safety_e3": np.isin(E_total, E3),
        "is_similarity": in_star, "is_geo": in_geo,
        "knn_rank_xy": pd.array([rank_map.get(kk) for kk in e_list], dtype="Int8"),
        "knn_rank_yx": pd.array(
            [rank_map.get(int(b) * n + int(a)) for a, b in
             zip(ei.tolist(), ej.tolist())], dtype="Int8"),
        "dist_km": np.where(in_geo, d_e, np.nan).astype(np.float32),
        "is_island_fallback": np.array([kk in fb_set for kk in e_list]),
    })
    p = out_dir / "edge_stats.parquet"
    stats.to_parquet(p, index=False)
    _put("edge_stats", p)

    inputs = {name: processed_dir / f"{name}.parquet"
              for name in ("panel_monthly", "nodes_static", "edges_highway_knn")}
    meta = {
        "stage": "03_build_graphs (research/23)",
        "built": datetime.now().isoformat(timespec="seconds"),
        "similarity": gcfg.similarity,
        "demeanging": "cross_sectional_by_month: lr_dm[i,t] = lr[i,t] − mean_i lr[·,t]",
        "knn_k": int(gcfg.knn_k), "knn_sym": gcfg.knn_sym,
        "fdr_q": float(gcfg.fdr_q), "fdr_df": int(df),
        "safety_top": int(gcfg.safety_top),
        "geo": {"enabled": bool(gcfg.geo.enabled), "knn_k": int(gcfg.geo.knn_k),
                "lambda_km": float(lam)},
        "multiplex_omega": 1.0,  # решение 23 §2.5 (multilayer-Leiden)
        "weights": ("CLR-косинус профилей долей месяца t на E_total; geo-only "
                    "рёбра — константа exp(−d/λ); клип [1e-6, 1]"),
        "seed": int(cfg.seed), "sample_n": cfg.data.sample_n,
        "n_nodes": int(n), "n_months": int(T),
        "edges_star": int(len(E_star)), "edges_geo": int(len(G)),
        "edges_total": int(E),
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__,
                     "pandas": pd.__version__, "scipy": scipy.__version__},
        "sha256": {name: _sha256(path) for name, path in inputs.items()},
    }
    p = out_dir / "meta.yaml"
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, allow_unicode=True, sort_keys=False)
    _put("meta", p)

    metrics = {
        "n_nodes": int(n), "n_months": int(T),
        "edges_e1_knn_union": int(len(E1)),
        "pairs_total": int(len(r_all)),
        "edges_e2_fdr": int(bh.sum()),
        "share_pairs_bh_pass": round(float(bh.mean()), 6),
        "eff_abs_r_threshold_bh": round(float(np.abs(r_all[bh]).min()), 4)
        if bh.any() else None,
        "edges_e1_intersect_fdr": int(len(E_kf)),
        "edges_safety_added": int(len(E_star) - len(E_kf)),
        "edges_star": int(len(E_star)),
        "edges_geo": int(len(G)),
        "edges_both_layers": int((in_star & in_geo).sum()),
        "edges_total": int(E),
        "isolates": int((deg == 0).sum()),
        "n_components": int(n_comp),
        "deg_min": int(deg.min()), "deg_med": float(np.median(deg)),
        "deg_max": int(deg.max()),
        "mean_r_star": round(float(r_star.mean()), 4),
        "min_r_star": round(float(r_star.min()), 4),
        "med_r_raw_growth_antifact": round(float(np.median(R_raw[ii, jj])), 4),
        "med_r_demeanded_antifact": round(float(np.median(r_all)), 4),
        "weights_adj_corr_mean": round(float(np.mean(adj_corr)), 4),
        "weights_adj_corr_min": round(float(np.min(adj_corr)), 4),
        "weights_adj_corr_all_mean": round(float(np.mean(adj_corr_all)), 4),
        "weights_adj_corr_all_min": round(float(np.min(adj_corr_all)), 4),
        "weights_floor_hits": floor_hits,
        "sample_n": cfg.data.sample_n,
        "selfcheck_spec23": {"edges_star": 12_900, "isolates": 0, "n_components": 1,
                             "mean_r_star": 0.737, "weights_adj_corr": 0.949,
                             "med_r_raw": 0.906},
    }
    if ctx is not None:
        ctx.log(f"E* = {len(E_star)} (kNN∪ = {len(E1)}, ∩FDR = {len(E_kf)}, "
                f"safety +{len(E_star) - len(E_kf)}); geo = {len(G)}; "
                f"E_total = {E}; изоляты = {(deg == 0).sum()}; "
                f"компоненты = {n_comp}; mean r(E*) = {r_star.mean():.3f}; "
                f"corr весов соседних (E*) = {np.mean(adj_corr):.3f}")
        ctx.write_metrics(metrics)
    return out
