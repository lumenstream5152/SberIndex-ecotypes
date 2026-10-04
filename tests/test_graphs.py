"""Тесты графового этапа (research/23): инварианты на синтетической PP-Dir
панели, самоконтроль на реальной панели (requires_raw), детерминизм."""
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from ecotypes.config import load_config
from ecotypes.graphs import build_graphs
from ecotypes.panel import PARTS

from conftest import requires_raw

GEO_KNN_K = 10  # kNN-слой синтетического гео (аналог edges_highway_knn)


def _write_processed(mini: dict, proc) -> None:
    """Материализует PP-Dir dict (X доли (T,n,6), V объёмы (T,n), geo (n,2))
    как три parquet этапа 02: panel_monthly / nodes_static / edges_highway_knn."""
    proc.mkdir(parents=True, exist_ok=True)
    X, V, geo = mini["X"], mini["V"], mini["geo"]
    T, n, _ = X.shape
    tids = (np.arange(n) + 1).astype(np.int32)
    months = [str(p) for p in pd.period_range("2023-01", periods=T, freq="M")]

    panel = pd.DataFrame({
        "territory_id": np.tile(tids, T),
        "month": np.repeat(months, n),
        "month_idx": np.repeat(np.arange(T, dtype=np.int8), n),
        "log_all": np.log(V.reshape(T * n)).astype(np.float32),
        **{f"share_{s}": X[:, :, j].reshape(T * n).astype(np.float32)
           for j, s in enumerate(PARTS)},
    })
    panel.to_parquet(proc / "panel_monthly.parquet", index=False)
    pd.DataFrame({"territory_id": tids}).to_parquet(
        proc / "nodes_static.parquet", index=False)

    # гео-слой: kNN по евклидовым дистанциям латентных координат + симметризация
    # зеркалами — та же схема, что _build_edges этапа 02
    g = np.asarray(geo, dtype=np.float64)
    D = np.sqrt(((g[:, None, :] - g[None, :, :]) ** 2).sum(axis=2))
    np.fill_diagonal(D, np.inf)
    k = min(GEO_KNN_K, n - 1)
    part = np.argpartition(D, kth=k - 1, axis=1)[:, :k]
    order = np.argsort(np.take_along_axis(D, part, axis=1), axis=1, kind="stable")
    nbr = np.take_along_axis(part, order, axis=1)
    rows = np.repeat(np.arange(n), k)
    edges = pd.DataFrame({
        "tid_x": tids[rows], "tid_y": tids[nbr.ravel()],
        "dist_km": D[rows, nbr.ravel()].astype(np.float32),
        "rank": np.tile(np.arange(1, k + 1, dtype=np.int8), n),
    })
    rev = edges.rename(columns={"tid_x": "tid_y", "tid_y": "tid_x"})
    mirrors = rev.merge(edges[["tid_x", "tid_y"]].assign(_f=1),
                        on=["tid_x", "tid_y"], how="left")
    edges = pd.concat([edges, mirrors[mirrors._f.isna()].drop(columns=["_f"])],
                      ignore_index=True)
    edges["is_island_fallback"] = False
    edges = edges.sort_values(["tid_x", "rank", "tid_y"]).reset_index(drop=True)
    edges.to_parquet(proc / "edges_highway_knn.parquet", index=False)


@pytest.fixture(scope="module")
def synth_graphs(mini_panel, tmp_path_factory):
    d = tmp_path_factory.mktemp("graphs_syn")
    proc, out = d / "processed", d / "graphs"
    _write_processed(mini_panel, proc)
    cfg = load_config("configs/default.yaml")
    return build_graphs(cfg, processed_dir=proc, out_dir=out)


def test_synth_invariants(synth_graphs, mini_panel):
    paths = synth_graphs
    n = mini_panel["meta"]["n"]
    ei = np.load(paths["edge_index"])
    w = np.load(paths["weights"])
    mask = np.load(paths["layer_mask"])
    E = len(ei)
    assert ei.dtype == np.int32 and w.dtype == np.float32
    assert (ei[:, 0] < ei[:, 1]).all()          # каноничный порядок, нет самопетель
    assert w.shape == (E, 24)
    assert ((w > 0) & (w <= 1.0)).all()
    assert set(np.unique(mask)) <= {1, 2, 3}    # бит0 similarity, бит1 geo

    node_index = pd.read_parquet(paths["node_index"])
    assert list(node_index.territory_id) == sorted(node_index.territory_id)
    assert list(node_index.row_idx) == list(range(n))

    skel = sp.load_npz(paths["skeleton"])
    for t in range(1, 25):
        A = sp.load_npz(paths[f"snap_t{t:02d}"])
        assert A.shape == (n, n) and A.dtype == np.float32
        assert (A - A.T).nnz == 0               # симметрия CSR
        assert (A.diagonal() == 0).all()        # нулевая диагональ
        assert ((A.data > 0) & (A.data <= 1)).all()
        # все снимки — один каркас E_total
        assert np.array_equal(A.indptr, skel.indptr)
        assert np.array_equal(A.indices, skel.indices)
    deg = np.diff(skel.indptr)
    assert deg.min() >= min(GEO_KNN_K, 3)       # safety net: out-degree ≥ 3
    n_comp, _ = connected_components(skel, directed=False)
    assert n_comp == 1


def test_determinism(mini_panel, tmp_path):
    proc = tmp_path / "processed"
    _write_processed(mini_panel, proc)
    cfg = load_config("configs/default.yaml")
    p1 = build_graphs(cfg, processed_dir=proc, out_dir=tmp_path / "g1")
    p2 = build_graphs(cfg, processed_dir=proc, out_dir=tmp_path / "g2")
    assert np.array_equal(np.load(p1["edge_index"]), np.load(p2["edge_index"]))
    assert np.array_equal(np.load(p1["weights"]), np.load(p2["weights"]))
    assert np.array_equal(np.load(p1["layer_mask"]), np.load(p2["layer_mask"]))
    pd.testing.assert_frame_equal(pd.read_parquet(p1["edge_stats"]),
                                  pd.read_parquet(p2["edge_stats"]))


@pytest.fixture(scope="module")
def real_graphs(tmp_path_factory):
    cfg = load_config("configs/default.yaml")
    paths = build_graphs(cfg, out_dir=tmp_path_factory.mktemp("graphs_real"))
    return paths, cfg


@requires_raw
def test_real_selfcheck(real_graphs):
    """Самоконтроль 23 §1.3/§3.2 на реальной панели 2016 МО."""
    paths, _ = real_graphs
    w = np.load(paths["weights"])
    mask = np.load(paths["layer_mask"])
    stats = pd.read_parquet(paths["edge_stats"])

    n_star = int((mask & 1).sum())              # рёбра E* (бит0)
    assert 11_000 <= n_star <= 15_000           # спека: ≈12 900
    r_star = stats.loc[stats.is_similarity, "r"].to_numpy()
    assert 0.65 <= r_star.mean() <= 0.85        # спека: 0.737
    assert (r_star > 0).all()                   # доля r<0 на E* = 0

    A = sp.load_npz(paths["snap_t01"])
    assert np.diff(A.indptr).min() > 0          # 0 изолятов
    n_comp, _ = connected_components(A, directed=False)
    assert n_comp == 1

    adj = [np.corrcoef(w[:, t], w[:, t + 1])[0, 1] for t in range(w.shape[1] - 1)]
    assert np.mean(adj) > 0.9
    sim = (mask & 1).astype(bool)               # сравнимо со спековским 0.949 (на E*)
    adj_sim = [np.corrcoef(w[sim, t], w[sim, t + 1])[0, 1]
               for t in range(w.shape[1] - 1)]
    assert 0.85 < np.mean(adj_sim) < 0.99


@requires_raw
def test_real_determinism(real_graphs, tmp_path):
    paths, cfg = real_graphs
    p2 = build_graphs(cfg, out_dir=tmp_path / "graphs2")
    assert np.array_equal(np.load(paths["edge_index"]), np.load(p2["edge_index"]))
    assert np.array_equal(np.load(paths["weights"]), np.load(p2["weights"]))
    assert np.array_equal(np.load(paths["layer_mask"]), np.load(p2["layer_mask"]))
