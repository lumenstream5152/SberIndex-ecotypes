"""04: кластеризация (research/25/26/27) → labels.parquet, plateau_table, metrics.json."""
from __future__ import annotations
import argparse, time
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from ecotypes import cluster as cl, icvi
from ecotypes.config import load_config
from ecotypes.runctx import RunContext
from ecotypes.seeds import set_all_seeds
from ecotypes.synthetic import ari, nmi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--out", default="data/processed", help="графы из <out>/graphs/")
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    set_all_seeds(cfg.seed)
    name = args.config.split("/")[-1].removesuffix(".yaml")
    if args.overrides: name += "_" + args.overrides.split("/")[-1].removesuffix(".yaml")
    ctx, t0 = RunContext(cfg, config_name=name, stage="04_cluster"), time.time()
    gdir = Path(args.out) / "graphs"
    sim, geo = (sp.load_npz(gdir / f) for f in ("similarity_layer.npz", "geo_layer.npz"))
    A = sim.maximum(geo)  # взвешенный граф на поддержке skeleton (sim ∪ geo = E_total)
    nidx = pd.read_parquet(gdir / "node_index.parquet")
    nodes = nidx[["territory_id"]].merge(
        pd.read_parquet(Path(args.out) / "nodes_static.parquet"), on="territory_id")
    X, feat = cl.feature_matrix(nodes, cfg)
    seeds = list(range(cfg.cluster.leiden.seeds_per_snapshot))
    # X в плато: выбор ВНУТРИ бюджетных плато — по ICVI-композиту (25 §3.2 п.5)
    plat = cl.leiden_plateau(A, cfg.cluster.leiden.gammas_sweep, seeds, X=X)
    plat["table"].to_parquet(ctx.dir / "plateau_table.parquet", index=False)
    gamma_star = plat["gamma_star"] or cfg.cluster.leiden.gamma  # деградация 25 §3.2
    labels = {"leiden_consensus": cl.leiden_consensus_cspa(A, gamma=gamma_star, seeds=seeds)}
    zoo, K = cl.run_feature_zoo(X, cfg), cfg.cluster.kefrin.k
    labels |= {f"{m}_k{K}": dd[K] for m, dd in zoo.items()}
    labels["kefrin"] = cl.run_kefrin(A, X, K=K, rho=cfg.cluster.kefrin.rho,
                                     xi=cfg.cluster.kefrin.xi, seed=cfg.seed)
    labels["louvain"], labels["infomap"] = cl.run_louvain(A, seed=cfg.seed), cl.run_infomap(A, seed=cfg.seed)
    clr = [c for c in cl.FEATURE_COLS if c.startswith("clr_mean")]
    labels["eva"] = cl.run_eva(A, nodes[clr].to_numpy().argmax(1), alpha=cfg.cluster.eva.alpha)
    dm = cl.run_dmon(A, X, K=K, seed=cfg.seed)
    if dm is not None: labels["dmon"] = dm

    # стабильность консенсуса: ARI к одиночным прогонам при γ* (25 §3.3 аудит)
    singles = [cl.run_leiden(A, gamma=gamma_star, seed=s) for s in seeds]
    cons_ari = [ari(labels["leiden_consensus"], s) for s in singles]
    row_star = plat["table"].iloc[(plat["table"].gamma - gamma_star).abs().argmin()]
    n = A.shape[0]
    cnt = np.bincount(labels["leiden_consensus"])
    metrics = {
        "gamma_star": gamma_star, "plateau": plat["plateau"], "fallback": plat["fallback"],
        "features": feat, "gmm_bic": cl.gmm_bic(X, cfg.cluster.kmeans.ks, cfg.seed),
        "plateau_at_gamma_star": {c: float(row_star[c]) for c in plat["table"].columns},
        "consensus": {
            "ari_to_singles_med": float(np.median(cons_ari)),
            "ari_to_singles_min": float(np.min(cons_ari)),
            "seed_ari_med": float(row_star.ari_med),   # парный ARI между seed'ами
            "seed_ari_min": float(row_star.ari_min),
            "singleton_share": float((cnt == 1).mean()),  # красные флаги 25 §6
            "max_share": float(cnt.max() / n),
        },
        "methods": {},
    }
    out = nidx.copy()
    for m, lab in labels.items():
        out[m] = lab
        k = len(np.unique(lab))
        metrics["methods"][m] = {"k": int(k), "Q": round(icvi.q_modularity(A, lab), 4),
                                 "SW": round(icvi.sw(X, lab), 4) if 1 < k < len(lab) else None}
    # согласие методов: парная NMI-матрица (25 §6.5 / 27 §3)
    names = list(labels)
    metrics["nmi_pairwise"] = {
        a: {b: round(nmi(labels[a], labels[b]), 4) for b in names} for a in names}
    nmi_vals = [nmi(labels[a], labels[b]) for a, b in combinations(names, 2)]
    metrics["nmi_pairwise_summary"] = {"med": float(np.median(nmi_vals)),
                                       "min": float(np.min(nmi_vals)),
                                       "max": float(np.max(nmi_vals))}
    out.to_parquet(ctx.dir / "labels.parquet", index=False); ctx.write_metrics(metrics)
    ctx.log(f"γ*={gamma_star}, плато={plat['plateau']}, методов={len(labels)}, "
            f"NMI(med/min/max)={metrics['nmi_pairwise_summary']['med']:.3f}/"
            f"{metrics['nmi_pairwise_summary']['min']:.3f}/"
            f"{metrics['nmi_pairwise_summary']['max']:.3f}; "
            f"{time.time() - t0:.1f}s → {ctx.dir}")
    ctx.close()


if __name__ == "__main__":
    main()
