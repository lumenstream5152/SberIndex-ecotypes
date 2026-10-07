"""06c: нулевой бутстреп ICVI прод-типологии (permutation baseline).

Протокол статей жюри (ESWA-2026): z = (obs − mean_null)/std_null и percentile
по 200 перестановкам меток (кардинальности типов сохраняются). Закрывает пробел,
найденный злым жюри JR1/J1: permutation_baseline был реализован, но нигде
не вызывался — z-scores против случайных разбиений не существовали как артефакт.

Входы: прод-разбиение leiden_consensus из последнего run 04 (gamma_star),
X_static (04), A = similarity ∪ geo слои графа M2. Выход: outputs/<run_id>/
icvi_null.parquet (index × {obs, null_mean, null_std, z, percentile, direction}).
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ecotypes import cluster as cl
from ecotypes import icvi
from ecotypes.config import load_config
from ecotypes.interpret import find_run
from ecotypes.runctx import RunContext
from ecotypes.seeds import set_all_seeds, stage_seed

N_PERM = 200


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--out", default="data/processed")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    cfg = load_config(args.config, overrides=args.overrides)
    set_all_seeds(cfg.seed)

    name = args.config.split("/")[-1].removesuffix(".yaml")
    if args.overrides:
        name += "_" + args.overrides.split("/")[-1].removesuffix(".yaml")
    ctx = RunContext(cfg, config_name=name, stage="06c_icvi_null")
    t0 = time.time()
    root = Path(args.out)

    run04 = find_run(Path("outputs"), "gamma_star", require="labels.parquet")
    lab04 = pd.read_parquet(run04 / "labels.parquet")
    lab04 = lab04.sort_values("row_idx").reset_index(drop=True)
    if "leiden_consensus" not in lab04.columns:
        raise RuntimeError(f"в {run04}/labels.parquet нет leiden_consensus")
    labels = lab04["leiden_consensus"].to_numpy()
    ctx.log(f"прод-разбиение из {run04.name}: k={len(np.unique(labels))}")

    nidx = pd.read_parquet(root / "graphs" / "node_index.parquet")
    nidx = nidx.sort_values("row_idx").reset_index(drop=True)
    nodes = nidx[["territory_id"]].merge(pd.read_parquet(root / "nodes_static.parquet"),
                                         on="territory_id")
    X, _ = cl.feature_matrix(nodes, cfg)
    A = sp.load_npz(root / "graphs" / "similarity_layer.npz").maximum(
        sp.load_npz(root / "graphs" / "geo_layer.npz"))

    tasks = [  # (имя, func, args, higher_better)
        ("SW", icvi.sw, (X,), True),
        ("CH_over_N", icvi.ch_over_n, (X,), True),
        ("S_Dbw", icvi.sdbw, (X,), False),
        ("AVI", icvi.avi, (A,), True),
        ("AVU", icvi.avu, (A,), False),
        ("MQ", lambda a, lab: icvi.mq(a, lab, variant="mancoridis"), (A,), True),
    ]
    rows = []
    for nm, fn, a, hb in tasks:
        obs = float(fn(*a, labels))
        r = icvi.permutation_baseline(
            fn, *a, labels=labels, runs=N_PERM,
            rng=np.random.default_rng(stage_seed(cfg.seed, f"icvi_null:{nm}")),
            higher_better=hb)
        rows.append(dict(index=nm, higher_better=hb, n_perm=N_PERM, obs=obs, **r))
        ctx.log(f"{nm}: obs={obs:.4f}, null {r['null_mean']:.4f}±"
                f"{r['null_std']:.4f}, z={r['z']:.2f}, pct={r['percentile']:.3f}")
    df = pd.DataFrame(rows)
    df.to_parquet(ctx.dir / "icvi_null.parquet", index=False)
    ctx.write_metrics({"n_perm": N_PERM, "labels_run04": run04.name,
                       "elapsed_s": round(time.time() - t0, 1)})
    ctx.close()


if __name__ == "__main__":
    main()
