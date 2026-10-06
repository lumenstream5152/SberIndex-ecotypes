"""06a: решающий эксперимент мер сходства (research/24) — 14 мер (M1–M11 +
X1–X3) → единый kNN(10)-union графоформат + гейт приемлемости (bootstrap-J ≥ 0.25).

Выходы в <out>/measures/:
- {M1..M11}.npz — edge_index (E×2 int32, i<j), weights (E float32, = сходство),
  territory_id (порядок строк); M4_leadlag.npz — ориентированная лаг-карта.
- table_A_gate.parquet — все 14 мер: mean/sd сходства, bootstrap-J, вердикт.
- table_topology.parquet — 11 графов: рёбра, степени, изоляты, % гиганта,
  Jaccard к M2, доля рёбер внутри субъекта РФ.
- jaccard_heatmap.parquet — 11×11 пересечения kNN-рёбер (фигура 1).
metrics.json — в outputs/<run_id>/ через RunContext.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from ecotypes import measures as ms
from ecotypes.config import load_config, load_prereg
from ecotypes.runctx import RunContext
from ecotypes.seeds import set_all_seeds

SCORING = [f"M{i}" for i in range(1, 12)]
ANTI = ["X1", "X2", "X3"]


def _sim_stats(res: ms.SimResult, n: int) -> tuple[float, float]:
    """mean/sd сходства по верхнему треугольнику; для edge-мер (M8) — по рёбрам."""
    if res.edges is not None:
        w = res.edges[2]
        return float(w.mean()), float(w.std())
    iu = np.triu_indices(n, 1)
    v = res.matrix[iu]
    return float(v.mean()), float(v.std())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--out", default="data/processed",
                    help="каталог панели; меры пишутся в <out>/measures/")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    set_all_seeds(cfg.seed)
    sc = load_prereg()["similarity"]
    k = int(sc["knn_k"])
    B = int(sc["gate"]["bootstraps"])
    thr = float(sc["gate"]["bootstrap_jaccard_min"])
    alpha_grid = [float(a) for a in sc["m9_alpha_grid"]]

    name = args.config.split("/")[-1].removesuffix(".yaml")
    if args.overrides:
        name += "_" + args.overrides.split("/")[-1].removesuffix(".yaml")
    ctx = RunContext(cfg, config_name=name, stage="06a_measures")

    t0 = time.time()
    data = ms.from_processed(args.out)
    ctx.log(f"данные: n={data.n}, T={data.T}; k={k}, B={B}, порог={thr}")

    # --- M9: α по ноге T (odd/even — split-half вырожден на T=24, см. ниже),
    # ДО любого композита (24 §4.4) ------------------------------------------------
    t_a = time.time()
    alpha, alpha_js = ms.m9_alpha_split_half(data, alpha_grid, k)
    ctx.log(f"M9: odd/even Jaccard по α: "
            f"{ {a: round(j, 4) for a, j in alpha_js.items()} } → α*={alpha} "
            f"({time.time() - t_a:.1f}s)")

    fns = {
        "X1": ms.sim_X1, "X2": ms.sim_X2, "X3": ms.sim_X3,
        "M1": ms.sim_M1, "M2": ms.sim_M2, "M3": ms.sim_M3, "M4": ms.sim_M4,
        "M5": ms.sim_M5, "M6": ms.sim_M6, "M7": ms.sim_M7, "M8": ms.sim_M8,
        "M9": (lambda d, lvl_idx=None, grw_idx=None:
               ms.sim_M9(d, lvl_idx=lvl_idx, grw_idx=grw_idx, alpha=alpha)),
        "M10": (lambda d, lvl_idx=None, grw_idx=None:
                ms.sim_M10(d, lvl_idx=lvl_idx, grw_idx=grw_idx, k=k)),
        "M11": ms.sim_M11,
    }

    out_dir = Path(args.out) / "measures"
    out_dir.mkdir(parents=True, exist_ok=True)

    table_a: list[dict] = []
    topo: list[dict] = []
    edge_sets: dict[str, np.ndarray] = {}
    timings: dict[str, float] = {}

    for mname in ANTI + SCORING:
        fn = fns[mname]
        t_m = time.time()
        res = fn(data)
        mean_s, sd_s = _sim_stats(res, data.n)

        if mname in SCORING:
            ei, w = ms.knn_union_graph(res, k)
            np.savez_compressed(out_dir / f"{mname}.npz", edge_index=ei, weights=w,
                                territory_id=data.tids)
            edge_sets[mname] = ei
            if mname == "M4":
                np.savez_compressed(out_dir / "M4_leadlag.npz",
                                    lead_lag=res.extra["lead_lag"].astype(np.int8),
                                    territory_id=data.tids)
            st = ms.topology_stats(ei, data.n, region=data.region)
            st["measure"] = mname
            topo.append(st)

        j_mean, j_sd, _ = ms.bootstrap_jaccard(fn, data, k, B,
                                               seed=ms.measure_seed(cfg.seed, mname))
        verdict = ms.gate_verdict(j_mean, thr)
        table_a.append({"measure": mname, "kind": "anti" if mname in ANTI else "scoring",
                        "mean_sim": round(mean_s, 6), "sd_sim": round(sd_s, 6),
                        "boot_jaccard": round(j_mean, 4),
                        "boot_jaccard_sd": round(j_sd, 4),
                        "verdict": verdict})
        timings[mname] = time.time() - t_m
        ctx.log(f"{mname}: mean={mean_s:.4f} sd={sd_s:.4f} bootJ={j_mean:.3f} "
                f"→ {verdict} ({timings[mname]:.1f}s)")

    # --- Jaccard к M2 + heatmap 11×11 ------------------------------------------------
    for st in topo:
        st["jaccard_to_M2"] = round(ms.edge_set_jaccard(edge_sets[st["measure"]],
                                                        edge_sets["M2"], data.n), 4)
    heat = pd.DataFrame(
        [[ms.edge_set_jaccard(edge_sets[a], edge_sets[b], data.n) for b in SCORING]
         for a in SCORING], index=SCORING, columns=SCORING)

    # --- запись таблиц ------------------------------------------------------------------
    pa = out_dir / "table_A_gate.parquet"
    pd.DataFrame(table_a).to_parquet(pa, index=False)
    pt = out_dir / "table_topology.parquet"
    pd.DataFrame(topo)[["measure", "edges", "deg_mean", "deg_min", "deg_med",
                        "deg_max", "isolates", "giant_pct", "within_region_share",
                        "jaccard_to_M2"]].to_parquet(pt, index=False)
    ph = out_dir / "jaccard_heatmap.parquet"
    heat.reset_index(names="measure").to_parquet(ph, index=False)

    metrics = {
        "n_nodes": int(data.n), "n_months": int(data.T),
        "knn_k": k, "gate": {"bootstraps": B, "threshold": thr,
                             "stratify": "calendar_month"},
        "m9_alpha_chosen": alpha,
        "m9_alpha_leg": "odd_even (split-half 12/12 вырожден на T=24: d[12+j] ≡ −d[j])",
        "m9_alpha_split_half_jaccards": {str(a): round(j, 4) for a, j in alpha_js.items()},
        "table_A": table_a,
        "topology": topo,
        "jaccard_heatmap": heat.round(4).to_dict(),
        "timings_s": {m: round(t, 1) for m, t in timings.items()},
        "total_s": round(time.time() - t0, 1),
        "deviations": [
            "M11': внешний пакет не встал (epfl-lts2/graph-learning — 404; "
            "rodrigo-pena/graph-learning — не pip-пакет, uv add отказал). Своя "
            "реализация log-модели Kalofolias 2016 (проекционный градиент, "
            "α=1, β=1 фикс до прогона). Требует записи в PREREG_DEVIATIONS.md.",
            "M8: d_eff = max(d, 1 км) — в edges_highway_knn есть d=0 "
            "(совпадающие центроиды); веса нормированы на max.",
            "M9: α выбран по odd/even-половинам (пререг secondary-вариант ноги T) "
            "вместо split-half 1–12/13–24: при T=24 с month-of-year "
            "десезонизацией sa[t+12] ≡ −sa[t] → приросты половин точно "
            "анти-равны, corr-матрицы совпадают, split-half Jaccard = 1.0 для "
            "любого α — критерий пуст. То же тождество делает первичную T-ногу "
            "скоринга вырожденной для всех sa-мер (M1–M4, M9, M10) — флаг "
            "для этапа скоринга. Требует записи в PREREG_DEVIATIONS.md.",
            "M4 lead_lag: исправлена инверсия знака относительно docstring "
            "(теперь ℓ>0 = j лидирует i на ℓ мес).",
        ],
    }
    ctx.write_metrics(metrics)
    n_pass = sum(r["verdict"] == "pass" for r in table_a)
    ctx.log(f"гейт: {n_pass}/14 pass; итого {time.time() - t0:.0f}s → {out_dir}/")
    ctx.close()


if __name__ == "__main__":
    main()
