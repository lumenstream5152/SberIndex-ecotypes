"""06b: сетка синтетической валидации методов (PREREG §B method.synthetic).

Сетки (числа — из configs/prereg.yaml, не литералы):
- A: K∈{4,6,8} × drift∈{0,0.05,0.10}, α_btw=80, μ_edge=0.25 — 15 реплик/ячейка;
- B: K=6, δ=0.05, α_btw∈{40,160};
- C: LFR mu∈{0.3,0.4} (генератор networkx LFR + Dir-атрибуты, K = n_communities).
Лёгкие методы — все ячейки; тяжёлые (EVA/DMoN) — только центральные ячейки
(A: K=6 δ=0.05; C: mu=0.3), ≤5 реплик (prereg §B). DMoN — честный skip без TF.

Метрики (ecotypes.synthetic): NMI и ARI против модальной истины, per-snapshot
NMI (статические метки, тайлом), F1 детекции смены типа (окно ±1 мес). Дрейф-
метрики — отчётный блок, в композит НЕ входят (prereg §B).

Выход: outputs/synth_grid/synth_grid_<ts>.parquet (длинная таблица
cell×replica×method) + synth_grid_summary_<ts>.parquet (медианы по ячейке)
+ synth_grid_meta_<ts>.json.

Верификация (перед ночным прогоном): --cells central --grids A --replicas 3.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from ecotypes import benchmark as bm
from ecotypes.config import load_config, load_prereg
from ecotypes.seeds import set_all_seeds, stage_seed
from ecotypes.synthetic import (ari, f1_type_switch, make_lfr_dir,
                                make_ppdataset, nmi, per_snapshot_nmi)

log = logging.getLogger("06b")

OUT_DIR = Path("outputs/synth_grid")


def _cells(grid: str, prereg_syn: dict, mode: str) -> list[dict]:
    """Ячейки сетки. mode: 'central' — только центральные (A: K=6 δ=0.05;
    B: не определён как центральный — пропускается; C: mu=0.3), 'all' — все."""
    ga, gb = prereg_syn["grid_A"], prereg_syn["grid_B"]
    if grid == "A":
        all_cells = [dict(grid="A", K=int(K), drift=float(d),
                          alpha_btw=float(ga["alpha_btw"]), mu_edge=float(ga["mu_edge"]))
                     for K in ga["K"] for d in ga["drift"]]
        central = [c for c in all_cells if c["K"] == 6 and c["drift"] == 0.05]
        return central if mode == "central" else all_cells
    if grid == "B":
        all_cells = [dict(grid="B", K=int(gb["K"]), drift=float(gb["drift"]),
                          alpha_btw=float(a), mu_edge=float(ga["mu_edge"]))
                     for a in gb["alpha_btw"]]
        return [] if mode == "central" else all_cells
    if grid == "C":
        all_cells = [dict(grid="C", lfr_mu=float(mu)) for mu in prereg_syn["grid_C_lfr_mu"]]
        central = [c for c in all_cells if c["lfr_mu"] == 0.3]
        return central if mode == "central" else all_cells
    raise ValueError(grid)


def _make_instance(cell: dict, seed: int, n: int, T: int) -> dict:
    if cell["grid"] == "C":
        return make_lfr_dir(seed=seed, n=n, mu=cell["lfr_mu"], T=T)
    return make_ppdataset(seed=seed, K=cell["K"], n=n, T=T,
                          alpha_btw=cell["alpha_btw"], delta=cell["drift"],
                          mu_edge=cell["mu_edge"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--replicas", type=int, default=None,
                    help="переопределить prereg method.synthetic.replicas")
    ap.add_argument("--cells", default="all",
                    choices=["central", "all", "центральная", "все"],
                    help="central=только центральные ячейки (верификация)")
    ap.add_argument("--grids", default="ABC", help="подмножество из A,B,C")
    ap.add_argument("--n", type=int, default=2016)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")

    cfg = load_config(args.config, overrides=args.overrides)
    set_all_seeds(cfg.seed)
    psyn = load_prereg()["method"]["synthetic"]
    replicas = args.replicas or int(psyn["replicas"])
    mode = {"центральная": "central", "все": "all"}.get(args.cells, args.cells)

    cells = [c for g in args.grids for c in _cells(g, psyn, mode)]
    if not cells:
        raise SystemExit("пустой набор ячеек (проверь --cells/--grids)")
    log.info("ячеек: %d, реплик(лёгкие): %d, тяжёлые ≤5", len(cells), replicas)

    rows: list[dict] = []
    t0 = time.time()
    for cell in cells:
        cell_key = (f"{cell['grid']}:K{cell.get('K', '-')}:d{cell.get('drift', '-')}"
                    f":a{cell.get('alpha_btw', '-')}:mu{cell.get('lfr_mu', '-')}")
        is_central = (cell["grid"] == "A" and cell.get("K") == 6
                      and cell.get("drift") == 0.05) or \
                     (cell["grid"] == "C" and cell.get("lfr_mu") == 0.3)
        methods = list(bm.METHODS_LIGHT)
        if is_central:
            methods += bm.METHODS_HEAVY
        for rep in range(replicas):
            seed = int(stage_seed(cfg.seed, f"synth:{cell_key}:{rep}"))
            ds = _make_instance(cell, seed, args.n, 24)
            K_true = (int(ds["meta"]["n_communities"]) if cell["grid"] == "C"
                      else int(cell["K"]))
            X = bm.synthetic_feature_matrix(ds)
            z_true = bm.modal_labels(ds["z"])
            for mname in methods:
                if mname in bm.METHODS_HEAVY and rep >= 5:
                    continue  # тяжёлые — ≤5 реплик (prereg §B)
                tm = time.time()
                lab = bm.run_method(mname, ds["A"], X, K_true, seed, cfg)
                if lab is None:
                    rows.append(dict(cell=cell_key, replica=rep, method=mname,
                                     skipped="unavailable (dmon/TF)"))
                    continue
                tiled = np.tile(lab, (24, 1))
                rows.append(dict(
                    cell=cell_key, replica=rep, method=mname,
                    nmi=nmi(z_true, lab), ari=ari(z_true, lab),
                    per_snapshot_nmi_mean=float(per_snapshot_nmi(ds["z"], tiled).mean()),
                    f1_type_switch=f1_type_switch(ds["z"], tiled, window=1),
                    fit_s=round(time.time() - tm, 2)))
            log.info("%s rep %d/%d (%.0fs)", cell_key, rep + 1, replicas,
                     time.time() - t0)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    df = pd.DataFrame(rows)
    p_long = OUT_DIR / f"synth_grid_{ts}.parquet"
    df.to_parquet(p_long, index=False)
    ok = df[df.get("skipped").isna()] if "skipped" in df else df
    summary = (ok.groupby(["cell", "method"])
               .agg(nmi_med=("nmi", "median"), nmi_mean=("nmi", "mean"),
                    ari_med=("ari", "median"),
                    per_snapshot_nmi_mean=("per_snapshot_nmi_mean", "mean"),
                    f1_mean=("f1_type_switch", "mean"),
                    replicas_ok=("nmi", "count"))
               .reset_index())
    p_sum = OUT_DIR / f"synth_grid_summary_{ts}.parquet"
    summary.to_parquet(p_sum, index=False)
    meta = dict(config=args.config, replicas=replicas, cells_mode=mode,
                grids=list(args.grids), cells=cells, total_s=round(time.time() - t0, 1),
                note="дрейф-метрики (per-snapshot NMI, F1 ±1 мес) — отчётный блок, "
                     "в композит метода НЕ входят (prereg §B); NMI_synth ноги "
                     "композита = медиана по сетке A")
    (OUT_DIR / f"synth_grid_meta_{ts}.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    log.info("готово: %s (%d строк), %s; %.0fs", p_long, len(df), p_sum,
             time.time() - t0)


if __name__ == "__main__":
    main()
