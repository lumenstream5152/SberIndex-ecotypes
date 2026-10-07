"""07e: вспомогательные публикационные артефакты (закрывает находки JR1).

(1) type_id_map.parquet — карта между двумя пространствами меток: макро-слой
    (leiden_consensus k=3, из последнего run 04) × помесячный слой (реестр
    динамики, smoothed). Двусторонняя: доля макро-типа в каждом типе реестра
    и доля типа реестра в каждом макро-типе.
(2) events_sensitivity.csv — чувствительность числа admitted-событий к порогу
    displacement (q50…q95) при неизменных остальных условиях скрина (№12):
    показывает, что выводы не висят на одном квантиле.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from ecotypes.config import load_config
from ecotypes.interpret import find_run
from ecotypes.runctx import RunContext
from ecotypes.seeds import set_all_seeds

Q_GRID = [0.50, 0.60, 0.70, 0.75, 0.80, 0.90, 0.95]


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
    ctx = RunContext(cfg, config_name=name, stage="07e_publish_aux")
    root = Path(args.out)

    # (1) карта макро × реестр
    run04 = find_run(Path("outputs"), "gamma_star", require="labels.parquet")
    lab04 = pd.read_parquet(run04 / "labels.parquet").sort_values("row_idx")
    macro = lab04.set_index("territory_id")["leiden_consensus"]
    labels = pd.read_parquet(root / "dynamics" / "labels.parquet")
    labels = labels.merge(macro.rename("macro"), on="territory_id")
    rows = []
    for (mo, reg), g in labels.groupby(["macro", "type_id_smooth"]):
        rows.append(dict(macro_type=int(mo), registry_type=int(reg),
                         n_node_months=int(len(g)),
                         n_nodes=int(g.territory_id.nunique())))
    m = pd.DataFrame(rows)
    tot_reg = m.groupby("registry_type")["n_nodes"].sum()
    tot_mac = m.groupby("macro_type")["n_nodes"].sum()
    m["share_within_registry_type"] = m.apply(
        lambda r: r.n_nodes / tot_reg[r.registry_type], axis=1)
    m["share_within_macro_type"] = m.apply(
        lambda r: r.n_nodes / tot_mac[r.macro_type], axis=1)
    m = m.sort_values(["registry_type", "macro_type"]).reset_index(drop=True)
    m.to_parquet(ctx.dir / "type_id_map.parquet", index=False)
    ctx.log(f"карта: {m.registry_type.nunique()} типов реестра × "
            f"{m.macro_type.nunique()} макро, {len(m)} пар")

    # (2) sensitivity порога displacement: порог считается, как в скрине (№12),
    # по ВСЕМ узел-месяцам (CLR-дистанции смежных месяцев), не по событиям
    from ecotypes.dynamics import CLR_COLS
    ev = pd.read_parquet(root / "dynamics" / "events_all.parquet")
    panel = pd.read_parquet(root / "panel_monthly.parquet")
    labels = pd.read_parquet(root / "dynamics" / "labels.parquet")
    months = sorted(labels["month"].astype(str).unique())
    tids = np.sort(labels["territory_id"].unique())
    X = np.stack([panel.assign(month=panel["month"].astype(str))
                       .pivot(index="territory_id", columns="month", values=c)
                       .reindex(index=tids, columns=months).to_numpy(np.float64)
                  for c in CLR_COLS])
    D = np.sqrt((np.diff(X, axis=2) ** 2).sum(axis=0))      # (n, T−1), все узел-месяцы
    # база: события, прошедшие no_flicker и seed_agreement (admitted или
    # отклонённые только из-за displacement)
    base = ev[(ev.admitted) | (ev.reject_reason.fillna("") == "displacement")]
    rows = []
    for q in Q_GRID:
        thr = float(np.nanquantile(D, q))
        rows.append(dict(q=q, threshold=round(thr, 4),
                         n_admitted=int((base.displacement >= thr).sum())))
    s = pd.DataFrame(rows)
    s.to_csv(ctx.dir / "events_sensitivity.csv", index=False)
    ctx.log(f"sensitivity: {dict(zip(s.q, s.n_admitted))}")

    ctx.write_metrics({"type_id_map_pairs": int(len(m)),
                       "sensitivity": {str(r.q): int(r.n_admitted)
                                       for r in s.itertuples()}})
    ctx.close()


if __name__ == "__main__":
    main()
