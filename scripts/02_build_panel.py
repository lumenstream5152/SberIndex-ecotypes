"""02: собрать панель и атрибуты узлов (research/21 §3) в data/processed/."""
from __future__ import annotations

import argparse
import time

import pandas as pd

from ecotypes.config import load_config
from ecotypes.panel import build_panel
from ecotypes.runctx import RunContext
from ecotypes.seeds import set_all_seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--out", default="data/processed",
                    help="выходной каталог (smoke → data/processed_smoke)")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    set_all_seeds(cfg.seed)
    ctx = RunContext(cfg, config_name=args.config.split("/")[-1].removesuffix(".yaml"),
                     stage="02_build_panel")
    t0 = time.time()
    paths = build_panel(cfg, out_dir=args.out)
    ctx.log(f"build_panel: {time.time() - t0:.1f}s → {args.out}/")

    panel = pd.read_parquet(paths["panel_monthly"])
    nodes = pd.read_parquet(paths["nodes_static"])
    edges = pd.read_parquet(paths["edges_highway_knn"])
    excl = pd.read_parquet(paths["excluded_incomplete"])
    ctx.write_metrics({
        "panel_rows": len(panel),
        "nodes": len(nodes),
        "node_months_expected": len(nodes) * 24,
        "edges": len(edges),
        "excluded_incomplete": len(excl),
        "excluded_mech": excl.mech.value_counts().to_dict(),
        "ma_missing": int(nodes.ma_missing.sum()),
        "geo_island": int(nodes.geo_island.sum()),
        "rosstat_missing": int(nodes.rosstat_missing.sum()),
        "regions": int(nodes.region_code.nunique()),
        "sample_n": cfg.data.sample_n,
        "seconds": round(time.time() - t0, 1),
    })
    ctx.close()


if __name__ == "__main__":
    main()
