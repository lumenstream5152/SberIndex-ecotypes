"""05: динамика типов (research/29) → data/processed/dynamics/* + metrics.json."""
from __future__ import annotations
import argparse, logging, time
from pathlib import Path
import pandas as pd
from ecotypes import dynamics as dyn
from ecotypes.config import load_config
from ecotypes.runctx import RunContext
from ecotypes.seeds import set_all_seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--out", default="data/processed", help="графы из <out>/graphs/, продукты в <out>/dynamics/")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    cfg = load_config(args.config, overrides=args.overrides)
    set_all_seeds(cfg.seed)
    name = args.config.split("/")[-1].removesuffix(".yaml")
    if args.overrides:
        name += "_" + args.overrides.split("/")[-1].removesuffix(".yaml")
    ctx, t0 = RunContext(cfg, config_name=name, stage="05_dynamics"), time.time()
    root = Path(args.out)
    months = sorted(pd.read_parquet(root / "panel_monthly.parquet",
                                    columns=["month"]).month.unique())
    res = dyn.run_dynamics(root / "graphs", root / "edges_highway_knn.parquet",
                           root / "panel_monthly.parquet", months, cfg,
                           out_dir=root / "dynamics")
    m = res["metrics"]
    ctx.write_metrics(m)
    ctx.log(f"типов всего={m['k_total_types']}, переходы "
            f"{m['mover_share_raw']:.3f}→{m['mover_share_smoothed']:.3f}, "
            f"ARI cross mean={m['ari_cross_mean']:.3f} vs perturb "
            f"{m['ari_perturb_med_mean']:.3f}, flagged={m['n_flagged']}/{m['n_pairs']}, "
            f"события {m['n_events_raw']}raw→{m['n_events_smoothed']}smoothed"
            f"→{m['n_events_admitted']}admitted; "
            f"{time.time() - t0:.1f}s → {root / 'dynamics'}")
    ctx.close()


if __name__ == "__main__":
    main()
