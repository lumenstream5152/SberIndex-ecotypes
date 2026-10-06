"""07d: лаг-лидерство (research/22) → <out>/laglead/{lead_graph.parquet,
lead_summary.json} + копии и metrics.json (с синтетической верификацией)
в outputs/<run_id>/."""
from __future__ import annotations

import argparse
import shutil
import time
from pathlib import Path

from ecotypes import laglead as ll
from ecotypes.config import load_config
from ecotypes.runctx import RunContext
from ecotypes.seeds import set_all_seeds, stage_seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--out", default="data/processed",
                    help="каталог панели; продукты пишутся в <out>/laglead/")
    ap.add_argument("--labels",
                    default="outputs/20261005_2129_default_fcd0587/labels.parquet",
                    help="labels.parquet с leiden_consensus для согласия с типами")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    set_all_seeds(cfg.seed)
    name = args.config.split("/")[-1].removesuffix(".yaml")
    if args.overrides:
        name += "_" + args.overrides.split("/")[-1].removesuffix(".yaml")
    ctx = RunContext(cfg, config_name=name, stage="07d_laglead")

    t0 = time.time()
    seed = stage_seed(cfg.seed, "07d_laglead")
    edges, summary = ll.run_laglead(args.out, labels_path=args.labels,
                                    seed=seed, log=ctx.log)
    synth = ll.synthetic_verification(seed=seed ^ 0xBEEF)
    ctx.log(f"синтетика: recovery={synth['planted']['exact_recovery']:.2f}, "
            f"null FDR rate={synth['null']['fdr_discovery_rate']:.3f}")

    dest = Path(args.out) / "laglead"
    for f in ("lead_graph.parquet", "lead_summary.json"):
        shutil.copy2(dest / f, ctx.dir / f)
    ctx.write_metrics({"stage": "07d_laglead", "seed": seed,
                       "summary": summary, "synthetic_verification": synth,
                       "total_s": round(time.time() - t0, 1)})
    eb = summary["edges_by_threshold"]
    ctx.log(f"рёбер FDR: {summary['fdr']['n_significant']:,}; "
            f"r>0.80: {eb['r>0.80_core']:,} (спека ≈1.4k), "
            f"r>0.70: {eb['r>0.70_layer']:,} (спека ≈11k); "
            f"{time.time() - t0:.0f}s")
    ctx.close()


if __name__ == "__main__":
    main()
