"""03: построить графы (research/23) из data/processed/ → data/processed/graphs/."""
from __future__ import annotations

import argparse
import time

from ecotypes.config import load_config
from ecotypes.graphs import build_graphs
from ecotypes.runctx import RunContext
from ecotypes.seeds import set_all_seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--out", default="data/processed",
                    help="каталог панели; графы пишутся в <out>/graphs/ (smoke → data/processed_smoke)")
    args = ap.parse_args()

    cfg = load_config(args.config, overrides=args.overrides)
    set_all_seeds(cfg.seed)
    name = args.config.split("/")[-1].removesuffix(".yaml")
    if args.overrides:
        name += "_" + args.overrides.split("/")[-1].removesuffix(".yaml")
    ctx = RunContext(cfg, config_name=name, stage="03_build_graphs")
    t0 = time.time()
    build_graphs(cfg, processed_dir=args.out, out_dir=f"{args.out}/graphs", ctx=ctx)
    ctx.log(f"build_graphs: {time.time() - t0:.1f}s → {args.out}/graphs/")
    ctx.close()


if __name__ == "__main__":
    main()
