"""07b: драйверы переходов и раннее предупреждение (research/31).
LightGBM y1/y3 на smoothed-метках, rolling-origin с эмбарго, PR-AUC + lift +
isotonic, базлайны margin-rank/logreg5, TreeSHAP + карточки + event-study,
радар 2024-12 (unverified). Продукты → outputs/<run_id>/."""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ecotypes.config import load_config
from ecotypes.drivers import run_all
from ecotypes.runctx import RunContext
from ecotypes.seeds import set_all_seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--out", default="data/processed",
                    help="корень processed-данных (входы); продукты — в outputs/<run_id>/")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    cfg = load_config(args.config, overrides=args.overrides)
    set_all_seeds(cfg.seed)

    name = args.config.split("/")[-1].removesuffix(".yaml")
    if args.overrides:
        name += "_" + args.overrides.split("/")[-1].removesuffix(".yaml")
    ctx = RunContext(cfg, config_name=name, stage="07b_drivers")
    t0 = time.time()

    m = run_all(cfg, root=Path(args.out), ctx=ctx)
    slim = {"h1_pooled": m["h1"]["pooled"], "h3_pooled_pr_auc": m["h3"]["pooled"]["lgbm"]["pr_auc"],
            "verdict": m["verdict"], "radar_sanity": m["radar_sanity"],
            "prevalence_y1": m["prevalence_y1"], "prevalence_y3": m["prevalence_y3"]}
    ctx.write_metrics(slim)
    v = m["verdict"]
    p1 = m["h1"]["pooled"]
    ctx.log(f"вердикт: публикуем {v['publish']}; PR-AUC h1 lgbm "
            f"{p1['lgbm']['pr_auc']:.3f} vs margin-rank {p1['margin_rank']['pr_auc']:.3f} "
            f"vs logreg5 {p1['logreg5']['pr_auc']:.3f} (base {p1['lgbm']['prevalence']:.3f}); "
            f"итого {time.time() - t0:.0f}s")
    ctx.close()


if __name__ == "__main__":
    main()
