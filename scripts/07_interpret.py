"""07: движок описаний типов (research/33). Макро k=3 (leiden_consensus 04)
+ подтипы (type_id_smooth, снимок последнего месяца, пометка стабильности):
Миркин / суррогатное дерево / TreeSHAP → согласие → bootstrap → паспорта →
черновики имён (configs/type_names_draft.yaml, approved=false) → внешняя
валидация (моногорода 1398-р, «Четыре России», Энгель, курорты, Азнакаево).
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

from ecotypes.config import load_config
from ecotypes.interpret import run_all
from ecotypes.runctx import RunContext
from ecotypes.seeds import set_all_seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--boot", type=int, default=100,
                    help="bootstrap-реплик Миркина/дерева (33 §7)")
    ap.add_argument("--shap-replicas", type=int, default=30,
                    help="bootstrap-реплик SHAP (дорогой, 33 §7)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    cfg = load_config(args.config, overrides=args.overrides)
    set_all_seeds(cfg.seed)

    name = args.config.split("/")[-1].removesuffix(".yaml")
    if args.overrides:
        name += "_" + args.overrides.split("/")[-1].removesuffix(".yaml")
    ctx = RunContext(cfg, config_name=name, stage="07_interpret")
    t0 = time.time()

    res = run_all(cfg, out_root=Path(args.out), ctx=ctx,
                  shap_boot_replicas=args.shap_replicas, boot_B=args.boot)
    for layer, r in res["results"].items():
        agr = r["agreement"]["summary"]
        ctx.log(f"[{layer}] согласие описателей по типам: "
                + ", ".join(f"K{int(row.type_id)}={row.agreement_score:.2f}"
                            for row in agr.itertuples()))
    eng = res["validation"]["engel"]
    ctx.log(f"Энгель-чек: spearman(зарплата, доля продовольствия) = "
            f"{eng['spearman_wage_food']:.2f}, монотонно обратная = "
            f"{eng['monotone_inverse']}")
    ctx.log(f"Азнакаевский кейс: {res['validation']['aznakay']['line']}")
    ctx.log(f"итого {time.time() - t0:.0f}s → {ctx.dir}")
    ctx.close()


if __name__ == "__main__":
    main()
