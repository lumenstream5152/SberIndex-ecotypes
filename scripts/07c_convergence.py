"""07c: клубы сходимости Phillips–Sul (research/30). Ряды: МП/Продовольствие, Продовольствие/Все,
Прочее/Все (food в панели = общепит!) → outputs/<run>/: clubs_*.parquet, summary, clubs_x_types."""
import argparse, json, time
from pathlib import Path

import pandas as pd
from sklearn.metrics import adjusted_rand_score
from ecotypes import logt as lt
from ecotypes.config import load_config
from ecotypes.runctx import RunContext
from ecotypes.seeds import set_all_seeds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--out", default="outputs", help="корень outputs/<run>/")
    args = ap.parse_args()
    cfg = load_config(args.config, overrides=args.overrides)
    set_all_seeds(cfg.seed)
    name = args.config.split("/")[-1].removesuffix(".yaml")
    name += "_" + args.overrides.split("/")[-1].removesuffix(".yaml") if args.overrides else ""
    ctx, t0 = RunContext(cfg, config_name=name, stage="07c_convergence", out_root=args.out), time.time()
    df = pd.read_parquet("data/processed/panel_monthly.parquet")
    series = {"mpfood": df.share_market / df.share_prod, "food": df.share_prod, "proch": df.share_proch}
    lp = Path("outputs/20261005_2129_default_fcd0587/labels.parquet")  # leiden_consensus из 04
    types = pd.read_parquet(lp, columns=["territory_id", "leiden_consensus"]) if lp.exists() else None
    if types is None: ctx.log(f"WARNING: {lp} нет — кросс-таб и ARI скипнуты")
    summary, xtabs = {"labels_source": str(lp) if types is not None else None}, []
    for sname, s in series.items():
        w = s.to_frame("v").join(df[["territory_id", "month"]]).pivot(
            index="territory_id", columns="month", values="v").sort_index(axis=1)
        Y = w.to_numpy(dtype="float64").T  # (T=24 мес, n=2016 МО); панель сбалансирована
        res, sig, bet = lt.club_classification(Y), lt.sigma_convergence(Y), lt.beta_convergence(Y)
        verdict = lt.verdict_2of3(lt.logt_test(Y), sig, bet)
        out = pd.DataFrame({"territory_id": w.index, "club": res["clubs"]})
        out.to_parquet(ctx.dir / f"clubs_{sname}.parquet", index=False)
        ari = None
        if types is not None:
            j = types.merge(out, on="territory_id")
            ari = float(adjusted_rand_score(j.leiden_consensus, j.club))
            x = pd.crosstab(j.club, j.leiden_consensus).stack().rename("n").reset_index()
            xtabs.append(x.assign(series=sname))
        summary[sname] = {"logt": lt.logt_test(Y), "sigma": {k: v for k, v in sig.items() if k != "cv_t"},
                          "beta": bet, **verdict, "n_clubs": res["n_clubs"], "club_sizes": res["club_sizes"],
                          "club_tstats": res["club_tstats"], "n_divergent": int(len(res["divergent"])),
                          "ari_types": ari, "merge_log": res["merge_log"]}
        ctx.log(f"{sname}: t={summary[sname]['logt']['t_stat']:.2f} клубов={res['n_clubs']} "
                f"диверг={len(res['divergent'])} {verdict['verdict']} ari={ari}")
    if xtabs: pd.concat(xtabs).to_csv(ctx.dir / "clubs_x_types.csv", index=False)
    (ctx.dir / "convergence_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    ctx.write_metrics({"stage": "07c_convergence", "n_units": int(w.shape[0]), "elapsed_s": time.time() - t0,
                       "verdicts": {s: summary[s]["verdict"] for s in series}})
    ctx.close()


if __name__ == "__main__":
    main()
