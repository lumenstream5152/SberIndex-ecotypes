"""06: сравнительный слой (PREREG §A меры + §B методы).

Часть 1 (меры): кандидаты, прошедшие гейт (table_A_gate.parquet; M11 fail —
в таблицу Б с пометкой, X1–X3 анти-примеры — не входят никогда) → ноги
Q/S/T/R/H → композит 0.30·z(Q)+0.25·z(S)+0.20·z(T)+0.15·z(R)+0.10·z(H)
(веса из configs/prereg.yaml) → margin rule (парный бутстреп 1000) →
Dirichlet(20·w)×2000 → LOMO×5 → table_B_measures.parquet / table_C_ranks.parquet.

Часть 2 (методы): labels всех методов последнего прогона 04 → ICVI-панель
(SW, CH/N, S_Dbw, MQ в композит; AVI/AVU — отчёт), стабильность (boot 90%×30 +
seed-IQR ×30), тайминг (замер полного fit в этом скрипте; run.log 04 хранит
только суммарное время), NMI_synth — из верификации сетки 06b (3 реплики
центральной ячейки, outputs/synth_grid/*cells_mode=central*; placeholder с
пометкой «верификация, ждём полную сетку» — итоговая нога = медиана по сетке A
из полного прогона 06b), интерпретируемость — из
configs/interpretability_rubric.yaml (пока пустая → обе версии композита).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from ecotypes import benchmark as bm
from ecotypes import cluster as cl
from ecotypes import icvi
from ecotypes import measures as ms
from ecotypes.config import load_config, load_prereg
from ecotypes.runctx import RunContext
from ecotypes.seeds import set_all_seeds, stage_seed
from ecotypes.synthetic import ari, make_ppdataset, nmi


def _find_run(outputs: Path, key: str, require: str | None = None) -> Path:
    """Последний outputs/<run>/metrics.json, содержащий ключ key (04/06a).
    require — обязательный файл рядом с metrics.json (напр. labels.parquet у 04):
    без него прогон 06 (чей metrics.json тоже несёт 'gamma_star') на повторном
    запуске находил бы сам себя вместо 04."""
    for d in sorted(outputs.iterdir(), reverse=True):
        m = d / "metrics.json"
        if not m.exists() or (require is not None and not (d / require).exists()):
            continue
        try:
            if key in json.loads(m.read_text(encoding="utf-8")):
                return d
        except Exception:
            continue
    raise FileNotFoundError(f"в {outputs} нет прогона с metrics.json[{key!r}]"
                            + (f" и файлом {require}" if require else ""))


def _latest_synth_verification(synth_dir: Path) -> tuple[Path, bool]:
    """Источник ноги NMI_synth: ПОСЛЕДНИЙ parquet сетки 06b. Возвращает
    (path, is_full). Полная сетка (cells_mode=all) → нога = медиана по сетке A
    (prereg §B), is_full=True. Иначе — верификация центральной ячейки
    (3 реплики), placeholder с пометкой «ждём полную сетку», is_full=False."""
    metas = sorted(synth_dir.glob("synth_grid_meta_*.json"), reverse=True)
    central_fallback = None
    for mp in metas:
        try:
            meta = json.loads(mp.read_text(encoding="utf-8"))
        except Exception:
            continue
        pq = mp.with_name(mp.name.replace("synth_grid_meta_", "synth_grid_")
                          .removesuffix(".json") + ".parquet")
        if not pq.exists():
            continue
        if meta.get("cells_mode") == "all":
            return pq, True
        if meta.get("cells_mode") == "central" and central_fallback is None:
            central_fallback = pq
    if central_fallback is not None:
        return central_fallback, False
    raise FileNotFoundError(f"в {synth_dir} нет прогонов сетки 06b")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", default=None)
    ap.add_argument("--out", default="data/processed")
    ap.add_argument("--skip-methods", action="store_true",
                    help="только часть 1 (меры)")
    ap.add_argument("--skip-measures", action="store_true",
                    help="только часть 2 (методы)")
    ap.add_argument("--synth-replicas", type=int, default=3,
                    help="реплик центральной ячейки для NMI_synth методов")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    cfg = load_config(args.config, overrides=args.overrides)
    set_all_seeds(cfg.seed)
    prereg = load_prereg()
    ps, pm = prereg["similarity"], prereg["method"]
    weights = {k: float(v) for k, v in ps["composite_weights"].items()}

    name = args.config.split("/")[-1].removesuffix(".yaml")
    if args.overrides:
        name += "_" + args.overrides.split("/")[-1].removesuffix(".yaml")
    ctx = RunContext(cfg, config_name=name, stage="06_icvi_compare")
    t0 = time.time()
    root = Path(args.out)
    outputs = Path("outputs")

    run04 = _find_run(outputs, "gamma_star", require="labels.parquet")
    m04 = json.loads((run04 / "metrics.json").read_text(encoding="utf-8"))
    gamma_star = float(m04["gamma_star"])
    ctx.log(f"γ* из {run04.name}: {gamma_star} (fallback={m04['fallback']}, "
            f"plateau={m04['plateau']})")

    nidx = pd.read_parquet(root / "graphs" / "node_index.parquet")
    nidx = nidx.sort_values("row_idx").reset_index(drop=True)
    nodes = nidx[["territory_id"]].merge(pd.read_parquet(root / "nodes_static.parquet"),
                                         on="territory_id")
    X, feat = cl.feature_matrix(nodes, cfg)
    seeds30 = list(range(int(ps["seeds"])))
    metrics: dict = {"gamma_star": gamma_star, "run04": run04.name,
                     "weights": weights}

    # ================= часть 1: композит мер =================
    if not args.skip_measures:
        table_a = pd.read_parquet(root / "measures" / "table_A_gate.parquet")
        candidates, excluded = bm.scoring_candidates(table_a, ps)
        ctx.log(f"кандидаты в скоринг: {candidates}; исключены: {excluded}")

        run06a = _find_run(outputs, "m9_alpha_chosen")
        m06a = json.loads((run06a / "metrics.json").read_text(encoding="utf-8"))
        alpha9 = float(m06a["m9_alpha_chosen"])
        ctx.log(f"M9 α*={alpha9} (из {run06a.name})")

        data = ms.from_processed(root)
        k = int(ps["knn_k"])
        fns = {
            "M1": ms.sim_M1, "M2": ms.sim_M2, "M3": ms.sim_M3, "M4": ms.sim_M4,
            "M5": ms.sim_M5, "M6": ms.sim_M6, "M7": ms.sim_M7, "M8": ms.sim_M8,
            "M9": (lambda d, lvl_idx=None, grw_idx=None:
                   ms.sim_M9(d, lvl_idx=lvl_idx, grw_idx=grw_idx, alpha=alpha9)),
            "M10": (lambda d, lvl_idx=None, grw_idx=None:
                    ms.sim_M10(d, lvl_idx=lvl_idx, grw_idx=grw_idx, k=k)),
        }

        # K-means протокол на X_static — общий для всех мер (prereg §C)
        km = bm.kmeans_protocol(X, list(cfg.cluster.kmeans.ks),
                                int(cfg.cluster.kmeans.n_init), seeds30,
                                ari_min=0.90)
        ctx.log(f"K-means протокол: K*={km['K']} (veto_failed={km['veto_failed']})\n"
                f"{km['table'].to_string(index=False)}")
        km_pairs = np.array([ari(km["runs"][i], km["runs"][j])
                             for i, j in combinations(range(len(km["runs"])), 2)])

        legs: dict[str, dict[str, float]] = {}
        q_sub: dict[str, dict[str, float]] = {}
        s_pairs: dict[str, np.ndarray] = {}
        r_inst: dict[str, np.ndarray] = {}
        extra: dict[str, dict] = {}
        halves = bm.oddeven_halves(data.T)

        def _checkpoint() -> None:
            """Страховка от внешнего kill (jetsam под memory pressure): ноги
            части 1 на диске после каждого R-инстанса; восстановление — ручное."""
            import pickle
            with open(ctx.dir / "_checkpoint_part1.pkl", "wb") as f:
                pickle.dump(dict(legs=legs, q_sub=q_sub, s_pairs=s_pairs,
                                 r_inst=r_inst, extra=extra, km=km), f)

        for mname in candidates:
            tm = time.time()
            A_m = bm.graph_from_npz(root / "measures" / f"{mname}.npz")
            ens = bm.leiden_ensemble(A_m, gamma_star, seeds30)

            # Q: Leiden ← атрибутивные ICVI на X_static; K-means ← сетевые на A_m
            cons = ens.consensus
            degenerate = len(np.unique(cons)) < 2
            if degenerate:  # вырожденный консенсус: ICVI не определены — NaN + флаг
                q_leiden = {"leiden_SW": np.nan, "leiden_CH_over_N": np.nan,
                            "leiden_neg_S_Dbw": np.nan}
            else:
                q_leiden = {
                    "leiden_SW": icvi.sw(X, cons),
                    "leiden_CH_over_N": icvi.ch_over_n(X, cons),
                    "leiden_neg_S_Dbw": -icvi.sdbw(X, cons),
                }
            q_sub[mname] = {
                **q_leiden,
                "kmeans_MQ": icvi.mq(A_m, km["labels"], variant="mancoridis"),
                "kmeans_internal_edge_ratio": bm.internal_edge_ratio(A_m, km["labels"]),
            }
            # S: средний попарный ARI 30 seeds, оба эстиматора. Попарный ARI
            # прогонов определён и при вырожденном консенсусе (нестабильность
            # seed-прогонов — легитимный сигнал, не подменяется).
            s_pairs[mname] = np.concatenate([ens.pair_ari, km_pairs])
            s_val = 0.5 * (float(ens.pair_ari.mean()) + float(km_pairs.mean()))
            # T: odd/even (отклонение №5); мера пересчитывается на нарезанной
            # под-панели (календарные месяцы позиций сохранены, см.
            # benchmark.sliced_measure_data) — без idx-аргументов
            t_labs = []
            for lvl_idx, _ in halves:
                res_h = fns[mname](bm.sliced_measure_data(data, lvl_idx))
                A_h = bm.measure_graph_from_sim(res_h, k)
                t_labs.append(bm.leiden_ensemble(A_h, gamma_star, seeds30).consensus)
            t_deg = any(len(np.unique(l)) < 2 for l in t_labs)
            t_val = np.nan if t_deg else ari(t_labs[0], t_labs[1])
            hmap = bm.hungarian_mapping(t_labs[0], t_labs[1])
            # H: 1 − Gini(степеней)
            deg = np.asarray(A_m.sum(axis=1)).ravel()
            h_val = 1.0 - bm.gini(deg)
            legs[mname] = {"Q": np.nan, "S": s_val, "T": t_val,
                           "R": np.nan, "H": h_val}  # Q,R ниже/дальше
            extra[mname] = {"t_hungarian": hmap, "k_leiden": int(len(np.unique(cons))),
                            "degenerate_consensus": bool(degenerate),
                            "leg_s_leiden": float(ens.pair_ari.mean()),
                            "leg_s_kmeans": float(km_pairs.mean())}
            ctx.log(f"{mname}: S={s_val:.4f} T={t_val:.4f} H={h_val:.4f} "
                    f"({time.time() - tm:.0f}s)")
        _checkpoint()

        # Q-среднее: z саб-индексов по кандидатам → среднее
        qdf = pd.DataFrame(q_sub).T
        q_mean = bm.leg_zscores(qdf).mean(axis=1)
        for mname in candidates:
            legs[mname]["Q"] = float(q_mean[mname])

        # R: NMI против истины на 10 синтетических инстансах (центральная ячейка).
        # Leiden на синтетике — γ из конфига (1.0), НЕ γ*: RB-γ не масштабно-
        # инвариантно, шкала весов синтетических kNN-графов иная, чем у реальных
        # (проверено 06.10: при γ*=0.293 corr-меры M1–M4 вырождаются в k=1 →
        # NMI≡0 по причине шкалы, не качества меры — нога мертва). Тот же фикс
        # a priori, что в 06b для методов («плато синтетики не выбирается»).
        gamma_synth = float(cfg.cluster.leiden.gamma)
        n_r = 10
        for i in range(n_r):
            ti = time.time()
            ds = make_ppdataset(seed=int(stage_seed(cfg.seed, f"benchR:{i}")),
                                K=6, delta=0.05, alpha_btw=80.0, mu_edge=0.25,
                                n=data.n, T=data.T)
            md = bm.synthetic_measure_data(ds)
            z_true = bm.modal_labels(ds["z"])
            for mname in candidates:
                res = fns[mname](md)
                A_s = bm.measure_graph_from_sim(res, k)
                cons_s = bm.leiden_ensemble(A_s, gamma_synth, seeds30).consensus
                r_inst.setdefault(mname, []).append(nmi(z_true, cons_s))
            ctx.log(f"R-инстанс {i + 1}/{n_r} ({time.time() - ti:.0f}s)")
            _checkpoint()
        for mname in candidates:
            arr = np.asarray(r_inst[mname], dtype=np.float64)
            r_inst[mname] = arr
            legs[mname]["R"] = float(arr.mean())

        legs_df = pd.DataFrame(legs).T.loc[candidates]
        legs_z = bm.leg_zscores(legs_df)
        scores = bm.composite(legs_z, weights)

        boot = bm.paired_bootstrap(legs_df, {"S": s_pairs, "R": r_inst},
                                   weights, B=1000, seed=stage_seed(cfg.seed, "bench_boot"))
        verdict = bm.margin_verdict(scores, boot.se_diff,
                                    float(ps["margin_rule_se"]),
                                    list(ps["tiebreak_order"]))
        sens = bm.dirichlet_sensitivity(
            legs_z, weights, float(ps["dirichlet_sensitivity"]["concentration"]),
            int(ps["dirichlet_sensitivity"]["draws"]), stage_seed(cfg.seed, "bench_dir"))
        lomo = bm.lomo_winners(legs_z, weights)
        if lomo["photo_finish"]:
            final = next(m for m in ps["tiebreak_order"]
                         if m in set(lomo["per_drop"].values()) | {lomo["full_winner"]})
            verdict = {**verdict, "verdict": verdict["verdict"] + "+photo_finish",
                       "winner": final}
            ctx.log(f"LOMO фотофиниш: {lomo['per_drop']} → выбор по tiebreak: {final}")

        table_b = legs_df.copy()
        table_b["verdict_gate"] = "pass"
        for c in legs_z.columns:
            table_b[f"z_{c}"] = legs_z[c]
        table_b["composite"] = scores
        # NaN-композит (NaN-нога, напр. M4: вырождение odd/even в k<2) →
        # rank=NA, кандидат выбывает из ранжирования явно, не молча
        table_b["rank"] = scores.rank(ascending=False).astype("Int64")
        table_b["ci_lo"] = boot.ci_lo
        table_b["ci_hi"] = boot.ci_hi
        for m, reason in excluded.items():
            if m in set(table_a.measure) - set(ps["anti_examples"]):
                table_b.loc[m] = np.nan
                table_b.loc[m, "verdict_gate"] = "fail"
        table_b = table_b.reset_index(names="measure")
        table_b.to_parquet(ctx.dir / "table_B_measures.parquet", index=False)
        bm.borda_table(legs_z, scores).reset_index(names="measure").to_parquet(
            ctx.dir / "table_C_ranks.parquet", index=False)

        metrics["measures"] = {
            "candidates": candidates, "excluded": excluded,
            "gamma_star": gamma_star, "gamma_synth_R": gamma_synth,
            "kmeans_protocol": {"K": km["K"], "veto_failed": km["veto_failed"],
                                "table": km["table"].to_dict("records")},
            "q_submetrics": qdf.to_dict(), "legs": legs_df.to_dict(),
            "scores": scores.to_dict(), "winner": verdict,
            "bootstrap": {"B": 1000,
                          "se_diff_vs_runnerup": boot.se_diff.to_dict(),
                          "p_beats_runnerup": boot.p_beats_runnerup.to_dict()},
            "dirichlet": sens.to_dict(), "lomo": lomo,
            "extra": extra,
            "notes": [
                "NaN-нога (M4: odd/even-половина вырождается в k<2 при γ*) → "
                "NaN-композит и rank=NA: кандидат выбывает из ранжирования явно; "
                "z-нормы ноги T считаются по кандидатам с конечным значением",
                "вырожденный консенсус (k<2 на графе меры при γ*) → атрибутивные "
                "ICVI NaN (Q из kmeans-сабиндексов), T NaN при вырождении половины; "
                "S берётся из попарного ARI прогонов (определён всегда); "
                "флаг degenerate_consensus в extra",
                "неотрицательность весов: clip(0) для знакопеременных шкал "
                "(как прод 03), сдвиг на минимум для целиком ≤0 шкал (M5–M7) — "
                "см. benchmark._shift_nonneg",
                "M8 статична: T=1 тривиально (как bootJ=1.0 в гейте) — свойство, не баг",
                "бутстреп парный: общие индексы ресэмпла по мерам; Q/T/H — точечные "
                "(без естественной репликации), дисперсию дают только S и R",
                "R: потолок статической NMI при δ=0.05 пересчитывается кодом "
                "(PREREG_DEVIATIONS №1), истина — модальные метки",
                "R: Leiden на синтетике — γ=1.0 из конфига (как 06b), НЕ γ*=0.293: "
                "RB-γ не масштабно-инвариантно, на синтетических весах γ* даёт "
                "k=1 у M1–M4 (проверено 06.10) — нога была бы мертва по причине "
                "шкалы, не качества; кандидат в PREREG_DEVIATIONS №9",
            ],
        }
        ctx.log(f"композит мер: победитель={verdict['winner']} "
                f"({verdict['verdict']}, margin={verdict['margin']:.4f}, "
                f"SE={verdict['se']:.4f}); LOMO changed={lomo['n_changed']}/5")

    # ================= часть 2: композит методов =================
    if not args.skip_methods:
        lab04 = pd.read_parquet(run04 / "labels.parquet")
        lab04 = lab04.sort_values("row_idx").reset_index(drop=True)
        method_cols = [c for c in lab04.columns if c not in ("row_idx", "territory_id")]
        # labels 04 несут суффикс K (kmeans_k6); run_method оперирует базовыми именами
        base_of = {m: re.sub(r"_k\d+$", "", m) for m in method_cols}
        A04 = sp.load_npz(root / "graphs" / "similarity_layer.npz").maximum(
            sp.load_npz(root / "graphs" / "geo_layer.npz"))
        # K методов — из labels 04 (leiden/louvain/infomap/eva — авто-k, K не используется)
        k_of = {m: int(lab04[m].nunique()) for m in method_cols}

        rows, icvi_report = {}, {}
        for mname in method_cols:
            tm = time.time()
            lab = lab04[mname].to_numpy()
            kk = len(np.unique(lab))
            panel = dict(SW=icvi.sw(X, lab), CH_over_N=icvi.ch_over_n(X, lab),
                         S_Dbw=icvi.sdbw(X, lab),
                         MQ=icvi.mq(A04, lab, variant="mancoridis"))
            icvi_report[mname] = {**panel, "AVI": icvi.avi(A04, lab),
                                  "AVU": icvi.avu(A04, lab)}
            stab = bm.method_stability(base_of[mname], A04, X, k_of[mname], cfg,
                                       n_boot=30, frac=0.9, n_seeds=30,
                                       seed=stage_seed(cfg.seed, f"stab:{mname}"),
                                       gamma=gamma_star)
            t_fit = time.time()
            bm.run_method(base_of[mname], A04, X, k_of[mname], cfg.seed, cfg,
                          gamma=gamma_star)
            fit_s = time.time() - t_fit
            rows[mname] = {**panel, **stab, "timing_s": fit_s, "k": kk}
            ctx.log(f"{mname}: ICVI+стабильность+тайминг ({time.time() - tm:.0f}s)")

        # NMI_synth: полная сетка 06b (медиана по сетке A, prereg §B), иначе —
        # верификация центральной ячейки (3 реплики, placeholder с пометкой).
        verif_pq, is_full = _latest_synth_verification(Path("outputs/synth_grid"))
        vdf = pd.read_parquet(verif_pq)
        if "skipped" in vdf:
            vdf = vdf[vdf["skipped"].isna()]
        if is_full:
            vdf = vdf[vdf["cell"].str.startswith("A:")]  # нога = сетка A (prereg §B)
        ctx.log(f"NMI_synth ← {verif_pq.name} ({len(vdf)} строк, "
                f"{'ПОЛНАЯ сетка A' if is_full else 'верификация центральной ячейки'})")
        nmi_rows: dict[str, list[float]] = {}
        nmi_cells: dict[str, int] = {}
        for mname in method_cols:
            vals = vdf.loc[vdf.method == base_of[mname]].sort_values(["cell", "replica"])
            nmi_rows[mname] = vals["nmi"].dropna().tolist()
            nmi_cells[mname] = int(vals["cell"].nunique())
        for mname in method_cols:
            # медиана по сетке (prereg §B: «медиана по сетке A»); у тяжёлых
            # методов — по доступным ячейкам (центральная, 5 реплик) с пометкой
            rows[mname]["nmi_synth"] = (float(np.median(nmi_rows[mname]))
                                        if nmi_rows[mname] else np.nan)
            rows[mname]["nmi_synth_cells"] = nmi_cells[mname]

        mdf = pd.DataFrame(rows).T
        # рубрика ключена базовыми именами методов (kmeans, не kmeans_k6)
        base_names = [base_of[m] for m in method_cols]
        interp, interp_status = bm.load_interpretability(
            "configs/interpretability_rubric.yaml", base_names)
        if interp is not None:
            interp.index = method_cols
        comp = bm.method_composite(mdf, pm, interp=interp)
        mdf_out = mdf.copy()
        mdf_out["composite_with_zero_interp"] = comp["composite_with_zero_interp"]
        if "composite_renormalized" in comp:
            mdf_out["composite_renormalized"] = comp["composite_renormalized"]
        mdf_out = mdf_out.reset_index(names="method")
        mdf_out.to_parquet(ctx.dir / "table_methods.parquet", index=False)
        # Wilcoxon signed-rank по общим репликам ЦЕНТРАЛЬНОЙ ячейки A (n=15,
        # prereg §B; BH-поправка по семейству пар внутри ячейки)
        central = vdf[vdf.cell == "A:K6:d0.05:a80.0:mu-"]
        wrows: dict[str, list[float]] = {}
        for mname in method_cols:
            vals = (central.loc[central.method == base_of[mname]]
                    .sort_values("replica")["nmi"].dropna().tolist())
            if len(vals) >= 2:
                wrows[mname] = vals
        wilc = bm.wilcoxon_table(pd.DataFrame(wrows)) if wrows else pd.DataFrame()

        nmi_note = (
            "NMI_synth — ПОЛНАЯ сетка 06b, медиана по сетке A (9 ячеек × 15 "
            "реплик, prereg §B); тяжёлые методы (eva) — только центральная "
            "ячейка A (5 реплик), помечено nmi_synth_cells" if is_full else
            "NMI_synth — центральная ячейка с 3 репликами (верификация), "
            "placeholder: ЖДЁМ ПОЛНУЮ СЕТКУ (06b, 15 реплик × все ячейки); "
            "итоговая нога — медиана по сетке A из 06b (prereg §B), "
            "значения и ранжирование методов ниже — НЕ финальные")
        metrics["methods"] = {
            "table": mdf.reset_index(names="method").to_dict("records"),
            "icvi_report_only_AVI_AVU": icvi_report,
            "interpretability": interp_status,
            "composite_with_zero_interp": comp["composite_with_zero_interp"].to_dict(),
            "composite_renormalized": comp.get(
                "composite_renormalized", pd.Series(dtype=float)).to_dict(),
            "winner_zero": str(comp["composite_with_zero_interp"].idxmax()),
            "winner_renormalized": (str(comp["composite_renormalized"].idxmax())
                                    if "composite_renormalized" in comp else None),
            "nmi_synth_source": {"file": verif_pq.name, "full_grid": is_full},
            "nmi_synth_replicas": nmi_rows,
            "wilcoxon_central_cell": wilc.to_dict("records"),
            "notes": [
                nmi_note,
                "Wilcoxon — по общим репликам центральной ячейки A "
                f"(n={max((len(v) for v in wrows.values()), default=0)}"
                f"{', дымовая проверка механики' if not is_full else ''})",
                "тайминг — замер полного fit в этом скрипте (run.log 04 хранит "
                "только суммарное время прогона)",
                ("интерпретируемость не заполнена → композит без ноги, обе версии "
                 "(с нулём и перенормированная) — молчаливой перенормировки нет"
                 if interp is None else
                 "интерпретируемость заполнена (оба оценщика) → композит полный"),
                "Leiden на синтетике — γ из конфига (1.0); плато синтетики "
                "не выбиралось (анти-тюнинг, фикс a priori)",
            ],
        }
        ctx.log(f"композит методов: winner(zero)={metrics['methods']['winner_zero']}, "
                f"winner(renorm)={metrics['methods']['winner_renormalized']}")

    metrics["total_s"] = round(time.time() - t0, 1)
    ctx.write_metrics(metrics)
    ctx.log(f"итого {time.time() - t0:.0f}s → {ctx.dir}")
    ctx.close()


if __name__ == "__main__":
    main()
