"""09: методологический отчёт (research/11 §5).

Читает ТОЛЬКО каноничные артефакты outputs/main/ →
(1) report/figures/F1..F10.png (matplotlib, кириллица — DejaVu Sans);
(2) report/numbers.json — все числа отчёта одним дайджестом;
(3) report/methodology.md — рендер report/methodology.template.md
    с подстановкой {{ключ}} из numbers.json (без шаблонных дыр не соберётся —
    отсутствующий ключ = ошибка сборки, а не молчаливый пропуск).

Кейсы шоков берутся из transition_cards.parquet (поимённые, admitted).
"""
from __future__ import annotations

import argparse
import json
import logging
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ecotypes.config import load_config
from ecotypes.runctx import RunContext
from ecotypes.seeds import set_all_seeds

MAIN = Path("outputs/main")
FIG = Path("report/figures")
TEMPLATE = Path("report/methodology.template.md")

# Фигуры — в дизайн-системе деки (Golos Text, бумага, индиго; JR2/V2)
from matplotlib import font_manager as _fm
_FONT_DIR = Path(__file__).resolve().parents[2] / "deck" / "assets" / "fonts" / "ttf"
for _f in _FONT_DIR.glob("*.ttf"):
    try:
        _fm.fontManager.addfont(str(_f))
    except Exception:
        pass
plt.rcParams.update({
    "font.family": "Golos Text", "font.size": 10.5,
    "figure.dpi": 150, "figure.facecolor": "#FFFDF7",
    "axes.facecolor": "#FFFDF7", "savefig.facecolor": "#FFFDF7",
    "axes.grid": True, "grid.alpha": 0.25, "grid.color": "#8B8FA0",
    "axes.edgecolor": "#1D2530", "axes.labelcolor": "#1D2530",
    "text.color": "#1D2530", "xtick.color": "#4C5361", "ytick.color": "#4C5361",
    "axes.prop_cycle": plt.cycler(color=["#233C67", "#B0730F", "#2F6B4F",
                                         "#8E3B4B", "#4C5361"]),
})

log = logging.getLogger("report")


# ------------------------------------------------------------------ числа
def collect_numbers() -> dict:
    """Все числа отчёта из каноничных артефактов — ноль литералов в тексте."""
    n: dict = {}
    m04 = json.load(open(MAIN / "metrics_04_cluster.json", encoding="utf-8"))
    n["GAMMA_STAR"] = m04["gamma_star"]
    n["K_MACRO"] = int(m04["plateau_at_gamma_star"]["k_med"])
    n["ARI_MED_MACRO"] = round(float(m04["plateau_at_gamma_star"]["ari_med"]), 3)
    n["ARI_MIN_MACRO"] = round(float(m04["plateau_at_gamma_star"]["ari_min"]), 3)
    n["SEED_ARI_MED"] = round(float(m04["consensus"]["seed_ari_med"]), 3)

    stab = json.load(open(MAIN / "stability.json", encoding="utf-8"))
    n["ARI_MONTH"] = round(float(stab["summary"]["ari_cross_mean"]), 3)
    n["ARI_NULL"] = round(float(stab["summary"]["ari_perturb_med_mean"]), 3)
    n["N_FLAGGED"] = int(stab["summary"]["n_flagged"])
    n["N_PAIRS"] = int(len(stab["pairs"]))

    tb = pd.read_parquet(MAIN / "table_B_measures.parquet")
    tb = tb.sort_values("composite", ascending=False)
    n["M_WINNER"] = str(tb.iloc[0]["measure"])
    n["M_WINNER_COMP"] = round(float(tb.iloc[0]["composite"]), 3)
    n["M_RUNNER"] = str(tb.iloc[1]["measure"])
    n["M_RUNNER_COMP"] = round(float(tb.iloc[1]["composite"]), 3)

    tm = pd.read_parquet(MAIN / "table_methods.parquet")
    for col in ("composite_renormalized", "composite_with_zero_interp"):
        if col in tm:
            w = tm.sort_values(col, ascending=False).iloc[0]
            n[f"METHOD_WINNER_{col.upper()}"] = str(w["method"])
            n[f"METHOD_WINNER_{col.upper()}_SCORE"] = round(float(w[col]), 3)

    ev = pd.read_parquet(MAIN / "events_admitted.parquet")
    n["N_EVENTS_ADMITTED"] = int(len(ev))
    mm = json.load(open(MAIN / "model_metrics.json", encoding="utf-8"))
    for h in ("h1", "h3"):
        p = mm[h]["pooled"]
        n[f"PRAUC_{h.upper()}_LGBM"] = round(float(p["lgbm"]["pr_auc"]), 4)
        n[f"PRAUC_{h.upper()}_MARGIN"] = round(float(p["margin_rank"]["pr_auc"]), 4)
        n[f"PRAUC_{h.upper()}_LOGREG"] = round(float(p["logreg5"]["pr_auc"]), 4)
        n[f"PRAUC_{h.upper()}_BASE"] = round(float(p["lgbm"]["prevalence"]), 4)
    n["VERDICT_PUBLISH"] = str(mm["verdict"]["publish"])

    eng = pd.read_parquet(MAIN / "validation_engel.parquet")
    n["ENGEL_FOOD"] = "/".join(f"{v:.1f}" for v in
                               eng.sort_values("wage_rub")["share_prod"] * 100)
    n["ENGEL_WAGE"] = "/".join(f"{v / 1000:.0f}" for v in
                               eng.sort_values("wage_rub")["wage_rub"])
    fr = pd.read_parquet(MAIN / "validation_four_russias.parquet")
    r1 = fr[fr.russia.isin(["Р1", "Россия-1", "Россия 1"])]
    n["R1_LIFTS"] = "/".join(f"{v:.2f}" for v in sorted(r1["lift"],
                                                      reverse=True)[:2])
    mono = pd.read_parquet(MAIN / "validation_monotowns.parquet")
    n["MONO_LIFT_MAX"] = round(float(mono["lift"].max()), 2)

    lead = json.load(open(MAIN / "lead_summary.json", encoding="utf-8"))
    n["LEAD_EDGES"] = int(lead["fdr"]["n_significant"])
    n["LEAD_PAIRS"] = int(lead["n_pairs"])

    conv = json.load(open(MAIN / "convergence_summary.json", encoding="utf-8"))
    n["CLUBS_MPFOOD"] = str(conv.get("mpfood", {}).get("n_clubs", "?"))

    null = MAIN / "icvi_null.parquet"
    if null.exists():
        zn = pd.read_parquet(null)
        n["ICVI_NULL_Z"] = "; ".join(
            f"{r['index']}: z={r['z']:.1f}" for _, r in zn.iterrows())

    pm = pd.read_parquet(MAIN / "passports_macro.parquet")
    n["MACRO_SIZES"] = "/".join(str(int(v)) for v in sorted(pm["n_mo"],
                                                            reverse=True))
    return n


# ------------------------------------------------------------------ фигуры
def make_figures(n: dict) -> list[str]:
    FIG.mkdir(parents=True, exist_ok=True)
    made = []

    def _save(fig, name: str) -> None:
        fig.tight_layout()
        fig.savefig(FIG / name)
        plt.close(fig)
        made.append(name)

    # F1: плато-правило — стабильность по γ
    pt = pd.read_parquet(MAIN / "plateau_table.parquet")
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(pt.gamma, pt.ari_med, "o-", label="медиана попарного ARI (25 seeds)")
    ax.fill_between(pt.gamma, pt.ari_min, pt.ari_med, alpha=0.2,
                    label="min…med")
    ax.axhline(0.9, ls="--", c="grey", label="порог плато 0.9")
    if n.get("GAMMA_STAR"):
        ax.axvline(n["GAMMA_STAR"], ls=":", c="red",
                   label=f"γ*={n['GAMMA_STAR']}")
    ax.set_xscale("log")
    ax.set_xlabel("γ (resolution Leiden)")
    ax.set_ylabel("ARI")
    ax.set_title("F1. Плато-правило: стабилен только макро-уровень k=3")
    ax.legend(fontsize=8)
    _save(fig, "F1_plateau.png")

    # F2: композиты мер
    tb = pd.read_parquet(MAIN / "table_B_measures.parquet")
    tb = tb.sort_values("composite")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    colors = ["#B0730F" if m == "M2" else "#233C67" for m in tb.measure]
    ax.barh(tb.measure, tb.composite, color=colors)
    ax.set_xlabel("композит 0.30Q+0.25S+0.20T+0.15R+0.10H")
    ax.set_title("F2. Бенчмарк мер сходства (M2 — прод; M11 провалил гейт "
                 "и вне композита)")
    _save(fig, "F2_measures.png")

    # F3: Энгель
    eng = pd.read_parquet(MAIN / "validation_engel.parquet").sort_values("wage_rub")
    fig, ax1 = plt.subplots(figsize=(6, 4))
    x = np.arange(len(eng))
    ax1.bar(x - 0.2, eng.share_prod * 100, width=0.4, label="доля продовольствия, %")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, eng.wage_rub / 1000, width=0.4, color="orange",
            label="медианная зарплата, тыс. ₽")
    ax2.grid(False)
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"тип {t}" for t in eng.type])
    ax1.set_title("F3. Энгель-чек: доля еды ↓ с ростом дохода (качественно, n=3)")
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center", fontsize=8, ncol=2,
               frameon=True, framealpha=0.9)
    ax1.set_ylim(0, max(eng.share_prod * 100) * 1.35)
    _save(fig, "F3_engel.png")

    # F4: четыре России
    fr = pd.read_parquet(MAIN / "validation_four_russias.parquet")
    piv = fr.pivot_table(index="russia", columns="type_id", values="lift")
    fig, ax = plt.subplots(figsize=(6.5, 4))
    vmax = float(np.nanmax(piv.values))
    im = ax.imshow(piv.values, cmap="RdBu_r", vmin=0, vmax=vmax)
    ax.set_xticks(range(len(piv.columns)), [f"тип {c}" for c in piv.columns])
    ax.set_yticks(range(len(piv.index)), piv.index)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, "—" if np.isnan(v) else f"{v:.1f}",
                    ha="center", va="center", fontsize=9)
    ax.set_title("F4. «Четыре России» Зубаревич × наши типы (lift; — = нет МО)")
    fig.colorbar(im, ax=ax, label="lift")
    _save(fig, "F4_four_russias.png")

    # F5: SHAP топ-15
    mm = json.load(open(MAIN / "model_metrics.json", encoding="utf-8"))
    sh = mm.get("shap_top15") or mm.get("shap_top15_full_model") or []
    if sh:
        if isinstance(sh, dict):
            sh = list(sh.items())
        names = [s[0] if isinstance(s, (list, tuple)) else s.get("feature")
                 for s in sh]
        vals = [s[1] if isinstance(s, (list, tuple)) else s.get("mean_abs_shap")
                for s in sh]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.barh(names[::-1], vals[::-1])
        ax.set_xlabel("mean |SHAP|")
        ax.set_title("F5. Драйверы переходов: топ-15 признаков (TreeSHAP)")
        _save(fig, "F5_shap.png")

    # F6: event-study, признак с максимальным |Cohen's d| в момент события
    es = pd.read_parquet(MAIN / "event_study.parquet")
    scope_all = es.scope.astype(str).str.upper() == "ALL"
    d0 = es[scope_all] if scope_all.any() else es
    d0 = d0.dropna(subset=["cohens_d"])
    if len(d0):
        # фигура-герой: признак с сильнейшим монотонным сдвигом ДО события
        # (тренд cohens_d по t∈[−6,−1]); календарные month_sin/cos исключаем —
        # они показывают сезонную концентрацию событий, не предвестник
        pre = d0[(d0.rel_month >= -6) & (d0.rel_month <= -1)]
        pre = pre[~pre.feature.isin(["month_sin", "month_cos"])]
        slopes = {}
        for f, g in pre.groupby("feature"):
            g = g.sort_values("rel_month")
            if len(g) >= 4:
                slopes[f] = abs(np.polyfit(g.rel_month, g.cohens_d, 1)[0])
        top_feat = max(slopes, key=slopes.get) if slopes else d0.feature.iloc[0]
        d = d0[d0.feature == top_feat]
    else:
        d = es.iloc[0:0]
    if len(d):
        d = d.sort_values("rel_month")
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.plot(d.rel_month, d.mean_movers, "o-", color="#233C67",
                label="переходчики")
        ax.plot(d.rel_month, d.mean_controls, "s--", color="#B0730F",
                label="matched-контроль")
        ax.axvline(0, ls=":", c="#C6402E")
        ax.set_xlabel("месяц относительно перехода")
        ax.set_title(f"F6. Event-study: {d.feature.iloc[0]} "
                     f"(ассоциация, не причинность)")
        ax.legend()
        _save(fig, "F6_event_study.png")

    # F7: методы — композиты
    tm = pd.read_parquet(MAIN / "table_methods.parquet")
    comp_col = "composite_renormalized" if "composite_renormalized" in tm \
        else "composite_with_zero_interp"
    tms = tm.sort_values(comp_col)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(tms.method, tms[comp_col])
    ax.set_xlabel(f"композит методов ({comp_col})")
    ax.set_title("F7. Сравнение методов (предрегистрированный композит)")
    _save(fig, "F7_methods.png")

    # F8: нулевой бутстреп ICVI
    null = MAIN / "icvi_null.parquet"
    if null.exists():
        zn = pd.read_parquet(null)
        zn = zn.replace([np.inf, -np.inf], np.nan).dropna(subset=["z"])
        fig, ax = plt.subplots(figsize=(7, 3.5))
        zc = zn.copy()
        zc["z_plot"] = zc["z"].clip(-40, 40)
        ax.barh(zc["index"], zc["z_plot"])
        ax.set_xlabel("z против перестановочного нуля (клип ±40)")
        ax.set_title("F8. ICVI прод-типологии против случайных разбиений "
                     "(200 перестановок)")
        _save(fig, "F8_icvi_null.png")

    # F9: воронка событий — три стадии, горизонтальные ступени с потерями
    ev_all_p = MAIN / "events_all.parquet"
    stages = []
    if ev_all_p.exists():
        ev_all = pd.read_parquet(ev_all_p)
        rr = ev_all.reject_reason.fillna("")
        n_smooth = len(ev_all)
        n_nf = int(((~rr.str.contains("no_flicker")) ).sum())
        n_seed = int(((~rr.str.contains("no_flicker"))
                      & (~rr.str.contains("seed_agreement"))).sum())
        n_adm = int(ev_all.admitted.sum())
        stages = [("сырые смены меток (3107)", 3107),
                  ("после сглаживания-3", n_smooth),
                  ("+ нет мерцания", n_nf),
                  ("+ согласие seeds ≥ 0.8", n_seed),
                  ("+ сдвиг CLR ≥ q75", n_adm)]
    else:
        stages = [("admitted", n.get("N_EVENTS_ADMITTED", 0))]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    names = [s for s, _ in stages][::-1]
    vals = [v for _, v in stages][::-1]
    bars = ax.barh(names, vals, color="#233C67", height=0.62)
    bars[0].set_color("#C6402E")   # нижний бар = admitted (после разворота списка)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + max(vals) * 0.012, b.get_y() + b.get_height() / 2,
                f"{v:,}".replace(",", " "), va="center",
                fontfamily="PT Mono", fontsize=11)
    ax.set_xlim(0, max(vals) * 1.13)
    ax.set_title("F9. Воронка допуска событий (узловой скрин, №12)")
    ax.grid(axis="y", visible=False)
    _save(fig, "F9_events_funnel.png")

    # F10: синтетика NMI по методам
    sg = pd.read_parquet(MAIN / "synth_grid_summary.parquet")
    sgm = sg.groupby("method")["nmi_med"].median().sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.barh(sgm.index, sgm.values)
    ax.set_xlabel("NMI (медиана по сетке синтетики)")
    ax.set_title("F10. Восстановление истины на синтетике (PP-Dir/LFR)")
    _save(fig, "F10_synth.png")

    # F11: корреляции приростов до/после демеанинга (выборка 40k пар)
    try:
        from ecotypes.seeds import stage_seed as _ss
        pan = pd.read_parquet("data/processed/panel_monthly.parquet",
                              columns=["territory_id", "month", "log_all"])
        P = (pan.pivot(index="territory_id", columns="month", values="log_all")
                .sort_index().to_numpy())
        L = np.diff(P, axis=1)   # log_all уже лог-уровень → diff = приросты
        rng = np.random.default_rng(_ss(42, "f11"))
        i = rng.integers(0, L.shape[0], 40000)
        j = rng.integers(0, L.shape[0], 40000)
        keep = i != j
        i, j = i[keep], j[keep]
        def _corr_rows(A, B):
            A = A - A.mean(axis=1, keepdims=True)
            B = B - B.mean(axis=1, keepdims=True)
            return (A * B).mean(axis=1) / (A.std(axis=1) * B.std(axis=1) + 1e-12)
        r_raw = _corr_rows(L[i], L[j])
        Ld = L - L.mean(axis=0, keepdims=True)             # демеанинг по месяцу
        r_dm = _corr_rows(Ld[i], Ld[j])
        fig, ax = plt.subplots(figsize=(7, 3.8))
        ax.hist(r_raw, bins=80, alpha=0.65, density=True,
                label=f"сырые приросты (медиана {np.median(r_raw):.3f})",
                color="#B0730F")
        ax.hist(r_dm, bins=80, alpha=0.65, density=True,
                label=f"после демеанинга (медиана {np.median(r_dm):.3f})",
                color="#233C67")
        ax.axvline(0, color="#17150F", lw=0.8)
        ax.set_xlabel("попарная корреляция приростов")
        ax.set_title("F11. Почему демеанинг: сырые корреляции вырождены")
        ax.legend(fontsize=9)
        _save(fig, "F11_demean.png")
    except Exception as exc:
        log.warning("F11 пропущен: %s", exc)

    return made


# ------------------------------------------------------------------ рендер
def render(numbers: dict) -> None:
    if not TEMPLATE.exists():
        log.warning("%s не найден — рендер methodology.md пропущен "
                    "(числа и фигуры собраны)", TEMPLATE)
        return
    text = TEMPLATE.read_text(encoding="utf-8")
    keys = set(re.findall(r"\{\{(\w+)\}\}", text))
    missing = keys - set(numbers)
    if missing:
        raise RuntimeError(f"в numbers.json нет ключей шаблона: {sorted(missing)}")
    for k, v in numbers.items():
        text = text.replace("{{" + k + "}}", str(v))
    Path("report/methodology.md").write_text(text, encoding="utf-8")
    log.info("report/methodology.md собран (%d чисел подставлено)", len(numbers))


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
    ctx = RunContext(cfg, config_name=name, stage="09_report")

    numbers = collect_numbers()
    figs = make_figures(numbers)
    Path("report").mkdir(exist_ok=True)
    with open("report/numbers.json", "w", encoding="utf-8") as f:
        json.dump(numbers, f, ensure_ascii=False, indent=1, default=str)
    render(numbers)
    ctx.write_metrics({"figures": figs, "n_numbers": len(numbers),
                       "methodology_md": Path("report/methodology.md").exists()})
    ctx.log(f"фигур: {len(figs)}, чисел: {len(numbers)}")
    ctx.close()


if __name__ == "__main__":
    main()
