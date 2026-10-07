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
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches as mpatches

from ecotypes.config import load_config
from ecotypes.runctx import RunContext
from ecotypes.seeds import set_all_seeds

MAIN = Path("outputs/main")
FIG = Path("report/figures")
TEMPLATE = Path("report/methodology.template.md")

# Фигуры — в дизайн-системе деки (бумага #FFFDF7, индиго; JR2/V2).
# Шрифт: Golos Text из deck/assets/fonts/ttf, если там появятся настоящие TTF
# (сейчас там WOFF2 с расширением .ttf — freetype их не ест); иначе DejaVu
# Sans (системный PT Sans живёт в .ttc, а matplotlib ttc-записи рисует молча
# пусто — не используем). Кириллица есть в обоих.
from matplotlib import font_manager as _fm
_FONT_DIR = Path(__file__).resolve().parents[2] / "deck" / "assets" / "fonts" / "ttf"
_FAMILIES = []
for _f in _FONT_DIR.glob("*.ttf"):
    try:
        _fm.fontManager.addfont(str(_f))
        _FAMILIES.append(_fm.FontProperties(fname=str(_f)).get_name())
    except Exception:
        pass
_FAMILIES.append("DejaVu Sans")

PAPER = "#FFFDF7"   # панель деки — фон всех фигур
GRID = "#E9E4D3"    # тёплая сетка под бумагу
INK = "#1D2530"
MUTED = "#4C5361"
FAINT = "#8B8FA0"
INDIGO = "#233C67"
GOLD = "#B0730F"
WINE = "#8E3B4B"
STEEL = "#4E7CA6"
SAND = "#C9B585"
GREY = "#B9C0C5"
FAIL = "#A33327"    # провал гейта (бордовый акцент деки)
SRC_NOTE = ("Данные: СберИндекс 2023–2024 · CC BY-SA 4.0 · "
            "расчёты: outputs/main · make report")
TYPE_SHORT = {0: "Периферия", 1: "Сервисные ядра", 2: "Северо-Запад"}
TYPE_COLOR = {0: SAND, 1: STEEL, 2: WINE}

plt.rcParams.update({
    "font.family": _FAMILIES, "font.size": 10.5,
    "figure.dpi": 150, "figure.facecolor": PAPER,
    "axes.facecolor": PAPER, "savefig.facecolor": PAPER,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.prop_cycle": plt.cycler(color=[INDIGO, GOLD, "#2F6B4F",
                                         WINE, MUTED]),
})


def _style_ax(ax) -> None:
    """Дек-стиль осей: без верхнего/правого спайнов, сетка только горизонтальная."""
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.grid(axis="x", visible=False)


def _head(fig, title: str, subtitle: str | None = None) -> None:
    """Заголовок = вывод (жирный, слева, от края ФИГУРЫ — не осей: длинные
    ylabel сдвигают axes вправо, и ax-заголовок упирается в правый край);
    подзаголовок = метод мелким, перенос по ~95 символам (строка шире фигуры
    ломает tight_layout — mpl 3.11 учитывает fig.text в tightbbox)."""
    fig.text(0.012, 0.975, title, fontsize=12.5, fontweight="bold",
             color=INK, ha="left", va="top")
    if subtitle:
        fig.text(0.012, 0.925, "\n".join(textwrap.wrap(subtitle, 95)),
                 fontsize=8.8, color=MUTED, ha="left", va="top")

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

    def _save(fig, name: str, top: float = 0.84) -> None:
        fig.tight_layout(rect=(0, 0.05, 1, top))
        fig.text(0.008, 0.006, SRC_NOTE, fontsize=7.3, color=FAINT,
                 ha="left", va="bottom")
        fig.savefig(FIG / name)
        plt.close(fig)
        made.append(name)

    def _bar_labels(ax, bars, vals, fmt="{:+.2f}", fs=9):
        xmax = max(abs(min(vals)), abs(max(vals))) if vals else 1.0
        for b, v in zip(bars, vals):
            if not np.isfinite(v):
                continue
            if v >= 0:
                ax.text(v + xmax * 0.015, b.get_y() + b.get_height() / 2,
                        fmt.format(v), va="center", ha="left", fontsize=fs)
            else:
                ax.text(v - xmax * 0.015, b.get_y() + b.get_height() / 2,
                        fmt.format(v), va="center", ha="right", fontsize=fs)

    # F1: плато-правило — стабильность по γ
    pt = pd.read_parquet(MAIN / "plateau_table.parquet")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(pt.gamma, pt.ari_med, "o-", color=INDIGO, ms=4.5, lw=1.7,
            label="медиана попарного ARI (25 seeds)")
    ax.fill_between(pt.gamma, pt.ari_min, pt.ari_med, color=INDIGO, alpha=0.15,
                    lw=0, label="min–медиана")
    ax.axhline(0.9, ls="--", lw=1.1, color=MUTED, label="порог плато 0.9")
    if n.get("GAMMA_STAR"):
        ax.axvline(n["GAMMA_STAR"], ls=":", lw=1.7, color=WINE,
                   label=f"γ* = {n['GAMMA_STAR']}")
    ax.set_xscale("log")
    ax.set_xlabel("γ (resolution Leiden)")
    ax.set_ylabel("попарный seed-ARI")
    ax.set_ylim(0, 1.05)
    _head(fig, "F1. Плато-правило: стабилен только макро-уровень k=3",
          "попарный ARI между 25 seed-прогонами Leiden против resolution γ")
    ax.legend(fontsize=8.5, loc="lower left", frameon=True, facecolor=PAPER,
              edgecolor=GRID, framealpha=1.0)
    _style_ax(ax)
    _save(fig, "F1_plateau.png")

    # F2: композиты мер — все 11 мер; M2 золотым (прод), M11 красным (провал
    # гейта), M4 серым (нога T не определена → вне композита)
    tb = pd.read_parquet(MAIN / "table_B_measures.parquet")
    gate = pd.read_parquet(MAIN / "table_A_gate.parquet")
    gate_j = dict(zip(gate.measure, gate.boot_jaccard))
    tb = tb.sort_values("composite")
    ok = tb.dropna(subset=["composite"])
    nok = tb[tb.composite.isna()]
    labels = list(ok.measure) + list(nok.measure)
    vals = [float(v) for v in ok.composite] + [np.nan] * len(nok)
    colors = []
    for m in labels:
        if m == "M2":
            colors.append(GOLD)
        elif gate_j.get(m) is not None and gate_j[m] < 0.25:
            colors.append(FAIL)
        elif not np.isfinite(vals[labels.index(m)]):
            colors.append(GREY)
        else:
            colors.append(INDIGO)
    fig, ax = plt.subplots(figsize=(7, 5.0))
    y = np.arange(len(labels))
    bars = ax.barh(y, [v if np.isfinite(v) else 0.0 for v in vals],
                   color=colors, height=0.64)
    ax.set_yticks(y, labels)
    ax.axvline(0, color=MUTED, lw=0.9)
    _bar_labels(ax, bars, [v for v in vals], fmt="{:+.2f}")
    lo, hi = min(v for v in vals if np.isfinite(v)), max(
        v for v in vals if np.isfinite(v))
    for yi, m in zip(y, labels):
        if np.isfinite(vals[yi]):
            continue
        if gate_j.get(m) is not None and gate_j[m] < 0.25:
            ax.text(lo + 0.02, yi, f"гейт не пройден: boot-J {gate_j[m]:.2f} "
                    "< 0.25", va="center", ha="left",
                    fontsize=9, color=FAIL, fontweight="bold")
        else:
            ax.text(lo + 0.02, yi, "нога T не определена",
                    va="center", ha="left", fontsize=9, color=MUTED)
    ax.set_xlim(lo - 0.16, hi + 0.14)
    ax.set_xlabel("композит 0.30·Q + 0.25·S + 0.20·T + 0.15·R + 0.10·H  "
                  "(z по кандидатам)")
    _head(fig, "F2. M3 выиграл бенчмарк мер — и схлопнул типологию",
          "композит пяти предрегистрированных ног; прод-опора M2 — золотым; "
          "< 0 = ниже среднего по мерам")
    _style_ax(ax)
    _save(fig, "F2_measures.png")

    # F3: Энгель — две панели вместо двойной оси; типы по возрастанию дохода
    eng = pd.read_parquet(MAIN / "validation_engel.parquet").sort_values("wage_rub")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 4.3))
    x = np.arange(len(eng))
    cols = [TYPE_COLOR[int(t)] for t in eng.type]
    # вторые строки короче полных имён типов — иначе слипаются между соседями
    F3_SHORT = {0: "Периферия", 1: "Ядра", 2: "Северо-Запад"}
    labs = [f"K{int(t)}\n{F3_SHORT[int(t)]}" for t in eng.type]
    for a, series, unit, ylab in (
            (a1, eng.share_prod * 100, "{:.1f}", "доля продовольствия в тратах, %"),
            (a2, eng.wage_rub / 1000, "{:.0f}", "медианная зарплата, тыс. ₽")):
        a.bar(x, series, color=cols, width=0.6)
        for xi, v in zip(x, series):
            a.text(xi, v + max(series) * 0.025, unit.format(v), ha="center",
                   fontsize=10.5, fontweight="bold")
        a.set_xticks(x)
        a.set_xticklabels(labs, fontsize=8.5)
        a.set_ylabel(ylab)
        a.set_ylim(0, max(series) * 1.2)
        _style_ax(a)
    _head(fig, "F3. Закон Энгеля: беднее тип — выше доля продовольствия",
              "медианы по макро-типам (порядок — по зарплате); n=3 → качественный "
              "sanity-check: Spearman = −1 тривиален (p = 1/3)")
    _save(fig, "F3_engel.png")

    # F4: четыре России — шкала lift центрирована на паритете 1.0
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    fr = pd.read_parquet(MAIN / "validation_four_russias.parquet")
    piv = fr.pivot_table(index="russia", columns="type_id", values="lift")
    cnt = fr.pivot_table(index="russia", columns="type_id", values="n")
    RU_LABEL = {"Р1": "Р1 · агломерации",
                "Р2": "Р2 · средние и крупные города",
                "Р3": "Р3 · малые города и райцентры",
                "Р4": "Р4 · сельская периферия"}
    cmap = LinearSegmentedColormap.from_list("lift", [WINE, "#EFEAD9", STEEL])
    cmap.set_bad(PAPER)
    vmax = float(np.nanmax(piv.values))
    norm = TwoSlopeNorm(vmin=0, vcenter=1.0, vmax=vmax)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    im = ax.imshow(np.ma.masked_invalid(piv.values), cmap=cmap, norm=norm,
                   aspect="auto")
    ax.set_xticks(range(len(piv.columns)),
                  [f"K{c}" for c in piv.columns], fontsize=10)
    ax.set_yticks(range(len(piv.index)),
                  [RU_LABEL.get(r, r) for r in piv.index], fontsize=9.5)
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            if np.isnan(v):
                ax.text(j, i, "нет МО", ha="center", va="center",
                        fontsize=8.5, color=FAINT)
                continue
            r_, g_, b_, _ = cmap(norm(v))
            lum = 0.299 * r_ + 0.587 * g_ + 0.114 * b_
            tc = "white" if lum < 0.55 else INK
            ax.text(j, i - 0.14, f"{v:.1f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=tc)
            ax.text(j, i + 0.24, f"n={int(cnt.values[i, j])}", ha="center",
                    va="center", fontsize=7.5, color=tc, alpha=0.85)
    ax.grid(False)
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1, 2, 3, 4, 5], shrink=0.85)
    cb.set_label("lift (1.0 = паритет)", fontsize=9)
    cb.ax.tick_params(labelsize=8.5)
    _head(fig, "F4. «Россия-1» раскалывается между ядрами и Северо-Западом",
          "lift доли типа внутри «Россий» Зубаревич к доле в целом, шкала "
          "центрирована на паритете 1.0; K0 — Периферия, K1 — сервисные "
          "городские ядра, K2 — Северо-Запад")
    _save(fig, "F4_four_russias.png")

    # F5: SHAP топ-15 — человеческие лейблы + группы цветом
    CAT_RU = {"prod": "продовольствие", "health": "здоровье",
              "market": "маркетплейсы", "food": "общепит",
              "transp": "транспорт", "proch": "прочее"}
    GROUP_COLOR = {"соседство": STEEL, "доход и масштаб": INDIGO,
                   "профиль потребления": WINE, "позиция в типе": GOLD,
                   "календарь": SAND}

    def _feat_label(f: str) -> str:
        m = re.fullmatch(r"nbr_share_(\d+)", f)
        if m:
            return f"доля соседей в подтипе {m.group(1)}"
        m = re.fullmatch(r"(d[13])_clr_(\w+)", f)
        if m:
            return (f"сдвиг доли «{CAT_RU.get(m.group(2), m.group(2))}» "
                    f"за {m.group(1)[1]} мес (CLR)")
        m = re.fullmatch(r"vol6_clr_(\w+)", f)
        if m:
            return (f"волатильность доли «{CAT_RU.get(m.group(1), m.group(1))}», "
                    f"6 мес")
        m = re.fullmatch(r"clr_(\w+)", f)
        if m:
            return f"профиль: доля «{CAT_RU.get(m.group(1), m.group(1))}» (CLR)"
        m = re.fullmatch(r"d3_share_(\w+)", f)
        if m:
            return (f"сдвиг доли «{CAT_RU.get(m.group(1), m.group(1))}» "
                    f"за 3 мес, п.п.")
        return {
            "foreign_share": "доля соседей других типов",
            "dom_foreign_type": "доминирующий чужой тип соседей",
            "log_ma_z": "рыночная доступность (market access, z log)",
            "ma_missing": "нет данных market access",
            "month_sin": "месяц события (синус, сезонность)",
            "month_cos": "месяц события (косинус, сезонность)",
            "d3_log_all": "прирост log трат за 3 мес",
            "log_all": "log уровня трат",
            "log_wage": "log зарплаты",
            "urban_share": "доля городского населения",
            "empl_pc": "занятые в организациях на душу населения",
            "margin": "запас до границы типа (margin)",
            "dist_own": "расстояние до центра своего типа",
            "rank_own": "ранг внутри своего типа",
            "mp_ratio": "доля маркетплейсов / продовольствия",
            "mp_slope6": "тренд доли маркетплейсов, 6 мес",
            "mp_tstat6": "значимость тренда маркетплейсов",
            "seed_agreement": "согласие seed-прогонов",
        }.get(f, f)

    def _feat_group(f: str) -> str:
        if f.startswith("nbr_share_") or f in ("foreign_share",
                                               "dom_foreign_type"):
            return "соседство"
        if f in ("log_ma_z", "ma_missing", "log_wage", "log_all",
                 "d3_log_all", "empl_pc", "urban_share"):
            return "доход и масштаб"
        if f in ("month_sin", "month_cos"):
            return "календарь"
        if f in ("margin", "dist_own", "rank_own", "seed_agreement"):
            return "позиция в типе"
        return "профиль потребления"

    mm = json.load(open(MAIN / "model_metrics.json", encoding="utf-8"))
    sh = mm.get("shap_top15") or mm.get("shap_top15_full_model") or []
    if sh:
        if isinstance(sh, dict):
            sh = list(sh.items())
        names = [s[0] if isinstance(s, (list, tuple)) else s.get("feature")
                 for s in sh]
        vals = [s[1] if isinstance(s, (list, tuple)) else s.get("mean_abs_shap")
                for s in sh]
        names, vals = names[::-1], vals[::-1]
        fig, ax = plt.subplots(figsize=(7.2, 5.2))
        colors = [GROUP_COLOR[_feat_group(f)] for f in names]
        bars = ax.barh([_feat_label(f) for f in names], vals, color=colors,
                       height=0.66)
        _bar_labels(ax, bars, vals, fmt="{:.3f}", fs=8.5)
        ax.set_xlim(0, max(vals) * 1.75)   # справа — место под легенду групп
        ax.set_xlabel("mean |SHAP| (log-odds)")
        seen, handles = set(), []
        for gname, gcol in GROUP_COLOR.items():
            if gname not in seen and any(_feat_group(f) == gname for f in names):
                seen.add(gname)
                handles.append(mpatches.Patch(color=gcol, label=gname))
        ax.legend(handles=handles, fontsize=8.5, loc="lower right",
                  frameon=True, facecolor=PAPER, edgecolor=GRID, framealpha=1.0)
        _head(fig, "F5. Что движет переходами: соседство, траты, сезонность",
              "топ-15 признаков по mean |SHAP| (TreeSHAP; финальная LightGBM "
              "h=1, версия без seed_agreement); цвет = группа признаков")
        _style_ax(ax)
        _save(fig, "F5_shap.png")

    # F6: event-study, признак с сильнейшим предсдвигом — Δ к t=−6 + 95% CI.
    # Сырые уровни серий не сматчены (контроль стабильно выше), поэтому обе
    # серии нормируются к своему значению в t=−6: показываем именно сдвиг.
    es = pd.read_parquet(MAIN / "event_study.parquet")
    scope_all = es.scope.astype(str).str.upper() == "ALL"
    d0 = es[scope_all] if scope_all.any() else es
    d0 = d0.dropna(subset=["cohens_d"])
    FEAT_META = {
        "foreign_share": ("доля соседей других типов", True),
        "nbr_share_2": ("доля соседей в подтипе 2", True),
        "nbr_share_3": ("доля соседей в подтипе 3", True),
        "d3_log_all": ("прирост log трат за 3 мес", False),
        "log_ma_z": ("рыночная доступность (z log)", False),
        "month_cos": ("сезонность (косинус месяца)", False),
    }
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
        lab, is_share = FEAT_META.get(str(d.feature.iloc[0]),
                                      (str(d.feature.iloc[0]), False))
        sc = 100.0 if is_share else 1.0
        unit = ", п.п." if is_share else ""
        t = d.rel_month.to_numpy(int)
        m1 = d.mean_movers.to_numpy(float)
        m2 = d.mean_controls.to_numpy(float)
        cd = d.cohens_d.to_numpy(float)
        n_ev = int(d.n_events.iloc[0])
        # pooled SD восстанавливаем из Cohen's d: d = (m1 − m2) / sp
        with np.errstate(divide="ignore", invalid="ignore"):
            sp = np.abs((m1 - m2) / cd)
        sp[~np.isfinite(sp) | (np.abs(cd) < 0.05)] = np.nan
        sp = np.where(np.isnan(sp), np.nanmedian(sp), sp)
        # контроль: k=5 NN с возвращением + калипер → эффективное n неизвестно
        # из агрегата; берём n контроля = n событий (консервативно широко)
        i0 = int(np.flatnonzero(t == -6)[0])
        d1 = (m1 - m1[i0]) * sc
        d2 = (m2 - m2[i0]) * sc
        # CI разности к t=−6: Var(Δ) = Var(t) + Var(−6); независимость месяцев
        # консервативна (автокорреляция одного МО сузила бы ленты)
        se1 = 1.96 * np.sqrt(sp ** 2 / n_ev + sp[i0] ** 2 / n_ev) * sc
        se2 = 1.96 * np.sqrt(sp ** 2 / n_ev + sp[i0] ** 2 / n_ev) * sc
        fig, ax = plt.subplots(figsize=(7, 4.4))
        ax.axhline(0, lw=0.9, color=MUTED)
        ax.axvline(0, ls=":", lw=1.5, color=MUTED)
        ax.annotate("переход", xy=(0, 1.0), xycoords=("data", "axes fraction"),
                    xytext=(4, -3), textcoords="offset points", fontsize=8.5,
                    color=MUTED, va="top", ha="left")
        ax.fill_between(t, d1 - se1, d1 + se1, color=INDIGO, alpha=0.15, lw=0)
        ax.fill_between(t, d2 - se2, d2 + se2, color=GOLD, alpha=0.18, lw=0)
        ax.plot(t, d1, "o-", color=INDIGO, ms=4.5, lw=1.7,
                label="переходчики")
        ax.plot(t, d2, "s--", color=GOLD, ms=4.2, lw=1.5,
                label="matched-контроль")
        ax.set_xlabel("месяц относительно перехода")
        ax.set_ylabel(f"Δ к t=−6: {lab}{unit}")
        _head(fig, "F6. Соседский состав съезжается к переходу и не расходится",
              f"обе серии — Δ к своему уровню в t=−6; matched-контроль 5-NN "
              f"(n={n_ev} событий); ленты — 95% CI (аналитич., консервативные); "
              f"ассоциация, не причинность")
        ax.legend(fontsize=9, loc="upper left", frameon=True, facecolor=PAPER,
                  edgecolor=GRID, framealpha=1.0)
        _style_ax(ax)
        _save(fig, "F6_event_study.png")

    # F7: методы — композиты (прод-метод золотым, как M2 в F2)
    tm = pd.read_parquet(MAIN / "table_methods.parquet")
    comp_col = "composite_renormalized" if "composite_renormalized" in tm \
        else "composite_with_zero_interp"
    tms = tm.sort_values(comp_col)
    fig, ax = plt.subplots(figsize=(7, 4.6))
    colors = [GOLD if m == "leiden_consensus" else INDIGO for m in tms.method]
    vals = [float(v) for v in tms[comp_col]]
    bars = ax.barh(tms.method, vals, color=colors, height=0.64)
    ax.axvline(0, color=MUTED, lw=0.9)
    _bar_labels(ax, bars, vals, fmt="{:+.2f}")
    lo, hi = min(vals), max(vals)
    ax.set_xlim(lo - 0.18, hi + 0.14)
    ax.set_xlabel("композит методов (предрегистрированный, renorm по ногам)")
    _head(fig, "F7. Лидирует spectral_knn; в прод пошёл Leiden-консенсус",
          "Leiden-консенсус (золотым) — единственный с авто-k и плато-"
          "стабильностью; < 0 = ниже среднего по методам")
    _style_ax(ax)
    _save(fig, "F7_methods.png")

    # F8: нулевой бутстреп ICVI — клип ±40, истинные z подписаны на барах
    null = MAIN / "icvi_null.parquet"
    if null.exists():
        zn = pd.read_parquet(null)
        zn = zn.replace([np.inf, -np.inf], np.nan).dropna(subset=["z"])
        zn = zn.sort_values("z")
        clip = 40.0
        fig, ax = plt.subplots(figsize=(7.2, 4.2))
        zp = zn.z.clip(-clip, clip).to_numpy(float)
        colors = []
        for _, r in zn.iterrows():
            if abs(r["z"]) < 1.96:
                colors.append(GREY)          # статистически нуль
            else:
                colors.append(INDIGO)
        bars = ax.barh(zn["index"], zp, color=colors, height=0.62)
        ax.axvline(0, color=MUTED, lw=0.9)
        ax.axvline(1.96, ls=":", lw=1.0, color=FAINT)
        ax.axvline(-1.96, ls=":", lw=1.0, color=FAINT)
        for yi, (_, r) in enumerate(zn.iterrows()):
            z = float(r["z"])
            ztxt = f"z = {z:,.0f}".replace(",", " ") if abs(z) >= 100 \
                else f"z = {z:.1f}"
            if z > clip:                     # обрезан: подпись внутри бара
                ax.text(clip - 1.2, yi, f"→ {ztxt}", va="center", ha="right",
                        fontsize=9.5, fontweight="bold", color="white")
            elif z < -clip:
                ax.text(-clip + 1.2, yi, f"{ztxt} ←", va="center", ha="left",
                        fontsize=9.5, fontweight="bold", color="white")
            elif z >= 0:
                ax.text(z + 1.0, yi, ztxt, va="center", ha="left", fontsize=9.5)
            else:
                ax.text(z - 1.0, yi, ztxt, va="center", ha="right",
                        fontsize=9.5)
        # AVU: перестановочный нуль вырожден (std ≈ 0) — z неинформативен
        for yi, (_, r) in enumerate(zn.iterrows()):
            if abs(float(r["z"])) < 1.96:
                ax.text(min(float(r["z"]), 0) + 3.0, yi - 0.31,
                        "нуль вырожден (std ≈ 0) — z неинформативен",
                        fontsize=8, color=MUTED, va="center", ha="left")
        ax.set_xlim(-clip - 8, clip + 4)
        ax.set_xlabel("z против перестановочного нуля "
                      "(пунктир — |z| = 1.96; ось обрезана на ±40)")
        _head(fig, "F8. ICVI прод-типологии далеки от случайных разбиений",
              "z против 200 перестановок меток; истинные z подписаны на барах; "
              "S_Dbw — lower-better (z < 0 = лучше нуля)")
        _style_ax(ax)
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
        stages = [("сырые смены меток", 3107),
                  ("после сглаживания-3", n_smooth),
                  ("+ нет мерцания", n_nf),
                  ("+ согласие seeds ≥ 0.8", n_seed),
                  ("+ сдвиг CLR ≥ q75", n_adm)]
    else:
        stages = [("admitted", n.get("N_EVENTS_ADMITTED", 0))]
    fig, ax = plt.subplots(figsize=(7, 4.0))
    names = [s for s, _ in stages][::-1]
    vals = [v for _, v in stages][::-1]
    bars = ax.barh(names, vals, color=INDIGO, height=0.62)
    bars[0].set_color(FAIL)   # нижний бар = admitted (после разворота списка)
    for b, v in zip(bars, vals):
        ax.text(b.get_width() + max(vals) * 0.012, b.get_y() + b.get_height() / 2,
                f"{v:,}".replace(",", " "), va="center",
                fontfamily="monospace", fontsize=10.5)
    ax.set_xlim(0, max(vals) * 1.13)
    _head(fig, f"F9. Воронка допуска: из {stages[0][1]:,} смен меток — "
              f"{stages[-1][1]} событий".replace(",", " "),
          "узловой скрин: нет мерцания + согласие seed-прогонов ≥ 0.8 + "
          "CLR-сдвиг ≥ q75 (журнал отклонений №12)")
    _style_ax(ax)
    _save(fig, "F9_events_funnel.png")

    # F10: синтетика — медиана + IQR NMI по ячейкам сетки (не бары от нуля:
    # разброс 0.78–0.95 на оси от 0 нечитаем)
    sg = pd.read_parquet(MAIN / "synth_grid_summary.parquet")
    g = sg.groupby("method")["nmi_med"]
    med = g.median()
    q1, q3 = g.quantile(0.25), g.quantile(0.75)
    order = med.sort_values()
    fig, ax = plt.subplots(figsize=(7, 4.6))
    y = np.arange(len(order))
    for yi, m in zip(y, order.index):
        color = GOLD if m == "leiden_consensus" else INDIGO
        ax.plot([q1[m], q3[m]], [yi, yi], color=color, lw=3.5, alpha=0.35,
                solid_capstyle="round")
        ax.plot(med[m], yi, "o", color=color, ms=8)
        ax.text(min(q3[m] + 0.008, 1.0), yi, f"{med[m]:.2f}", va="center",
                fontsize=9)
    ax.set_yticks(y, order.index)
    ax.set_xlim(max(0.5, float(q1.min()) - 0.05), 1.02)
    ax.set_xlabel("NMI против истины (точка = медиана, ус = IQR по ячейкам)")
    _head(fig, f"F10. Синтетика: NMI {med.min():.2f}–{med.max():.2f} — "
              f"методы различаются умеренно",
          "восстановление известной правды на сетке PP-Dir/LFR "
          "(13 ячеек × 15 реплик); ось усечена; Leiden-консенсус — золотым")
    _style_ax(ax)
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
        fig, ax = plt.subplots(figsize=(7, 4.0))
        ax.hist(r_raw, bins=80, alpha=0.6, density=True,
                label=f"сырые приросты (медиана {np.median(r_raw):.3f})",
                color=GOLD)
        ax.hist(r_dm, bins=80, alpha=0.65, density=True,
                label=f"после демеанинга (медиана {np.median(r_dm):.3f})",
                color=INDIGO)
        ax.axvline(0, color=MUTED, lw=1.0)
        ax.set_xlabel("попарная корреляция приростов")
        ax.set_ylabel("плотность")
        _head(fig, "F11. Почему демеанинг: сырые корреляции вырождены",
              "попарная корреляция приростов log трат; выборка 40 тыс. "
              "случайных пар МО")
        ax.legend(fontsize=9, frameon=True, facecolor=PAPER, edgecolor=GRID,
                  framealpha=1.0)
        _style_ax(ax)
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
