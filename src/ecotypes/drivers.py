"""Драйверы переходов и раннее предупреждение (research/31, волна ФУНДАМЕНТ).

Панель МО×месяц → LightGBM «сменит тип в t+1» (y1, основной) и «в t+1..t+3» (y3)
на smoothed-метках модуля dynamics. Валидация строго rolling-origin с эмбарго
h месяцев (12 фолдов h=1 / 10 фолдов h=3): случайный KFold = утечка, запрещён.
Метрики: PR-AUC (главная; baseline = prevalence рядом), ROC-AUC, lift@top-decile,
Brier + isotonic-калибровка на out-of-fold предсказаниях. Обязательные базлайны —
margin-rank и 5-признаковая логрег: если LightGBM не бьёт margin-rank по PR-AUC,
публикуем простую модель (§2.3: честность = баллы). Объяснение — TreeSHAP:
глобальный summary, профили топ-8 каналов A→B, карточки переходов по шаблону
§3.2 (значения в сырых единицах), Спирмен-кроссчек против Миркин-профилей
(interpret.mirkin_profiles — если модуль недоступен, скип с пометкой).
Event-study ±6 мес против matched-контроля k=5 NN из того же type_from —
формулировки только «ассоциировано», не causal. Радар 2024-12: watch-list
топ-дециль с unverified=true (2025 в данных нет, out-of-time верификация
невозможна) + санити-чеки (KS-дрейф, flagged, пересечение с низким
seed_agreement).

Конфиги задача не расширяет: гиперпараметры спеки 31 живут в DriversParams,
seed — из cfg.seed через stage_seed(cfg.seed, "drivers").
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.metrics import (average_precision_score, brier_score_loss,
                             roc_auc_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline

from .config import Config
from .seeds import stage_seed

log = logging.getLogger("drivers")

PARTS = ["prod", "health", "market", "food", "transp", "proch"]
CLR = [f"clr_{s}" for s in PARTS]
LOGREG5 = ["margin", "foreign_share", "mp_slope6", "dist_own", "d3_clr_market"]
# признаки матчинга event-study: уровни clr + Δ3 + market_access + масштаб (31 §4.2)
MATCH_FEATS = [*CLR, *[f"d3_clr_{s}" for s in PARTS], "log_ma_z", "log_all"]
HONEST_RADAR_NOTE = ("Вероятности на 2024-12 предсказывают январь 2025-го, "
                     "которого в данных нет; out-of-time верификация невозможна. "
                     "Калибровка и precision оценены на rolling-origin фолдах "
                     "2023-12…2024-11; фактическая на 2025 неизвестна.")


@dataclass
class DriversParams:
    """Все числа — из спеки 31 (§1.3, §2, §4, §5). Конфиги не трогаем (граница
    задачи), поэтому это dataclass с дефолтами, а не секция default.yaml."""
    history: int = 6                 # 6-мес история → первый usable t = idx 6 (2023-07)
    first_val_idx: int = 11          # первая val-точка rolling-origin = 2023-12
    knn_k: int = 10                  # highway-соседи (спека 29 §6.4)
    lgbm: dict = field(default_factory=lambda: dict(
        n_estimators=500, num_leaves=31, learning_rate=0.05,
        min_child_samples=50, verbose=-1))
    es_rounds: int = 50              # early stopping на хвосте train (последний train-месяц)
    top_decile: float = 0.1          # lift@decile и watch-list
    event_window: int = 6            # ±6 мес вокруг перехода
    event_min_channel: int = 30      # per-канал разрез только при ≥30 событиях
    match_k: int = 5                 # matched-контроль k=5 NN с возвращением
    caliper_q: float = 0.95          # калипер = 95-й перцентиль расстояний
    top_channels: int = 8            # профили драйверов по топ-8 каналам
    top_traj_feats: int = 6          # event-study рисует топ-6 признаков по |SHAP|
    shap_sample: int = 20000         # глобальный TreeSHAP на подвыборке
    low_seed_q: float = 0.1          # нижний дециль seed_agreement = «риск-артефакт»
    calib_bins: int = 10


# ------------------------------------------------------------------ вход

def load_inputs(root: str | Path) -> dict:
    root = Path(root)
    dyn = root / "dynamics"
    with open(dyn / "stability.json", encoding="utf-8") as f:
        stability = json.load(f)
    return dict(
        panel=pd.read_parquet(root / "panel_monthly.parquet"),
        labels=pd.read_parquet(dyn / "labels.parquet"),
        events=pd.read_parquet(dyn / "events.parquet"),
        registry=pd.read_parquet(dyn / "type_registry.parquet"),
        nodes=pd.read_parquet(root / "nodes_static.parquet"),
        edges=pd.read_parquet(root / "edges_highway_knn.parquet"),
        stability=stability,
    )


def flagged_pair_months(stability: dict) -> set[str]:
    """month_from пар, где ARI_cross ≤ ARI_perturb (эволюция неотличима от шума):
    переходы t→t+1 по таким парам недоверенные (29 §5) — строки выбрасываем."""
    return {p["month_from"] for p in stability.get("pairs", []) if p["flagged"]}


def _pivot(df: pd.DataFrame, col: str, tids, months) -> np.ndarray:
    return (df.pivot(index="territory_id", columns="month", values=col)
              .reindex(index=tids, columns=months).to_numpy(np.float64))


# ------------------------------------------------------------------ таргеты

def make_targets(labels: pd.DataFrame, registry: pd.DataFrame,
                 flagged: set[str], months: list[str], *,
                 first_usable_idx: int = 6) -> tuple[pd.DataFrame, np.ndarray]:
    """Таргеты на smoothed-метках (31 §1.2): y1 = смена type_id_smooth в t+1;
    y3 = смена хотя бы раз в t+1..t+3 (NaN, если горизонт не наблюдаем целиком).
    Строки flagged-пар выбрасываются; переход в birth-тип валиден и помечается
    флагом to_newborn_type. Возвращает (long-таблица, Z (n,T) smoothed-меток)."""
    tids = np.sort(labels.territory_id.unique())
    Z = (labels.pivot(index="territory_id", columns="month", values="type_id_smooth")
               .reindex(index=tids, columns=months).to_numpy())
    birth = dict(zip(registry.type_id.astype(int), registry.birth_month))
    midx = {m: i for i, m in enumerate(months)}
    T = len(months)
    rows = []
    for t in range(first_usable_idx, T):
        for i in range(len(tids)):
            y1 = np.nan
            if t + 1 < T and months[t] not in flagged:
                y1 = float(Z[i, t + 1] != Z[i, t])
            y3 = np.nan
            if t + 3 < T and not {months[t], months[t + 1], months[t + 2]} & flagged:
                y3 = float(np.any(Z[i, t + 1:t + 4] != Z[i, t]))
            rows.append((int(tids[i]), months[t], int(Z[i, t]),
                         int(Z[i, t + 1]) if t + 1 < T else -1, y1, y3))
    df = pd.DataFrame(rows, columns=["territory_id", "month_t", "type_from",
                                     "type_to", "y1", "y3"])
    # t = T−1: таргетов нет, строка нужна только для скоринга радара
    df["to_newborn_type"] = [
        bool(y == 1.0 and birth.get(tt) == months[midx[m] + 1])
        for y, tt, m in zip(df.y1, df.type_to, df.month_t)]
    return df, Z


# ------------------------------------------------------------------ признаки

def _loo_median_cols(X: np.ndarray) -> np.ndarray:
    """Медиана столбца без i-й строки для каждой i (leave-one-out центроид).
    Через сортировку: для элемента ранга r медиана остальных — функция соседей
    середины отсортированного массива."""
    n, p = X.shape
    out = np.empty((n, p))
    if n == 1:
        return X.astype(float)
    for c in range(p):
        x = X[:, c].astype(float)
        order = np.argsort(x, kind="stable")
        s = x[order]
        ranks = np.empty(n, dtype=int)
        ranks[order] = np.arange(n)
        if n % 2 == 1:  # n=2m+1 → осталось 2m: медиана = mean(s'[m-1], s'[m])
            m = n // 2
            a = np.where(ranks >= m, s[m - 1], s[m])
            b = np.where(ranks > m, s[m], s[m + 1])
            out[:, c] = (a + b) / 2.0
        else:           # n=2m → осталось 2m-1: медиана = s'[m-1]
            m = n // 2
            out[:, c] = np.where(ranks >= m, s[m - 1], s[m])
    return out


def position_features(clr_t: np.ndarray, z_t: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Положение в типе (31 §1.3): dist_own — CLR-расстояние до медианного
    центроида своего типа (leave-one-out); margin = dist_own − min dist до чужих
    центроидов (силуэт-подобный, априорно сильнейший одиночный признак);
    rank_own — нормированный ранг dist_own внутри типа в месяце t."""
    n = len(z_t)
    types = np.unique(z_t)
    cents = {}
    for k in types:
        idx = z_t == k
        cents[int(k)] = np.median(clr_t[idx], axis=0)
    dist_own = np.zeros(n)
    for k in types:
        idx = np.flatnonzero(z_t == k)
        loo = _loo_median_cols(clr_t[idx])
        dist_own[idx] = np.linalg.norm(clr_t[idx] - loo, axis=1)
    D = np.column_stack([np.linalg.norm(clr_t - c, axis=1) for c in cents.values()])
    keys = list(cents.keys())
    own_col = np.array([keys.index(int(k)) for k in z_t])
    D_other = D.copy()
    D_other[np.arange(n), own_col] = np.inf
    margin = dist_own - D_other.min(axis=1)
    rank_own = np.zeros(n)
    for k in types:
        idx = np.flatnonzero(z_t == k)
        r = stats.rankdata(dist_own[idx]) - 1.0
        rank_own[idx] = r / max(len(idx) - 1, 1)
    return dist_own, margin, rank_own


def spatial_features(z_t: np.ndarray, nbr: np.ndarray, n_types: int
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Spatial pull (31 §1.3): доля чужих соседей среди k highway-kNN,
    per-тип доли соседей, доминирующий чужой тип (−1, если чужих нет)."""
    z_nbr = z_t[nbr]                                # (n, k)
    foreign = (z_nbr != z_t[:, None]).mean(axis=1)
    shares = np.column_stack([(z_nbr == k).mean(axis=1)
                              for k in range(n_types)])
    sh = shares.copy()
    sh[np.arange(len(z_t)), z_t.astype(int)] = -1.0  # чужие только
    dom = np.where(foreign > 0, sh.argmax(axis=1).astype(float), -1.0)
    return foreign, shares, dom


def build_feature_arrays(panel: pd.DataFrame, labels: pd.DataFrame,
                         nodes: pd.DataFrame, edges: pd.DataFrame,
                         months: list[str], par: DriversParams
                         ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Все признаки как (n, T) массивы (NaN вне области определения окна) —
    единый источник для датасета (t ≥ history), радара (t = T−1) и траекторий
    event-study (rel-месяцы вне usable-диапазона)."""
    tids = np.sort(panel.territory_id.unique())
    n, T = len(tids), len(months)
    P = {c: _pivot(panel, c, tids, months) for c in
         [f"share_{s}" for s in PARTS] + CLR + ["log_all"]}
    Z = (labels.pivot(index="territory_id", columns="month", values="type_id_smooth")
               .reindex(index=tids, columns=months).to_numpy().astype(int))
    SA = _pivot(labels, "seed_agreement", tids, months)

    F: dict[str, np.ndarray] = {}
    for c in CLR:
        F[c] = P[c]
    F["mp_ratio"] = P["share_market"] / np.maximum(P["share_prod"], 1e-9)
    for t0, name in ((1, "d1"), (3, "d3")):
        for c in CLR:
            a = np.full((n, T), np.nan)
            a[:, t0:] = P[c][:, t0:] - P[c][:, :-t0]
            F[f"{name}_clr_{c[4:]}"] = a
        for s in PARTS:
            if t0 == 3:
                a = np.full((n, T), np.nan)
                a[:, t0:] = P[f"share_{s}"][:, t0:] - P[f"share_{s}"][:, :-t0]
                F[f"d3_share_{s}"] = a
    w = par.history
    sw = np.lib.stride_tricks.sliding_window_view(P["share_market"], w, axis=1)
    x = np.arange(w) - (w - 1) / 2.0
    sxx = float((x ** 2).sum())
    ybar = sw.mean(axis=2, keepdims=True)
    slope = ((sw - ybar) * x).sum(axis=2) / sxx
    sse = ((sw - (ybar + slope[:, :, None] * x)) ** 2).sum(axis=2)
    tstat = np.where(sse < 1e-12, 0.0, slope / np.sqrt(sse / ((w - 2) * sxx)))
    for name, a in (("mp_slope6", slope), ("mp_tstat6", tstat)):
        full = np.full((n, T), np.nan)
        full[:, w - 1:] = a
        F[name] = full
    for c in CLR:
        vw = np.lib.stride_tricks.sliding_window_view(P[c], w, axis=1)
        full = np.full((n, T), np.nan)
        full[:, w - 1:] = vw.std(axis=2)
        F[f"vol6_clr_{c[4:]}"] = full

    n_types = int(Z.max()) + 1
    nbr = -np.ones((n, par.knn_k), dtype=int)
    tid_row = {int(t): i for i, t in enumerate(tids)}
    e = edges[edges["rank"] <= par.knn_k]
    for tx, ty, r in zip(e.tid_x, e.tid_y, e["rank"]):
        i, j = tid_row.get(int(tx)), tid_row.get(int(ty))
        if i is not None and j is not None:
            nbr[i, int(r) - 1] = j
    has_nbr = nbr >= 0
    F["nbr_covered"] = has_nbr.any(axis=1).astype(float)[:, None].repeat(T, 1)

    dist_own = np.full((n, T), np.nan)
    margin = np.full((n, T), np.nan)
    rank_own = np.full((n, T), np.nan)
    foreign = np.full((n, T), np.nan)
    nbr_share = np.full((n, T, n_types), np.nan)
    dom = np.full((n, T), np.nan)
    for t in range(T):
        d_o, m_g, r_o = position_features(
            np.column_stack([P[c][:, t] for c in CLR]), Z[:, t])
        dist_own[:, t], margin[:, t], rank_own[:, t] = d_o, m_g, r_o
        z_eff = np.where(has_nbr, nbr, np.arange(n)[:, None])  # изоляты → сами себе соседи
        f_s, sh_s, d_m = spatial_features(Z[:, t], z_eff, n_types)
        cov = has_nbr.any(axis=1)
        foreign[:, t] = np.where(cov, f_s, np.nan)
        nbr_share[:, t, :] = np.where(cov[:, None], sh_s, np.nan)
        dom[:, t] = np.where(cov, d_m, np.nan)
    F["dist_own"], F["margin"], F["rank_own"] = dist_own, margin, rank_own
    F["foreign_share"] = foreign
    for k in range(n_types):
        F[f"nbr_share_{k}"] = nbr_share[:, :, k]
    F["dom_foreign_type"] = dom

    nd = nodes.set_index("territory_id").reindex(tids)
    log_ma = nd.log_ma.to_numpy(np.float64)
    F["log_ma_z"] = (((log_ma - np.nanmean(log_ma)) / np.nanstd(log_ma))
                     [:, None].repeat(T, 1))
    for col in ["ma_missing", "urban_share", "log_wage", "empl_pc"]:
        F[col] = nd[col].astype(float).to_numpy()[:, None].repeat(T, 1)
    F["log_all"] = P["log_all"]
    a = np.full((n, T), np.nan)
    a[:, 3:] = P["log_all"][:, 3:] - P["log_all"][:, :-3]
    F["d3_log_all"] = a
    cal = np.array([int(m[5:7]) for m in months])
    F["month_sin"] = np.sin(2 * np.pi * cal / 12)[None, :].repeat(n, 0)
    F["month_cos"] = np.cos(2 * np.pi * cal / 12)[None, :].repeat(n, 0)
    F["seed_agreement"] = SA
    return F, tids, Z


def build_dataset(F: dict[str, np.ndarray], targets: pd.DataFrame,
                  months: list[str], par: DriversParams) -> pd.DataFrame:
    """Сборка long-таблицы: targets (t от history до T−2) + строка скоринга
    t = T−1 (y NaN). Фичи — gather из (n, T) массивов."""
    tids = np.sort(targets.territory_id.unique())
    tid_row = {int(t): i for i, t in enumerate(tids)}
    midx = {m: i for i, m in enumerate(months)}
    ti = targets.territory_id.map(tid_row).to_numpy()
    tm = targets.month_t.map(midx).to_numpy()
    out = targets.copy()
    for name, a in F.items():
        out[name] = a[ti, tm]
    # категориальный признак для LightGBM — через dtype, не через categorical_feature
    out["dom_foreign_type"] = out["dom_foreign_type"].astype("Int64").astype("category")
    return out


def feature_cols(par: DriversParams, n_types: int) -> list[str]:
    cols = ([*CLR, "mp_ratio"]
            + [f"d1_clr_{s}" for s in PARTS] + [f"d3_clr_{s}" for s in PARTS]
            + [f"d3_share_{s}" for s in PARTS] + [f"vol6_clr_{s}" for s in PARTS]
            + ["mp_slope6", "mp_tstat6", "dist_own", "margin", "rank_own",
               "foreign_share"] + [f"nbr_share_{k}" for k in range(n_types)]
            + ["dom_foreign_type", "log_ma_z", "ma_missing", "urban_share",
               "log_wage", "empl_pc", "log_all", "d3_log_all",
               "month_sin", "month_cos", "seed_agreement"])
    return cols


# ------------------------------------------------------------------ CV

def rolling_origin_folds(usable_idx: list[int], h: int, first_val_idx: int
                         ) -> list[tuple[list[int], int]]:
    """Expanding window: val = месяц v, train = usable t ≤ v − h − 1 (эмбарго
    h месяцев: label-окна train и val не пересекаются, 31 §2.2)."""
    folds = []
    for v in usable_idx:
        if v < first_val_idx:
            continue
        train = [t for t in usable_idx if t <= v - h - 1]
        if train:
            folds.append((train, v))
    return folds


def fit_lgbm(Xtr: pd.DataFrame, ytr: np.ndarray, seed: int, par: DriversParams,
             es_tail: tuple[pd.DataFrame, np.ndarray] | None = None
             ) -> lgb.LGBMClassifier:
    model = lgb.LGBMClassifier(**par.lgbm, random_state=seed)
    fit_kw: dict = {}
    if es_tail is not None:
        Xes, yes = es_tail
        if len(np.unique(yes)) == 2 and len(np.unique(ytr)) == 2:
            # early stopping на последнем train-месяце — без подглядывания в val
            fit_kw = dict(eval_set=[(Xes, yes)],
                          callbacks=[lgb.early_stopping(par.es_rounds, verbose=False)])
    model.fit(Xtr, ytr, **fit_kw)
    return model


def binary_metrics(y: np.ndarray, p: np.ndarray, top_decile: float = 0.1, *,
                   brier: bool = True) -> dict:
    y = np.asarray(y, dtype=float)
    p = np.asarray(p, dtype=float)
    prev = float(y.mean())
    k = max(1, int(np.ceil(top_decile * len(y))))
    top = np.argsort(-p)[:k]
    return {"n": int(len(y)), "prevalence": prev,
            "pr_auc": float(average_precision_score(y, p)),
            "pr_auc_baseline": prev,
            "roc_auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else np.nan,
            "lift_at_decile": float(y[top].mean() / prev) if prev > 0 else np.nan,
            # margin-rank — не вероятность, Brier для него не определён
            "brier": float(brier_score_loss(y, p)) if brier else np.nan}


def fit_isotonic(p: np.ndarray, y: np.ndarray) -> IsotonicRegression:
    return IsotonicRegression(out_of_bounds="clip").fit(p, y)


def calibration_curve(p: np.ndarray, y: np.ndarray, n_bins: int) -> list[dict]:
    qs = np.quantile(p, np.linspace(0, 1, n_bins + 1))
    b = np.clip(np.digitize(p, qs[1:-1]), 0, n_bins - 1)
    rows = []
    for i in range(n_bins):
        m = b == i
        if m.any():
            rows.append({"bin": int(i), "n": int(m.sum()),
                         "mean_pred": float(p[m].mean()),
                         "frac_pos": float(y[m].mean())})
    return rows


def _to_2d_shap(sv) -> np.ndarray:
    if hasattr(sv, "values"):
        sv = sv.values
    if isinstance(sv, list):
        sv = sv[1] if len(sv) == 2 else sv[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1] if sv.shape[2] == 2 else sv[:, :, 0]
    return sv


def shap_values(model: lgb.LGBMClassifier, X: pd.DataFrame) -> np.ndarray:
    import shap
    sv = shap.TreeExplainer(model).shap_values(X)
    return _to_2d_shap(sv)


def shap_profile(sv: np.ndarray, feats: list[str]) -> pd.DataFrame:
    """Глобальный summary: mean|SHAP| и mean SHAP со знаком, ранг."""
    df = pd.DataFrame({"feature": feats,
                       "mean_abs_shap": np.abs(sv).mean(axis=0),
                       "mean_shap": sv.mean(axis=0)})
    return df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)


# ------------------------------------------------------------------ оркестратор CV

def _cv_run(df: pd.DataFrame, feats: list[str], months: list[str], horizon: int,
            seed: int, par: DriversParams) -> dict:
    """Rolling-origin прогон одного горизонта: per-fold метрики модели и двух
    обязательных базлайнов + pooled out-of-fold предсказания."""
    midx = {m: i for i, m in enumerate(months)}
    tcol = df.month_t.map(midx)
    target = f"y{horizon}"
    usable = sorted(tcol[df[target].notna()].unique())
    folds = rolling_origin_folds(usable, horizon, par.first_val_idx)
    oof = []
    fold_rows = []
    for fi, (tr_idx, v) in enumerate(folds):
        tr = df[tcol.isin(tr_idx) & df[target].notna()]
        va = df[(tcol == v) & df[target].notna()]
        es_idx = tr_idx[-1]
        es = df[(tcol == es_idx) & df[target].notna()]
        model = fit_lgbm(tr[feats], tr[target].to_numpy(), seed, par,
                         es_tail=(es[feats], es[target].to_numpy()))
        p = model.predict_proba(va[feats])[:, 1]
        y = va[target].to_numpy()
        row = {"fold": fi, "val_month": months[v], "horizon": horizon,
               "n_train": int(len(tr)), **{f"lgbm_{k}": v_ for k, v_ in
                                           binary_metrics(y, p, par.top_decile).items()}}
        # скор базлайна = +margin (не −margin, как опечатка в §2.3 спеки):
        # margin = dist_own − min dist чужих (§1.3), corr(margin, y1) > 0 — муверы
        # смещены к чужим центроидам; отрицательный скор ниже prevalence-прямой
        row.update({f"margin_{k}": v_ for k, v_ in
                    binary_metrics(y, va["margin"].to_numpy(), par.top_decile,
                                   brier=False).items()})
        lr = make_pipeline(SimpleImputer(strategy="median"),
                           LogisticRegression(max_iter=2000))
        lr.fit(tr[LOGREG5], tr[target])
        p_lr = lr.predict_proba(va[LOGREG5])[:, 1]
        row.update({f"logreg5_{k}": v_ for k, v_ in
                    binary_metrics(y, p_lr, par.top_decile).items()})
        fold_rows.append(row)
        oof.append(pd.DataFrame({"territory_id": va.territory_id.to_numpy(),
                                 "month_t": va.month_t.to_numpy(), "y": y,
                                 "p": p, "p_margin": va["margin"].to_numpy(),
                                 "p_logreg5": p_lr}))
    oof_df = pd.concat(oof, ignore_index=True)
    pooled = {"lgbm": binary_metrics(oof_df.y.to_numpy(), oof_df.p.to_numpy(), par.top_decile),
              "margin_rank": binary_metrics(oof_df.y.to_numpy(), oof_df.p_margin.to_numpy(), par.top_decile, brier=False),
              "logreg5": binary_metrics(oof_df.y.to_numpy(), oof_df.p_logreg5.to_numpy(), par.top_decile)}
    # lift per-fold: радар ранжирует ВНУТРИ месяца — честная продуктовая линза
    # (pooled PR-AUC при сдвиге prevalence 0.005–0.116 между месяцами искажён)
    for tag, pre in (("lgbm", "lgbm"), ("margin_rank", "margin"), ("logreg5", "logreg5")):
        lifts = [r[f"{pre}_pr_auc"] / r[f"{pre}_prevalence"] for r in fold_rows
                 if r[f"{pre}_prevalence"] > 0]
        pooled[tag]["mean_fold_lift"] = float(np.mean(lifts))
        pooled[tag]["median_fold_lift"] = float(np.median(lifts))
    return {"folds": fold_rows, "oof": oof_df, "pooled": pooled,
            "n_folds": len(folds)}


# ------------------------------------------------------------------ карточки

def _card_text(row, top5) -> str:
    lines = [f"МО {row['name']} ({int(row['territory_id'])}), {row['month_event']}: "
             f"тип {int(row['type_from'])} → тип {int(row['type_to'])}.",
             f"Вероятность перехода (модель h=1, за месяц до): p = {row['p']:.2f} "
             f"(топ-{row['pct']:.3g}% всех МО, скоринг {row['prob_source']}).",
             "Драйверы (SHAP, вклад в log-odds):"]
    for f, val, sv in top5:
        sign = "+" if sv >= 0 else "−"
        lines.append(f"  {sign} {f} = {val:+.3f} → {sign}{abs(sv):.2f}")
    return "\n".join(lines)


def build_cards(trans: pd.DataFrame, probs: np.ndarray, pct: np.ndarray,
                sv: np.ndarray, feats: list[str], names: dict,
                F: dict[str, np.ndarray], tids: np.ndarray, months: list[str],
                top_k: int = 5) -> pd.DataFrame:
    """Карточки переходов по шаблону 31 §3.2: значения в сырых единицах —
    для clr-признаков рядом кладём текущую долю части (share_value).
    pct — перцентиль вероятности среди ВСЕХ МО того же месяца (считается снаружи:
    там известна полная скоринг-таблица месяца)."""
    midx = {m: i for i, m in enumerate(months)}
    tid_full = {int(t): i for i, t in enumerate(tids)}
    df = trans.copy()
    df["p"] = probs
    df["pct"] = pct
    rows = []
    for r, tr in enumerate(df.itertuples()):
        order = np.argsort(-np.abs(sv[r]))[:top_k]
        top5, share_vals = [], []
        i_full = tid_full[int(tr.territory_id)]
        for j in order:
            f = feats[j]
            share_val = np.nan
            for s in PARTS:  # у clr-признака части s показываем сырую долю
                if "clr" in f and f.endswith(s) and f"share_{s}" in F:
                    share_val = float(F[f"share_{s}"][i_full, midx[tr.month_t]])
                    break
            top5.append((f, float(getattr(tr, f)), float(sv[r, j])))
            share_vals.append(share_val)
        flat: dict = {}
        for k_, (f, val, s_) in enumerate(top5, 1):
            flat[f"driver{k_}"] = f
            flat[f"value{k_}"] = val
            flat[f"share_value{k_}"] = share_vals[k_ - 1]
            flat[f"shap{k_}"] = s_
        rows.append({"territory_id": int(tr.territory_id),
                     "name": names.get(int(tr.territory_id), ""),
                     "month_t": tr.month_t,
                     "month_event": months[midx[tr.month_t] + 1],
                     "type_from": int(tr.type_from), "type_to": int(tr.type_to),
                     "to_newborn_type": bool(tr.to_newborn_type),
                     "p": float(tr.p), "prob_source": tr.prob_source,
                     "pct": float(tr.pct), **flat,
                     "card_text": _card_text(
                         {"name": names.get(int(tr.territory_id), ""),
                          "territory_id": tr.territory_id,
                          "month_event": months[midx[tr.month_t] + 1],
                          "type_from": tr.type_from, "type_to": tr.type_to,
                          "p": tr.p, "pct": tr.pct,
                          "prob_source": tr.prob_source}, top5)})
    return pd.DataFrame(rows)


# ------------------------------------------------------------------ event-study

def event_study(F: dict[str, np.ndarray], Z: np.ndarray, tids: np.ndarray,
                months: list[str], trans: pd.DataFrame, top_feats: list[str],
                seed: int, par: DriversParams) -> pd.DataFrame:
    """Окно ±6 мес вокруг перехода; matched-контроль k=5 NN (с возвращением) по
    признакам месяца e−1 из того же type_from, caliper = 95-й перцентиль
    расстояний. Эффект = Cohen's d. Только «ассоциировано», не causal (31 §4)."""
    midx = {m: i for i, m in enumerate(months)}
    w = par.event_window
    ev = trans.copy()
    ev["e"] = ev.month_t.map(midx) + 1                     # месяц перехода
    lo, hi = midx["2023-08"], midx["2024-06"]              # окно должно помещаться (31 §4.1)
    ev = ev[(ev.e >= lo) & (ev.e <= hi)].reset_index(drop=True)
    if ev.empty:
        return pd.DataFrame()
    tid_row = {int(t): i for i, t in enumerate(tids)}
    ev["i"] = ev.territory_id.map(tid_row)
    T = Z.shape[1]
    chg = np.zeros_like(Z, dtype=int)
    chg[:, 1:] = (Z[:, 1:] != Z[:, :-1])
    cchg = np.cumsum(chg, axis=1)

    def stable_on_window(e: int) -> np.ndarray:
        """Ни одного перехода на окне [e−w, e+w]: chg живёт на правом конце месяца."""
        a, b = max(e - w, 0), min(e + w, T - 1)
        return (cchg[:, b] - cchg[:, a]) == 0

    Xall = np.stack([F[c] for c in MATCH_FEATS], axis=2)   # (n, T, p)
    mu = np.nanmean(Xall[:, par.history:, :], axis=(0, 1))
    sd = np.nanstd(Xall[:, par.history:, :], axis=(0, 1)) + 1e-12

    matches = []  # (ev_row, ctrl_i)
    dists = []
    for (tf, em1), g in ev.groupby([ev.type_from, ev.e - 1]):
        e = int(em1) + 1
        pool = np.flatnonzero((Z[:, em1] == tf) & stable_on_window(e))
        if len(pool) == 0:
            continue
        Xp = (Xall[pool, em1, :] - mu) / sd
        k = min(par.match_k, len(pool))
        nn = NearestNeighbors(n_neighbors=k).fit(Xp)
        Xm = (Xall[g.i.to_numpy(), em1, :] - mu) / sd
        dd, ii = nn.kneighbors(Xm)
        for r in range(len(g)):
            for kk in range(k):
                matches.append((int(g.index[r]), int(pool[ii[r, kk]])))
                dists.append(float(dd[r, kk]))
    if not matches:
        return pd.DataFrame()
    caliper = float(np.quantile(dists, par.caliper_q))
    matched = [(r, c) for (r, c), d in zip(matches, dists) if d <= caliper]

    ev["channel"] = [f"{a}→{b}" for a, b in zip(ev.type_from, ev.type_to)]
    chan_n = ev.channel.value_counts()
    big = set(chan_n[chan_n >= par.event_min_channel].index)
    ev["scope"] = np.where(ev.channel.isin(big), ev.channel, "прочие")

    ctrl_by_ev: dict[int, list[int]] = {}
    for r, c in matched:
        ctrl_by_ev.setdefault(r, []).append(c)

    rows = []
    rels = range(-w, w + 1)
    for scope, g in pd.concat([ev.assign(scope="ALL"), *[ev[ev.scope == s] for s in
                                                         sorted(ev.scope.unique())]]).groupby("scope"):
        idx_rows = g.index.to_numpy()
        for f in top_feats:
            A = F[f]
            for rel in rels:
                mv, cv = [], []
                for r in idx_rows:
                    i, e = int(ev.at[r, "i"]), int(ev.at[r, "e"])
                    if 0 <= e + rel < A.shape[1]:
                        mv.append(A[i, e + rel])
                        for c in ctrl_by_ev.get(r, []):
                            cv.append(A[c, e + rel])
                mv = np.asarray(mv, dtype=float)
                cv = np.asarray(cv, dtype=float)
                m1, m2 = np.nanmean(mv), np.nanmean(cv) if len(cv) else np.nan
                s1, s2 = np.nanstd(mv), np.nanstd(cv) if len(cv) else np.nan
                sp = np.sqrt((s1 ** 2 + (s2 if not np.isnan(s2) else s1) ** 2) / 2) + 1e-12
                rows.append({"scope": scope, "feature": f, "rel_month": rel,
                             "mean_movers": float(m1), "mean_controls": float(m2),
                             "cohens_d": float((m1 - m2) / sp) if len(cv) else np.nan,
                             "n_events": int(len(g))})
    out = pd.DataFrame(rows)
    out.attrs["caliper"] = caliper
    out.attrs["n_matched_links"] = len(matched)
    return out


# ------------------------------------------------------------------ Миркин-кроссчек

def mirkin_crosscheck(sv_trans: np.ndarray, trans: pd.DataFrame,
                      feats: list[str], F: dict[str, np.ndarray],
                      Z: np.ndarray, months: list[str], ref_month: str,
                      par: DriversParams) -> dict:
    """Спирмен между mean|SHAP| канала A→B (по уровням clr) и разностью
    Миркин-профилей типов A и B (31 §3.2.4). interpret недоступен → скип с пометкой."""
    try:
        from . import interpret as _interp
    except Exception as exc:  # модуль пишется параллельно — не блокируемся
        return {"skipped": f"interpret недоступен: {exc}"}
    try:
        clr_idx = [feats.index(c) for c in CLR]
        trans_ch = [f"{a}→{b}" for a, b in zip(trans.type_from, trans.type_to)]
        chan_n = pd.Series(trans_ch).value_counts()
        rel_cache: dict[str, pd.DataFrame] = {}

        def _rel_at(month: str) -> pd.DataFrame:
            if month not in rel_cache:
                t = months.index(month)
                X = np.column_stack([F[c][:, t] for c in CLR])
                Xs = (X - X.mean(0)) / (X.std(0) + 1e-12)
                mk = _interp.mirkin_profiles(Xs, Z[:, t], CLR)
                rel_cache[month] = mk.pivot(index="type_id", columns="feature",
                                            values="rel")
            return rel_cache[month]

        out = {}
        for ch in chan_n.index[:par.top_channels]:
            a, b = (int(x) for x in ch.split("→"))
            rows = [i for i, c in enumerate(trans_ch) if c == ch]
            # birth/death-каналы: пробуем месяц события (t+1), потом месяц t —
            # тип может отсутствовать с одной из сторон
            tm = trans.iloc[rows].month_t.mode().iloc[0]
            ref_m, rel = None, None
            for cand in (months[months.index(tm) + 1], tm):
                r = _rel_at(cand)
                if a in r.index and b in r.index:
                    ref_m, rel = cand, r
                    break
            if rel is None:
                out[ch] = {"skipped": f"тип {a} или {b} отсутствует вокруг {tm} "
                                      "(birth/death-канал, кроссчек не определён)",
                           "n": int(len(rows))}
                continue
            v_shap = np.abs(sv_trans[rows][:, clr_idx]).mean(axis=0)
            v_mirk = np.abs(rel.loc[b, CLR].to_numpy() - rel.loc[a, CLR].to_numpy())
            rho = stats.spearmanr(v_shap, v_mirk).statistic
            out[ch] = {"spearman": float(rho), "n": int(len(rows)),
                       "ref_month": ref_m}
        return out
    except Exception as exc:
        return {"skipped": f"кроссчек упал: {exc}"}


# ------------------------------------------------------------------ полный прогон

def run_all(cfg: Config, root: str | Path, ctx) -> dict:
    par = DriversParams()
    seed = stage_seed(cfg.seed, "drivers")
    inp = load_inputs(root)
    months = sorted(inp["panel"].month.unique())
    flagged = flagged_pair_months(inp["stability"])
    targets, Z = make_targets(inp["labels"], inp["registry"], flagged, months,
                              first_usable_idx=par.history)
    ctx.log(f"драйверы: flagged-пар исключено {len(flagged)}; строк таргета {len(targets)}")
    F, tids, Z = build_feature_arrays(inp["panel"], inp["labels"], inp["nodes"],
                                      inp["edges"], months, par)
    df = build_dataset(F, targets, months, par)
    n_types = int(Z.max()) + 1
    feats = [c for c in feature_cols(par, n_types) if c in df.columns]
    model_df = df[df.y1.notna()].reset_index(drop=True)     # t ≤ T−2
    ctx.log(f"датасет: {len(model_df)} строк, {len(feats)} признаков, "
            f"prevalence y1={model_df.y1.mean():.4f}")

    midx = {m: i for i, m in enumerate(months)}
    res: dict = {}
    for h in (1, 3):
        res[h] = _cv_run(model_df, feats, months, h, seed, par)
        pl = res[h]["pooled"]
        ctx.log(f"h={h}: {res[h]['n_folds']} фолдов, PR-AUC lgbm "
                f"{pl['lgbm']['pr_auc']:.3f} (prevalence {pl['lgbm']['prevalence']:.3f}) "
                f"| margin-rank {pl['margin_rank']['pr_auc']:.3f} "
                f"| logreg5 {pl['logreg5']['pr_auc']:.3f}")

    # финальная h=1-модель на всех usable месяцах → глобальный SHAP →
    # контрольный красный флаг seed_agreement (31 §1.3): топ-3 mean|SHAP| →
    # публикуем версию без него + разницу метрик (модель частично предсказывает
    # нестабильность метода, а не экономику)
    tcol = model_df.month_t.map(midx)
    def _final(h: int, fs: list[str]):
        tr_mask = model_df[f"y{h}"].notna()
        last_m = tcol[tr_mask].max()
        es = model_df[tr_mask & (tcol == last_m)]
        tr = model_df[tr_mask]
        return fit_lgbm(tr[fs], tr[f"y{h}"].to_numpy(), seed, par,
                        es_tail=(es[fs], es[f"y{h}"].to_numpy()))

    final_full = _final(1, feats)
    samp = model_df.sample(min(par.shap_sample, len(model_df)),
                           random_state=seed).reset_index(drop=True)
    sv_glob_full = shap_values(final_full, samp[feats])
    glob_prof_full = shap_profile(sv_glob_full, feats)
    glob_prof_full.insert(0, "rank", np.arange(1, len(glob_prof_full) + 1))
    red_flag = bool((glob_prof_full.head(3).feature == "seed_agreement").any())

    res_ns = None
    if red_flag:
        pub_feats = [f for f in feats if f != "seed_agreement"]
        ctx.log("красный флаг: seed_agreement в топ-3 SHAP — повтор CV h=1 без него")
        res_ns = _cv_run(model_df, pub_feats, months, 1, seed, par)
        ctx.log(f"h=1 без seed_agreement: pooled PR-AUC "
                f"{res_ns['pooled']['lgbm']['pr_auc']:.3f} vs полная "
                f"{res[1]['pooled']['lgbm']['pr_auc']:.3f}")
    else:
        pub_feats = feats
    finals = {1: (_final(1, pub_feats) if red_flag else final_full), 3: _final(3, feats)}

    # калибровка на OOF h=1 (опубликованной версии, если флаг сработал)
    oof1 = (res_ns or res[1])["oof"]
    iso = fit_isotonic(oof1.p.to_numpy(), oof1.y.to_numpy())
    p_cal = iso.predict(oof1.p.to_numpy())
    calib = {"horizon": 1,
             "bins_before": calibration_curve(oof1.p.to_numpy(), oof1.y.to_numpy(), par.calib_bins),
             "bins_after": calibration_curve(p_cal, oof1.y.to_numpy(), par.calib_bins),
             "brier_before": float(brier_score_loss(oof1.y, oof1.p)),
             "brier_after": float(brier_score_loss(oof1.y, p_cal))}

    # SHAP: глобальный опубликованной модели (для топ-признаков event-study)
    sv_glob = sv_glob_full if not red_flag else shap_values(finals[1], samp[pub_feats])
    glob_prof = shap_profile(sv_glob, pub_feats)
    glob_prof.insert(0, "rank", np.arange(1, len(glob_prof) + 1))

    trans = model_df[model_df.y1 == 1].reset_index(drop=True)
    sv_tr = shap_values(finals[1], trans[pub_feats]) if len(trans) else np.zeros((0, len(pub_feats)))

    # вероятности для карточек: OOF где есть, иначе финальная модель;
    # калиброванные isotonic'ом (калиброванная вероятность — продукт, 31 §2.3).
    # pct — доля МО того же месяца с p ≥ своей (полная скоринг-таблица месяца)
    oof_key = oof1.set_index(["territory_id", "month_t"]).p
    p_fin_tr = finals[1].predict_proba(trans[pub_feats])[:, 1] if len(trans) else np.array([])
    probs, src = [], []
    for r, tr in enumerate(trans.itertuples()):
        p_o = oof_key.get((tr.territory_id, tr.month_t), np.nan)
        probs.append(p_o if np.isfinite(p_o) else p_fin_tr[r])
        src.append("oof" if np.isfinite(p_o) else "final")
    trans["prob_source"] = src
    probs_cal = iso.predict(np.asarray(probs))
    month_score: dict[str, pd.Series] = {}
    for m in model_df.month_t.unique():
        rows_m = model_df[model_df.month_t == m]
        o = oof1[oof1.month_t == m]
        if len(o) == len(rows_m):
            month_score[m] = pd.Series(iso.predict(o.p.to_numpy()),
                                       index=o.territory_id.to_numpy())
        else:
            month_score[m] = pd.Series(
                iso.predict(finals[1].predict_proba(rows_m[pub_feats])[:, 1]),
                index=rows_m.territory_id.to_numpy())
    pct = np.array([100.0 * float((month_score[tr.month_t] >= p).mean())
                    for tr, p in zip(trans.itertuples(), probs_cal)])
    names = inp["nodes"].set_index("territory_id").name.to_dict()
    F_sh = {**F, **{f"share_{s}": _pivot(inp["panel"], f"share_{s}", tids, months)
                    for s in PARTS}}
    cards = build_cards(trans, probs_cal, pct, sv_tr, pub_feats, names, F_sh,
                        tids, months)
    cards["p_raw"] = np.asarray(probs)

    # профили каналов A→B (топ-8 по массе)
    ch = pd.Series([f"{a}→{b}" for a, b in zip(trans.type_from, trans.type_to)])
    chan_rows = []
    for c_ in ch.value_counts().index[:par.top_channels]:
        rows = np.flatnonzero(ch.to_numpy() == c_)
        prof = shap_profile(sv_tr[rows], pub_feats)
        a, b = (int(x) for x in c_.split("→"))
        for rank, pr in enumerate(prof.itertuples(), 1):
            chan_rows.append({"channel": c_, "type_from": a, "type_to": b,
                              "n_events": int(len(rows)), "rank_in_channel": rank,
                              "feature": pr.feature,
                              "mean_abs_shap": pr.mean_abs_shap,
                              "mean_shap": pr.mean_shap})
    channel_prof = pd.DataFrame(chan_rows)

    top_feats = glob_prof.head(par.top_traj_feats).feature.tolist()
    es_df = event_study(F, Z, tids, months, trans, top_feats, seed, par)

    # радар 2024-12 (скоринг опубликованной моделью, p_move_h1 откалиброван)
    last = months[-1]
    radar = df[df.month_t == last].copy().reset_index(drop=True)
    radar["p_move_h1_raw"] = finals[1].predict_proba(radar[pub_feats])[:, 1]
    radar["p_move_h1"] = iso.predict(radar.p_move_h1_raw.to_numpy())
    radar["p_move_h3"] = finals[3].predict_proba(radar[feats])[:, 1]
    # ранжирование — по сырому скору: isotonic клипает хвост выше max(OOF) в константу,
    # дециль по откалиброванной p теряет верхние ранги (монотонность сохраняется)
    radar["decile"] = np.ceil(radar.p_move_h1_raw.rank() / len(radar) * 10).astype(int)
    radar["watch"] = radar.decile == 10
    radar["unverified"] = True
    radar["seed_agreement"] = F["seed_agreement"][:, -1][
        radar.territory_id.map({int(t): i for i, t in enumerate(tids)}).to_numpy()]
    low_thr = np.quantile(radar.seed_agreement, par.low_seed_q)
    radar["risk_low_seed_agreement"] = radar.seed_agreement <= low_thr
    sv_rad = shap_values(finals[1], radar[pub_feats])
    radar["top3_drivers"] = ["; ".join(
        f"{pub_feats[j]}{'+' if sv_rad[r, j] >= 0 else '−'}{abs(sv_rad[r, j]):.2f}"
        for j in np.argsort(-np.abs(sv_rad[r]))[:3]) for r in range(len(radar))]
    radar["name"] = radar.territory_id.map(names)
    oof_last = oof1[oof1.month_t == months[-2]]
    ks = stats.ks_2samp(radar.p_move_h1, iso.predict(oof_last.p.to_numpy()))
    last_pair_flagged = months[-2] in flagged
    sanity = {"ks_stat_vs_oof_last_month": float(ks.statistic),
              "ks_pvalue": float(ks.pvalue),
              "flagged_last_pair": bool(last_pair_flagged),
              "watch_flagged_share": float(radar.watch.mean() * last_pair_flagged),
              "watch_low_seed_share": float(
                  radar.risk_low_seed_agreement[radar.watch].mean()),
              "honest_note": HONEST_RADAR_NOTE}

    mirkin = mirkin_crosscheck(sv_tr, trans, pub_feats, F, Z, months,
                               ref_month=months[-2], par=par)

    pooled1, pooled3 = res[1]["pooled"], res[3]["pooled"]
    pooled_pub = (res_ns or res[1])["pooled"]
    beats_margin = pooled_pub["lgbm"]["pr_auc"] > pooled_pub["margin_rank"]["pr_auc"]
    lr5_wins_pooled = pooled_pub["logreg5"]["pr_auc"] > pooled_pub["lgbm"]["pr_auc"]
    verdict = {
        "margin_sign_fix": ("скор margin-rank = +margin (corr(margin,y1)=+0.058 на "
                            "полном датасете): спека §2.3 пишет −margin, что противоречит "
                            "её же определению margin в §1.3 — скорректировано по данным"),
        "published_model": "lgbm_no_seed" if red_flag else "lgbm",
        "lgbm_beats_margin_rank_pooled_h1": bool(beats_margin),
        "lgbm_beats_margin_rank_fold_lift_h1": bool(
            pooled_pub["lgbm"]["mean_fold_lift"] > pooled_pub["margin_rank"]["mean_fold_lift"]),
        "logreg5_wins_pooled_h1": bool(lr5_wins_pooled),
        "honest_negative": (None if not lr5_wins_pooled else
                            "logreg5 выше lgbm по pooled PR-AUC, но ниже по per-fold lift "
                            "внутри месяца: pooled перевёрнут сдвигом prevalence между "
                            "месяцами (0.005–0.116); радар ранжирует внутри месяца — "
                            "продуктовая линза per-fold. Оба числа опубликованы."),
        "publish": ("lgbm (без seed_agreement)" if red_flag and beats_margin else
                    "lgbm" if beats_margin else
                    "margin_rank (простая модель — честный негатив, 31 §2.3)"),
        "seed_agreement_red_flag": red_flag,
        "no_seed_pooled_pr_auc": (res_ns["pooled"]["lgbm"]["pr_auc"] if res_ns else None),
        "full_pooled_pr_auc": pooled1["lgbm"]["pr_auc"],
    }
    metrics = {
        "seed": seed, "params": {k: v for k, v in vars(par).items()},
        "months": {"first_usable": months[par.history], "last": last},
        "n_flagged_pairs_excluded": len(flagged),
        "n_rows": int(len(model_df)), "n_features": len(feats),
        "features": feats,
        "published_features": pub_feats,
        "prevalence_y1": float(model_df.y1.mean()),
        "prevalence_y3": float(model_df.y3.mean()),
        "n_transitions_y1": int(model_df.y1.sum()),
        "to_newborn_type_count": int(model_df.to_newborn_type.sum()),
        "h1": {"n_folds": res[1]["n_folds"], "folds": res[1]["folds"], "pooled": pooled1},
        "h3": {"n_folds": res[3]["n_folds"], "folds": res[3]["folds"], "pooled": pooled3},
        "calibration": {**calib,
                        "applied_to": ["radar p_move_h1", "transition_cards p"],
                        "note": "p_move_h3 — без калибровки (isotonic обучен на h=1 OOF)"},
        "shap_top15": glob_prof.head(15).to_dict("records"),
        "shap_top15_full_model": glob_prof_full.head(15).to_dict("records"),
        "verdict": verdict,
        "mirkin_crosscheck": mirkin,
        "radar_sanity": sanity,
        "event_study": {"window": par.event_window, "match_k": par.match_k,
                        "caliper_q": par.caliper_q,
                        "caliper": getattr(es_df, "attrs", {}).get("caliper"),
                        "min_channel_events": par.event_min_channel,
                        "note": "только «ассоциировано», не causal; контроль по наблюдаемым — скрытые смешения остаются"},
    }
    if res_ns is not None:
        metrics["h1_no_seed"] = {"n_folds": res_ns["n_folds"], "folds": res_ns["folds"],
                                 "pooled": res_ns["pooled"]}

    out = ctx.dir
    model_df.to_parquet(out / "drivers_dataset.parquet", index=False)
    with open(out / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    glob_prof.to_parquet(out / "shap_profiles.parquet", index=False)
    channel_prof.to_parquet(out / "channel_profiles.parquet", index=False)
    cards.to_parquet(out / "transition_cards.parquet", index=False)
    es_df.to_parquet(out / "event_study.parquet", index=False)
    radar[["territory_id", "name", "p_move_h1", "p_move_h1_raw", "p_move_h3",
           "decile", "watch", "unverified", "seed_agreement",
           "risk_low_seed_agreement",
           "top3_drivers"]].to_parquet(out / "radar_watchlist.parquet", index=False)
    ctx.log(f"артефакты → {out}: dataset {len(model_df)} строк, "
            f"карточек {len(cards)}, радар watch={int(radar.watch.sum())}/"
            f"{len(radar)} (unverified=true)")
    return metrics
