"""Сравнительный слой (PREREG §A меры / §B методы): композиты, margin rule,
чувствительность. Сборка 05.10.2026.

Все веса и пороги читаются из configs/prereg.yaml через load_prereg() —
литералов композита в коде нет (CI-проверка test_benchmark сверяет чувствительность
композита к файлу весов).

Дизайн-решения (зафиксированы здесь, не в prereg — уровень реализации):
- z-нормировка — по кандидатам внутри ноги (ddof=0); нулевая дисперсия → все z=0
  (нога не дискриминирует, а не взрывается).
- Q-нога: z каждого саб-индекса по кандидатам → среднее (5 саб-индексов: SW, CH/N,
  −S_Dbw для Leiden-консенсуса; MQ-mancoridis, internal-edge ratio для K-means) →
  z(Q) по кандидатам в композите. Перекрёстная схема prereg §A: Leiden судится
  ТОЛЬКО атрибутивными индексами на X_static, K-means — ТОЛЬКО сетевыми на графе
  меры (анти-циркулярность; циркулярные ячейки помечаются в table_B флагом).
- K-means K-протокол (prereg §C): вето стабильности (медиана попарного ARI
  30 seeds ≥ 0.90) → argmin S_Dbw внутри прошедших. BIC — для GMM (04), к KMeans
  неприменим. Если вето не проходит ни один K — берём argmax ARI с флагом
  veto_failed=True (честная деградация, не молчаливая).
- T-нога: odd/even месяцы (PREREG_DEVIATIONS №5: split-half 12/12 вырожден при
  month-of-year десезонизации). Мера пересчитывается на половинах, кластер —
  Leiden-CSPA на kNN-графе половины, ARI инвариантен к перестановке меток
  (Hungarian-матчинг считается и репортится как карта соответствия кластеров).
  M8 статична → T=1 тривиально (честное свойство, как bootJ=1.0 в гейте).
- R-нога: PP-Dir центральная ячейка (K=6, δ=0.05, α_btw=80, μ_edge=0.25, n=2016,
  T=24), 10 инстансов; мера пересчитывается на синтетических рядах через адаптер
  synthetic_measure_data; истина — модальные метки (потолок статической NMI при
  δ=0.05 пересчитывается кодом, не литералом — PREREG_DEVIATIONS №1).
- Парный бутстреп margin rule: единицы репликации — seed-пары ARI (S-нога,
  по обоим эстиматорам) и синтетические инстансы (R-нога); индексы ресэмпла общие
  для всех мер (парность). Q/T/H — точечные значения без естественной репликации,
  в бутстрепе фиксированы (их дисперсия недоступна без пересборки данных —
  оговорено в выходе).
- LOMO × 5 = leave-one-leg-out (5 ног), веса оставшихся перенормируются на сумму 1.
  «Победитель меняется в >1 из 5» (prereg) = число ног, чьё удаление смещает
  топ-1, строго больше 1.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.stats import wilcoxon as _scipy_wilcoxon
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from . import icvi
from . import measures as ms
from .cluster import leiden_consensus_cspa, run_leiden
from .config import Config, load_prereg
from .seeds import stage_seed
from .synthetic import ari, nmi

log = logging.getLogger(__name__)

__all__ = [
    "zscore", "gini", "internal_edge_ratio", "hungarian_mapping", "modal_labels",
    "f1_report_value",
    "graph_from_npz", "synthetic_measure_data", "synthetic_feature_matrix",
    "leiden_ensemble", "kmeans_protocol", "scoring_candidates",
    "oddeven_halves", "measure_graph_from_sim",
    "leg_zscores", "composite", "paired_bootstrap", "margin_verdict",
    "dirichlet_sensitivity", "lomo_winners", "borda_table",
    "METHODS_LIGHT", "METHODS_HEAVY", "run_method", "method_stability",
    "method_composite", "wilcoxon_table", "bh_adjust",
]

# Лёгкие методы (вся сетка синтетики) и тяжёлые (только центральные ячейки,
# prereg §B). dmon — опциональный try-import, при отсутствии TF честный skip.
METHODS_LIGHT = ["leiden_consensus", "kmeans", "ward", "gmm_full", "gmm_diag",
                 "spectral_knn", "kefrin", "louvain", "infomap"]
METHODS_HEAVY = ["eva", "dmon"]


# ---------------------------------------------------------------- общее

def zscore(s: pd.Series | np.ndarray) -> pd.Series:
    """z по кандидатам внутри ноги (ddof=0). Нулевая дисперсия → все 0."""
    idx = s.index if isinstance(s, pd.Series) else None
    s = pd.Series(np.asarray(s, dtype=np.float64), index=idx)
    sd = s.std(ddof=0)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - s.mean()) / sd


def gini(x: np.ndarray) -> float:
    """Джини неотрицательного вектора (степени графа). Константный вектор → 0."""
    x = np.sort(np.asarray(x, dtype=np.float64))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    cum = np.cumsum(x)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def internal_edge_ratio(A: sp.csr_matrix, labels: np.ndarray) -> float:
    """Доля веса рёбер внутри кластеров (сетевой судья K-means, prereg §A)."""
    A = sp.csr_matrix(A)
    labels = np.asarray(labels)
    coo = sp.triu(A, k=1).tocoo()
    inside = labels[coo.row] == labels[coo.col]
    tot = coo.data.sum()
    return float(coo.data[inside].sum() / tot) if tot > 0 else 0.0


def hungarian_mapping(a: np.ndarray, b: np.ndarray) -> dict[int, int]:
    """Hungarian-матчинг меток a→b по стоимости 1−Jaccard (τ-протокол динамики).
    ARI инвариантен к перестановке — карта нужна для отчётности, не для ARI."""
    a, b = np.asarray(a), np.asarray(b)
    ca, cb = np.unique(a), np.unique(b)
    cost = np.ones((len(ca), len(cb)))
    for i, x in enumerate(ca):
        ma = a == x
        for j, y in enumerate(cb):
            mb = b == y
            inter = np.logical_and(ma, mb).sum()
            union = np.logical_or(ma, mb).sum()
            cost[i, j] = 1.0 - (inter / union if union else 0.0)
    r, c = linear_sum_assignment(cost)
    return {int(ca[i]): int(cb[j]) for i, j in zip(r, c)}


def modal_labels(z: np.ndarray) -> np.ndarray:
    """(T,n) помесячные метки → (n,) модальные (статическая истина синтетики)."""
    z = np.asarray(z)
    n = z.shape[1]
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        out[i] = np.bincount(z[:, i]).argmax()
    return out


def f1_report_value(z_true: np.ndarray, z_pred: np.ndarray, window: int = 1) -> float:
    """F1 детекции смены типа для отчётного дрейф-блока (06b).

    Диагноз по верификации 06b (05.10, f1_mean=0.0 у всех методов): багом не
    является. Все методы зоопарка статические, их метки тайлятся по месяцам →
    предсказанных смен 0 → при δ>0 tp=0, fn=|дрейф| → F1≡0 структурно (метрика
    осмысленна только для per-snapshot методов, которых в 06b нет). При δ=0
    истинных событий нет вообще → F1 = 0/0 формально не определён; сырые 0.0 в
    таких ячейках читались бы как «провал детекции», поэтому здесь → NaN
    («нет событий»). f1_type_switch (synthetic.py) не трогаем — её арифметика
    верна (истина vs истина = 1.0; смена в окне ±window = 1.0; вне окна = 0.0).
    """
    from .synthetic import f1_type_switch

    z_true = np.asarray(z_true)
    if not bool((z_true[1:] != z_true[:-1]).any()):
        return float("nan")
    return f1_type_switch(z_true, z_pred, window=window)


# ---------------------------------------------------------------- графы мер

def _shift_nonneg(w: np.ndarray) -> np.ndarray:
    """Приведение весов к неотрицательным (Leiden RBConfiguration отказывает
    отрицательные). Шкала со значимым нулём (знакопеременные сходства —
    корреляции: 0 = «нет близости») → отсечение нуля, как прод-граф 03
    (clip(R,0)); шкала целиком ≤ 0 (отрицанные дистанции M5–M7, нуля в ней
    нет) → сдвиг на минимум (слабейшее удержанное kNN-ребро получает вес 0,
    порядок сохраняется). Сдвиг corr-мер запрещён: он раздувает шкалу весов
    и смещает эффективное разрешение RB-γ (проверено: M1 при сдвиге
    вырождается в k=1). Уровень реализации, зафиксирован здесь."""
    if w.size == 0:
        return w
    if float(w.max()) <= 0.0:
        return w - float(w.min())
    return np.clip(w, 0.0, None)


def graph_from_npz(path: str | Path) -> sp.csr_matrix:
    """npz этапа 06a (edge_index i<j, weights=сходство) → симметричный csr,
    диагональ 0, веса неотрицательны (_shift_nonneg). Порядок узлов = порядок
    territory_id в файле (= node_index)."""
    d = np.load(path)
    ei, w = d["edge_index"].astype(np.int64), d["weights"].astype(np.float64)
    n = int(len(d["territory_id"]))
    A = sp.csr_matrix((_shift_nonneg(w), (ei[:, 0], ei[:, 1])), shape=(n, n))
    A = A + A.T  # edge_index уникален (i<j) — сложение не дублирует
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A.tocsr()


def measure_graph_from_sim(res: ms.SimResult, k: int) -> sp.csr_matrix:
    """SimResult → единый kNN(k)-union csr (тот же формат, что этап 06a),
    веса неотрицательны (_shift_nonneg)."""
    ei, w = ms.knn_union_graph(res, k)
    n = res.matrix.shape[0]
    A = sp.csr_matrix((_shift_nonneg(w.astype(np.float64)), (ei[:, 0], ei[:, 1])),
                      shape=(n, n))
    A = A + A.T
    A.setdiag(0.0)
    A.eliminate_zeros()
    return A.tocsr()


# ---------------------------------------------------------------- синтетика → MeasureData

def synthetic_measure_data(ds: dict) -> ms.MeasureData:
    """Адаптер PP-Dir/LFR-Dir (ecotypes.synthetic) → MeasureData для пересчёта
    мер M1–M10 на синтетических рядах (нога R).

    Отображение: shares ← X (T,n,6); log_all ← log V (V — суммарный объём);
    sa_all — month-of-year десезонизация (та же функция, что panel.py);
    vals (n,T,5) = доля × объём (абсолютные категориальные значения);
    pop ← среднемесячный объём узла (прокси размера; у синтетики нет населения);
    hw ← kNN(10) по латентным geo-координатам (евклид, шкала уже в км —
    медиана попарных дистанций откалибрована генератором под 1816 км).
    region=None (топологические статистики вне скоринга)."""
    X, V, geo = ds["X"], ds["V"], ds["geo"]
    T, n, _ = X.shape
    shares = np.ascontiguousarray(X.transpose(1, 0, 2))          # (n,T,6)
    log_all = np.log(np.maximum(V.T, 1e-12))                     # (n,T)
    sa_all = ms.sa_from_log(log_all)
    clr = ms.clr_transform(shares)
    vals = shares[:, :, :5] * V.T[:, :, None]                    # (n,T,5)
    pop = V.mean(axis=0)                                         # (n,)
    # hw: geo kNN(10) union, дистанции евклидовы (км-шкала генератора)
    g2 = ((geo[:, None, :] - geo[None, :, :]) ** 2).sum(axis=2)
    np.fill_diagonal(g2, np.inf)
    k = min(10, n - 1)
    part = np.argpartition(g2, kth=k - 1, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = part.ravel()
    lo = np.minimum(rows, cols).astype(np.int64)
    hi = np.maximum(rows, cols).astype(np.int64)
    key = np.unique(lo * n + hi)
    ei, ej = (key // n).astype(np.int32), (key % n).astype(np.int32)
    dist = np.sqrt(g2[ei, ej])
    return ms.MeasureData(tids=np.arange(n, dtype=np.int32), log_all=log_all,
                          sa_all=sa_all, shares=shares, clr=clr, vals=vals,
                          pop=pop, hw=(ei, ej, dist), region=None)


def synthetic_feature_matrix(ds: dict) -> np.ndarray:
    """X-адаптер для фича-методов на синтетике (K-means/ward/GMM/spectral/KEFRiN).
    10 признаков по мотивам X_static: CLR-профиль (6) + level_mean + рост
    (mean/sd рыночно-остаточных приростов объёма) + сезонная амплитуда (sd sa).
    StandardScaler. Росстат-контекста в синтетике нет по построению."""
    data = synthetic_measure_data(ds)
    clr_prof = data.clr.mean(axis=1)                             # (n,6)
    level = data.log_all.mean(axis=1, keepdims=True)             # (n,1)
    d = np.diff(data.sa_all, axis=1)
    d = d - d.mean(axis=0, keepdims=True)                        # рыночные остатки
    gmean = d.mean(axis=1, keepdims=True)
    gsd = d.std(axis=1, keepdims=True)
    samp = data.sa_all.std(axis=1, keepdims=True)
    X = np.hstack([clr_prof, level, gmean, gsd, samp])
    return StandardScaler().fit_transform(X)


# ---------------------------------------------------------------- эстиматоры

@dataclass
class LeidenEnsemble:
    runs: list[np.ndarray]          # len(seeds) одиночных прогонов
    consensus: np.ndarray
    pair_ari: np.ndarray            # попарные ARI (C(len,2),)
    gamma: float
    seeds: list[int] = field(default_factory=list)


def leiden_ensemble(A: sp.csr_matrix, gamma: float, seeds: list[int]) -> LeidenEnsemble:
    """seeds прогонов Leiden + CSPA-консенсус + попарные ARI (S-нога).

    Co-association строится из тех же одиночных прогонов (а не повторным
    прогоном внутри leiden_consensus_cspa): run_leiden детерминирован по
    (A, gamma, seed), результат бит-в-бит совпадает с
    leiden_consensus_cspa(A, gamma, seeds) — проверено на M2/M5 05.10.
    Экономия ~50% CPU ансамбля (31 прогон вместо 61)."""
    runs = [run_leiden(A, gamma=gamma, seed=s) for s in seeds]
    n = A.shape[0]
    C = np.zeros((n, n), dtype=np.float64)
    for lab in runs:
        onehot = np.zeros((n, lab.max() + 1))
        onehot[np.arange(n), lab] = 1.0
        C += onehot @ onehot.T
    C /= len(seeds)
    np.fill_diagonal(C, 0.0)
    cons = run_leiden(sp.csr_matrix(C), gamma=gamma, seed=seeds[0])
    pairs = np.array([ari(runs[i], runs[j])
                      for i, j in combinations(range(len(runs)), 2)])
    return LeidenEnsemble(runs=runs, consensus=cons, pair_ari=pairs,
                          gamma=gamma, seeds=list(seeds))


def _kmeans_seed_runs(X: np.ndarray, K: int, n_init: int,
                      seeds: list[int]) -> list[np.ndarray]:
    return [KMeans(n_clusters=K, n_init=n_init, random_state=s).fit_predict(X)
            for s in seeds]


def kmeans_protocol(X: np.ndarray, ks: list[int], n_init: int, seeds: list[int],
                    ari_min: float = 0.90) -> dict:
    """Выбор K (prereg §C): вето стабильности (медиана попарного ARI ≥ ari_min)
    → argmin S_Dbw внутри прошедших; никто не прошёл → argmax ARI + veto_failed.
    KMeans на X_static общий для всех мер (сетевые судьи различаются по мерам)."""
    rows, runs_by_k = [], {}
    for K in ks:
        labs = _kmeans_seed_runs(X, K, n_init, seeds)
        runs_by_k[K] = labs
        pairs = [ari(labs[i], labs[j]) for i, j in combinations(range(len(labs)), 2)]
        ref = labs[0]
        rows.append(dict(K=int(K),
                         ari_med=float(np.median(pairs)),
                         ari_min=float(np.min(pairs)),
                         sdbw=icvi.sdbw(X, ref)))
    table = pd.DataFrame(rows)
    passed = table[table.ari_med >= ari_min]
    veto_failed = len(passed) == 0
    if veto_failed:
        K = int(table.loc[table.ari_med.idxmax(), "K"])
        log.warning("K-means: ни один K не прошёл вето ARI≥%.2f — K=%d по argmax ARI "
                    "(помечено veto_failed)", ari_min, K)
    else:
        K = int(passed.loc[passed.sdbw.idxmin(), "K"])
    return {"K": K, "runs": runs_by_k[K], "labels": runs_by_k[K][0],
            "table": table, "veto_failed": veto_failed}


# ---------------------------------------------------------------- кандидаты

def scoring_candidates(table_a: pd.DataFrame, prereg_sim: dict
                       ) -> tuple[list[str], dict[str, str]]:
    """Гейт-фильтрация (prereg §A): скоринг = кандидаты с verdict=pass, минус
    анти-примеры (никогда), минус failed admissibility (в таблицу Б с verdict=fail).
    Возвращает (in_scoring, excluded{name: причина})."""
    anti = set(prereg_sim["anti_examples"])
    cand = list(prereg_sim["candidates"])
    verdict = dict(zip(table_a.measure, table_a.verdict))
    in_scoring = [m for m in cand if verdict.get(m) == "pass"]
    excluded = {m: "anti_example" for m in anti}
    excluded |= {m: "failed admissibility (bootstrap-J < гейт)"
                 for m in cand if verdict.get(m) != "pass"}
    return in_scoring, excluded


def oddeven_halves(T: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """odd/even половины (lvl_idx, grw_idx) — нога T по PREREG_DEVIATIONS №5.
    grw_idx — для мер, принимающих индексы приростов (нога T использует
    sliced_measure_data: мера пересчитывается на подряде целиком)."""
    return [(np.arange(0, T, 2), np.arange(0, T - 1, 2)),
            (np.arange(1, T, 2), np.arange(1, T - 1, 2))]


def sliced_measure_data(data: "ms.MeasureData", lvl_idx: np.ndarray) -> "ms.MeasureData":
    """Под-панель по позициям месяцев (odd/even, нога T) с КОРРЕКТНЫМИ
    календарными месяцами: lcal/gcal переопределяются под истинные месяцы
    удержанных позиций (у odd-половины это 0,2,4,..., а не 0..11).

    Зачем свой класс: measures._deseasoned_growths при grw_idx индексирует
    урезанную матрицу приростов ПОЛНЫМ gcal исходной панели → IndexError
    (латентная несовместимость, в 06a не упиралась: M9 там — M2/M5, без календаря).
    measures.py вне границ правок — обход здесь: мера вызывается на нарезанном
    объекте без idx-аргументов (внутренний diff по подряду, философия
    «пересчёт меры на половинном ряде», одинаковая для обеих половин)."""
    lvl_idx = np.asarray(lvl_idx)

    class _Sliced(ms.MeasureData):
        @property
        def lcal(self) -> np.ndarray:  # истинные календарные месяцы уровней
            return lvl_idx % 12

        @property
        def gcal(self) -> np.ndarray:  # месяцы целей внутренних приростов подряда
            return (lvl_idx[1:]) % 12

    return _Sliced(tids=data.tids, log_all=data.log_all[:, lvl_idx],
                   sa_all=data.sa_all[:, lvl_idx], shares=data.shares[:, lvl_idx],
                   clr=data.clr[:, lvl_idx], vals=data.vals[:, lvl_idx],
                   pop=data.pop, hw=data.hw, region=data.region)


# ---------------------------------------------------------------- композит

def leg_zscores(legs: pd.DataFrame) -> pd.DataFrame:
    """z-нормировка каждой ноги по кандидатам. legs: index=кандидат, columns=ноги."""
    return legs.apply(zscore)


def composite(legs_z: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    """Score = Σ w_leg · z(leg). Веса — ТОЛЬКО из prereg.yaml (не литералы)."""
    missing = [l for l in weights if l not in legs_z.columns]
    if missing:
        raise KeyError(f"в ногах нет колонок под веса prereg: {missing}")
    out = sum(w * legs_z[l] for l, w in weights.items())
    return out.rename("composite")


@dataclass
class BootstrapResult:
    samples: pd.DataFrame        # (B × кандидаты) композиты ресэмплов
    se_diff: pd.Series           # SE разности «кандидат − runner-up точечного»
    ci_lo: pd.Series
    ci_hi: pd.Series
    p_beats_runnerup: pd.Series  # P(кандидат > точечный runner-up) по ресэмплам


def paired_bootstrap(point_legs: pd.DataFrame,
                     rep_legs: dict[str, dict[str, np.ndarray]],
                     weights: dict[str, float], B: int, seed: int) -> BootstrapResult:
    """Парный бутстреп композита (prereg §A, 1000 ресэмплов).

    point_legs — точечные значения ног (index=кандидат). rep_legs —
    {нога: {кандидат: массив реплик}} для ног с естественной репликацией
    (S: seed-пары ARI; R: синтетические инстансы). Индексы ресэмпла общие
    для всех кандидатов внутри ноги (парность); ноги без реплик фиксированы.
    z-нормы пересчитываются на каждом ресэмпле."""
    rng = np.random.default_rng(seed)
    cands = list(point_legs.index)
    B_samples = np.empty((B, len(cands)))
    rep_idx = {leg: {m: np.asarray(v) for m, v in d.items()}
               for leg, d in rep_legs.items()}
    for b in range(B):
        legs_b = point_legs.copy()
        for leg, per_m in rep_idx.items():
            n_rep = len(next(iter(per_m.values())))
            idx = rng.integers(0, n_rep, n_rep)
            for m, arr in per_m.items():
                legs_b.loc[m, leg] = float(arr[idx].mean())
        comp_b = composite(leg_zscores(legs_b), weights)
        B_samples[b] = comp_b.loc[cands].to_numpy()
    samples = pd.DataFrame(B_samples, columns=cands)
    point_scores = composite(leg_zscores(point_legs), weights)
    order = point_scores.sort_values(ascending=False)
    runner = order.index[1] if len(order) > 1 else order.index[0]
    diff = samples.sub(samples[runner], axis=0)
    return BootstrapResult(
        samples=samples,
        se_diff=diff.std(ddof=1),
        ci_lo=samples.quantile(0.025),
        ci_hi=samples.quantile(0.975),
        p_beats_runnerup=(diff > 0).mean(),
    )


def margin_verdict(scores: pd.Series, se_diff: pd.Series, se_mult: float,
                   tiebreak_order: list[str]) -> dict:
    """Правило победителя (prereg §A): отрыв топ-1 от топ-2 ≥ se_mult·SE разности
    → победа; иначе ничья → оккамов тай-брейк по tiebreak_order среди топ-2."""
    order = scores.sort_values(ascending=False)
    w, r = order.index[0], order.index[1]
    margin = float(order.iloc[0] - order.iloc[1])
    se = float(se_diff[w])
    if margin >= se_mult * se:
        return {"winner": w, "runner_up": r, "margin": margin, "se": se,
                "verdict": "win"}
    final = next(m for m in tiebreak_order if m in (w, r))
    return {"winner": final, "runner_up": r if final == w else w,
            "margin": margin, "se": se, "verdict": "tie_tiebreak",
            "tiebreak_between": [w, r]}


def dirichlet_sensitivity(legs_z: pd.DataFrame, weights: dict[str, float],
                          concentration: float, draws: int, seed: int
                          ) -> pd.DataFrame:
    """Dirichlet(concentration·w) × draws → P(top-1), P(top-2) каждого кандидата
    (чувствительность композита к весам, prereg §A)."""
    legs = list(weights)
    w0 = np.array([weights[l] for l in legs], dtype=np.float64)
    rng = np.random.default_rng(seed)
    top1 = pd.Series(0, index=legs_z.index)
    top2 = pd.Series(0, index=legs_z.index)
    Z = legs_z[legs].to_numpy()
    for _ in range(draws):
        w = rng.dirichlet(concentration * w0)
        s = Z @ w
        rk = pd.Series(s, index=legs_z.index).rank(ascending=False)
        top1[rk.idxmin()] += 1
        top2[rk[rk <= 2].index] += 1
    return pd.DataFrame({"P_top1": top1 / draws, "P_top2": top2 / draws})


def lomo_winners(legs_z: pd.DataFrame, weights: dict[str, float]) -> dict:
    """Leave-one-leg-out ×5: победитель при удалении каждой ноги (веса
    перенормируются). Фотофиниш, если >1 удалений смещают топ-1 (prereg §A)."""
    full = composite(legs_z, weights).idxmax()
    per_drop, n_changed = {}, 0
    for leg in weights:
        w = {l: v for l, v in weights.items() if l != leg}
        tot = sum(w.values())
        w = {l: v / tot for l, v in w.items()}
        win = composite(legs_z, w).idxmax()
        per_drop[leg] = win
        n_changed += win != full
    return {"full_winner": full, "per_drop": per_drop,
            "n_changed": n_changed, "photo_finish": n_changed > 1}


def borda_table(legs_z: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    """table_C: ранг по каждой ноге + Borda (сумма рангов, меньше = лучше) +
    ранг композита. NaN-нога (вырождение, напр. M4 odd/even k<2) → NaN-ранг
    (nullable Int64), кандидат без полной ноги в Borda не участвует."""
    ranks = legs_z.rank(ascending=False).astype("Int64")
    ranks.columns = [f"rank_{c}" for c in ranks.columns]
    out = ranks.copy()
    out["borda"] = ranks.sum(axis=1, min_count=len(ranks.columns))
    out["borda_rank"] = out["borda"].rank().astype("Int64")
    out["composite"] = scores
    out["composite_rank"] = scores.rank(ascending=False).astype("Int64")
    return out.sort_values("composite_rank", na_position="last")


# ---------------------------------------------------------------- методы

def run_method(name: str, A: sp.csr_matrix, X: np.ndarray, K: int, seed: int,
               cfg: Config, gamma: float | None = None) -> np.ndarray | None:
    """Единая точка запуска метода на (A, X, K) — разделяется 06 и 06b.
    Возвращает None для честного skip (dmon без TF).
    gamma: переопределение Leiden-γ (на реальных данных — γ* из plateau 04;
    None → cfg.cluster.leiden.gamma — рабочая точка синтетики/динамики)."""
    from . import cluster as cl

    if name == "leiden_consensus":
        seeds = list(range(cfg.cluster.leiden.seeds_per_snapshot))
        return cl.leiden_consensus_cspa(A, gamma=gamma if gamma is not None
                                        else cfg.cluster.leiden.gamma, seeds=seeds)
    if name == "kmeans":
        return KMeans(n_clusters=K, n_init=cfg.cluster.kmeans.n_init,
                      random_state=seed).fit_predict(X)
    if name == "ward":
        from sklearn.cluster import AgglomerativeClustering
        return AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X)
    if name in ("gmm_full", "gmm_diag"):
        from sklearn.mixture import GaussianMixture
        return GaussianMixture(n_components=K,
                               covariance_type="full" if name == "gmm_full" else "diag",
                               n_init=3, reg_covar=1e-6,
                               random_state=seed).fit_predict(X)
    if name == "spectral_knn":
        from sklearn.cluster import SpectralClustering
        return SpectralClustering(n_clusters=K, affinity="nearest_neighbors",
                                  n_neighbors=15, random_state=seed).fit_predict(X)
    if name == "kefrin":
        return cl.run_kefrin(A, X, K=K, rho=cfg.cluster.kefrin.rho,
                             xi=cfg.cluster.kefrin.xi, seed=seed)
    if name == "louvain":
        return cl.run_louvain(A, seed=seed)
    if name == "infomap":
        return cl.run_infomap(A, seed=seed)
    if name == "eva":
        attrs = np.asarray(X[:, :6]).argmax(1)  # CLR-блок первым (как 04)
        return cl.run_eva(A, attrs, alpha=cfg.cluster.eva.alpha)
    if name == "dmon":
        return cl.run_dmon(A, X, K=K, seed=seed)
    raise ValueError(f"неизвестный метод: {name!r}")


def _subsample_inputs(name: str, A: sp.csr_matrix, X: np.ndarray,
                      idx: np.ndarray) -> tuple[sp.csr_matrix | None, np.ndarray | None]:
    """Входы метода на подвыборке узлов (node bootstrap 90%, prereg §B)."""
    feat_only = name in ("kmeans", "ward", "gmm_full", "gmm_diag", "spectral_knn")
    if feat_only:
        return None, X[idx]
    return A[idx][:, idx], (X[idx] if X is not None else None)


def method_stability(name: str, A: sp.csr_matrix, X: np.ndarray, K: int,
                     cfg: Config, *, n_boot: int = 30, frac: float = 0.9,
                     n_seeds: int = 30, seed: int = 42,
                     gamma: float | None = None) -> dict:
    """Стабильность метода (prereg §B): 0.5·z(ARI бутстреп 90% узлов ×30) +
    0.5·(−z(IQR попарного ARI по seed)). Детерминированные методы (ward, eva)
    дают seed-IQR=0 — честное свойство. Возвращает сырые числа (z — снаружи,
    по методам). gamma — рабочая точка Leiden (на реальных = γ* из 04)."""
    rng = np.random.default_rng(seed)
    n = A.shape[0] if A is not None else X.shape[0]
    base = run_method(name, A, X, K, seed, cfg, gamma=gamma)
    if base is None:
        return {"skipped": "method unavailable (dmon/TF)"}
    # seed-разброс (стохастические методы; у детерминированных IQR=0 честно —
    # повторные прогоны дословно идентичны, считать их — пустой расход CPU).
    # leiden_consensus детерминирован реализацией: run_method игнорирует seed
    # (внутренний фиксированный seed-лист CSPA) — повторы дословно идентичны.
    deterministic = name in ("ward", "eva", "leiden_consensus")
    if deterministic:
        iqr = 0.0
    else:
        seed_labs = [run_method(name, A, X, K, seed + 1000 + s, cfg, gamma=gamma)
                     for s in range(n_seeds)]
        seed_labs = [l for l in seed_labs if l is not None]
        pairs = np.array([ari(seed_labs[i], seed_labs[j])
                          for i, j in combinations(range(len(seed_labs)), 2)])
        iqr = float(np.percentile(pairs, 75) - np.percentile(pairs, 25)) if len(pairs) else 0.0
    # node bootstrap 90% ×30
    boots = []
    m = int(round(frac * n))
    for b in range(n_boot):
        idx = np.sort(rng.choice(n, m, replace=False))
        As, Xs = _subsample_inputs(name, A, X, idx)
        lab_b = run_method(name, As if As is not None else A, Xs if Xs is not None else X,
                           K, seed + 5000 + b, cfg, gamma=gamma)
        if lab_b is None:
            continue
        boots.append(ari(base[idx], lab_b))
    return {"boot_ari_mean": float(np.mean(boots)) if boots else np.nan,
            "seed_ari_iqr": iqr, "n_boot_ok": len(boots)}


def method_composite(legs: pd.DataFrame, prereg_method: dict,
                     interp: pd.Series | None = None) -> dict:
    """Композит методов (prereg §B). legs: index=метод, колонки
    [nmi_synth, boot_ari_mean, seed_ari_iqr, SW, CH_over_N, S_Dbw, MQ, timing_s].
    ICVI-подкомпозит — по icvi_subweights; AVI/AVU сюда НЕ входят (только отчёт).
    Интерпретируемость: interp=None → нога отсутствует; репортятся ОБЕ версии —
    с нулём (веса как в prereg) и перенормированная (веса /(1−w_interp)).
    Молчаливая перенормировка запрещена (задача владельца)."""
    w = {k: float(v) for k, v in prereg_method["composite_weights"].items()}
    sub = {k: float(v) for k, v in prereg_method["icvi_subweights"].items()}
    df = legs.copy()
    df["timing_inv"] = -np.log(df["timing_s"])
    # стабильность: 0.5·z(boot ARI) + 0.5·(−z(IQR seed)) — внутри ноги
    stab = 0.5 * zscore(df["boot_ari_mean"]) + 0.5 * (-zscore(df["seed_ari_iqr"]))
    df["stability"] = stab
    # ICVI-подкомпозит: подвесы уже нормированы на вес ноги (сумма = w_icvi)
    icvi_comp = (sub["SW"] * zscore(df["SW"]) + sub["CH_over_N"] * zscore(df["CH_over_N"])
                 + sub["neg_S_Dbw"] * (-zscore(df["S_Dbw"])) + sub["MQ"] * zscore(df["MQ"]))
    df["icvi"] = icvi_comp / w["icvi"]  # нога в шкале z (вес вернём в композите)
    if interp is not None:
        df["interpretability"] = interp
    else:
        df["interpretability"] = 0.0
    cols = {"nmi_synth": "nmi_synth", "stability": "stability", "icvi": "icvi",
            "interpretability": "interpretability", "timing_inv": "timing_inv"}
    legs_z = pd.DataFrame({k: zscore(df[v]) if k != "interpretability" or interp is not None
                           else df[v] for k, v in cols.items()})
    score_zero = composite(legs_z, w)
    out = {"composite_with_zero_interp": score_zero,
           "interp_present": interp is not None}
    if interp is None:
        wn = {k: v / (1.0 - w["interpretability"]) for k, v in w.items()
              if k != "interpretability"}
        out["composite_renormalized"] = composite(
            legs_z[[l for l in wn]], wn)
        out["note"] = ("интерпретируемость не заполнена "
                       "(configs/interpretability_rubric.yaml, TODO владельцу): "
                       "версия with_zero — веса prereg как есть (нога=0 у всех), "
                       "renormalized — остальные веса /0.85")
    return out


# ---------------------------------------------------------------- интерпретируемость

def load_interpretability(path: str | Path, methods: list[str]
                          ) -> tuple[pd.Series | None, str]:
    """configs/interpretability_rubric.yaml → средняя оценка двух оценщиков по
    методу. Рубрика не заполнена (пусто/NaN/не все методы/не оба оценщика) →
    (None, причина) — композит считается без ноги, обе версии репортятся."""
    import yaml

    p = Path(path)
    if not p.exists():
        return None, f"файл рубрики отсутствует: {p}"
    try:
        with open(p, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        # 06.10: configs/interpretability_rubric.yaml стр.39 `spectral_knn:{...}`
        # без пробела после двоеточия — невалидный YAML; конфиг вне границ правок
        # скорингового слоя → честная деградация «ноги нет», не падение прогона.
        return None, f"файл рубрики не парсится как YAML ({p.name}: {e}) — нога отсутствует"
    scores = raw.get("scores") or {}
    vals = {}
    for m in methods:
        cell = scores.get(m) or {}
        a, b = cell.get("rater_a"), cell.get("rater_b")
        if a is None or b is None:
            return None, f"рубрика не заполнена (метод {m}: rater_a={a}, rater_b={b})"
        vals[m] = 0.5 * (_rater_score(a) + _rater_score(b))
    return pd.Series(vals), "ok"


def _rater_score(v) -> float:
    """Оценка оценщика: число (среднее 1–5) или покритериальный словарь
    {C1..C5: int} — оба формата разрешены шапкой рубрики."""
    if isinstance(v, dict):
        return float(np.mean([float(x) for x in v.values()]))
    return float(v)


# ---------------------------------------------------------------- статистика пар

def wilcoxon_table(scores: pd.DataFrame, correction: str = "BH") -> pd.DataFrame:
    """Парные Wilcoxon signed-rank по общим репликам (prereg §B). scores:
    (реплики × методы); NaN-строки отбрасываются попарно (тяжёлые методы имеют
    меньше реплик — «общие реплики» пары). BH-поправка по семейству пар.
    Вырожденные пары (все разности 0) → p=1.0 с флагом; <5 пар наблюдений →
    p=NaN (Wilcoxon не определён), флаг."""
    methods = list(scores.columns)
    rows = []
    for a, b in combinations(methods, 2):
        d = (scores[a] - scores[b]).dropna()
        n_common = int(len(d))
        if n_common < 5:
            rows.append(dict(a=a, b=b, stat=np.nan, p=np.nan,
                             n_common=n_common, degenerate=False))
            continue
        if (d == 0).all():
            rows.append(dict(a=a, b=b, stat=0.0, p=1.0,
                             n_common=n_common, degenerate=True))
            continue
        stat, p = _scipy_wilcoxon(d)
        rows.append(dict(a=a, b=b, stat=float(stat), p=float(p),
                         n_common=n_common, degenerate=False))
    out = pd.DataFrame(rows)
    ok = out["p"].notna()
    out["p_adj"] = np.nan
    if ok.any():
        out.loc[ok, "p_adj"] = bh_adjust(out.loc[ok, "p"].to_numpy())
    return out


def bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg q=0.05-поправка (семейство пар внутри ячейки)."""
    p = np.asarray(pvals, dtype=np.float64)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out
