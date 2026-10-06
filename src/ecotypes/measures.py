"""Меры сходства M1–M11 + анти-примеры X1–X3 (research/24 «Решающий эксперимент»).

Контракт:
- Каждая мера → SimResult с dense-матрицей сходства (n,n) float64 (сходство, не
  расстояние: больше = ближе; у DTW/Aitchison — отрицание дистанции). Диагональ
  не несёт смысла (kNN её исключает).
- Единый графоформат: knn_union_graph(res, k) → per-node kNN + union, веса =
  сходство. Уравнение плотностей для всех кандидатов (24 §2).
- Гейт приемлемости: bootstrap_jaccard — 30 стратифицированных по календарному
  месяцу бутстрепов рядов, mean Jaccard kNN(10)-списков; порог 0.25 (prereg).

Зафиксированные решения/отступления:
- M8: d_eff = max(d, 1 км) — в edges_highway_knn есть d=0 (совпадающие центроиды),
  иначе гравитация делится на ноль. Веса нормируются на max (к масштабу [0,1]).
- M11': внешний пакет не установился — epfl-lts2/graph-learning не существует
  (404), rodrigo-pena/graph-learning (MIT) — не пакет (нет setup.py/pyproject,
  uv add отказал), per-node CVX-сolvers на n=2016 не масштабируются. Реализован
  свой градиентный спуск log-модели Kalofolias 2016 (степенной log-барьер +
  Frobenius): min_W Σ w_ij z_ij − α Σ_i log(deg_i) + β||W||_F², W≥0, симметрия.
  α=1, β=1 фиксированы до прогона (не тюнинговались). Запись в PREREG_DEVIATIONS
  — за пределами границ этого модуля, фиксируется в metrics.json этапа 06a.
- Бутстреп стратифицирован по календарному месяцу ПОЗИЦИИ (для каждой позиции t
  тянем замену из пула её календарного месяца) — время не перемешивается, DTW
  остаётся осмысленным.
- M9: α выбирается по odd/even-половинам (пререг secondary-вариант ноги T), а не
  по split-half 1–12/13–24: на T=24 с month-of-year десезонизацией (по 2 точки
  на календарный месяц) приросты второго года — ТОЧНАЯ противоположность первого
  (sa[t+12] ≡ −sa[t] ⇒ d[12+j] ≡ −d[j]), корреляции инвариантны к знаку →
  split-half Jaccard = 1.0 у любого α, критерий пуст. Odd/even режет по РАЗНЫМ
  календарным месяцам, тождества нет. Отклонение зафиксировано в metrics.json
  06a; то же тождество угрожает первичной T-ноге скоринга для всех sa-мер.
"""
from __future__ import annotations

import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from numba import njit, prange
from scipy.sparse.csgraph import connected_components
from scipy.stats import rankdata

CAT5 = ("prod", "health", "market", "food", "transp")
PARTS6 = (*CAT5, "proch")
M8_DIST_FLOOR_KM = 1.0   # пол дистанции в гравитации (в данных есть d=0)
M8_GAMMA = 1.5           # фикс, середина литературного 1–2 (24 §1.2) — не тюнингуется
M11_ALPHA = 1.0          # степенной log-барьер (Kalofolias 2016), фикс до прогона
M11_BETA = 1.0           # Frobenius-регуляризатор, фикс до прогона
M11_ITERS = 400
SNF_T = 20               # фикс по пререгу (24 §1.2)
DTW_WINDOW = 3           # Сакое–Тиба; лаг >3 при T=24 не обоснован (24 §4.5)


# --- данные --------------------------------------------------------------------

@dataclass
class MeasureData:
    """Всё, что нужно мерам. Ряды (n, T), узлы в отсортированном порядке tids."""
    tids: np.ndarray          # (n,) int32
    log_all: np.ndarray       # (n, T) f64 — уровни log-суммы
    sa_all: np.ndarray        # (n, T) f64 — десезонированные уровни (month-of-year)
    shares: np.ndarray        # (n, T, 6) f64
    clr: np.ndarray           # (n, T, 6) f64
    vals: np.ndarray          # (n, T, 5) f64 — категориальные значения (для M3)
    pop: np.ndarray | None = None      # (n,) f64 — население (M8)
    hw: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None  # (ei, ej, dist) по строкам
    region: np.ndarray | None = None   # (n,) — код субъекта РФ (топология)

    @property
    def n(self) -> int:
        return len(self.tids)

    @property
    def T(self) -> int:
        return self.log_all.shape[1]

    @property
    def lcal(self) -> np.ndarray:
        """Календарный месяц каждой позиции уровня: (T,)."""
        return np.arange(self.T) % 12

    @property
    def gcal(self) -> np.ndarray:
        """Календарный месяц цели каждого прироста: (T−1,); прирост t — в месяц t+1."""
        return np.arange(1, self.T) % 12

    def slice_months(self, sl: slice) -> "MeasureData":
        """Под-панель по месяцам. ВНИМАНИЕ: половины 0:12 и 12:24 при T=24 с
        month-of-year sa дезинформативны для sa-мер (d[12+j] ≡ −d[j], см. шапку)."""
        return MeasureData(
            tids=self.tids, log_all=self.log_all[:, sl], sa_all=self.sa_all[:, sl],
            shares=self.shares[:, sl], clr=self.clr[:, sl], vals=self.vals[:, sl],
            pop=self.pop, hw=self.hw, region=self.region)


def sa_from_log(log_all: np.ndarray) -> np.ndarray:
    """Month-of-year десезонизация уровней в логах (как panel.py: sa = log − mean
    того же календарного месяца по узлу). Та же формула — для синтетики/тестов."""
    X = np.asarray(log_all, dtype=np.float64)
    T = X.shape[1]
    out = X.copy()
    for m in range(12):
        cols = np.flatnonzero(np.arange(T) % 12 == m)
        out[:, cols] = X[:, cols] - X[:, cols].mean(axis=1, keepdims=True)
    return out


def clr_transform(shares: np.ndarray, clip: float = 1e-9) -> np.ndarray:
    """CLR по последней оси."""
    lp = np.log(np.clip(shares, clip, None))
    return lp - lp.mean(axis=-1, keepdims=True)


def from_processed(processed_dir: str | Path) -> MeasureData:
    """Загрузка панели/узлов/highway-рёбер из data/processed."""
    processed_dir = Path(processed_dir)
    panel = pd.read_parquet(processed_dir / "panel_monthly.parquet")
    nodes = pd.read_parquet(processed_dir / "nodes_static.parquet")
    edges = pd.read_parquet(processed_dir / "edges_highway_knn.parquet")

    tids = np.sort(panel.territory_id.unique()).astype(np.int32)
    months = np.sort(panel.month_idx.unique())

    def _piv(col: str) -> np.ndarray:
        p = panel.pivot(index="territory_id", columns="month_idx", values=col)
        assert list(p.columns) == list(months)
        return p.loc[tids].to_numpy(np.float64)

    log_all = _piv("log_all")
    sa_all = _piv("sa_all")
    shares = np.stack([_piv(f"share_{s}") for s in PARTS6], axis=2)
    clr = np.stack([_piv(f"clr_{s}") for s in PARTS6], axis=2)
    vals = np.stack([_piv(f"val_{s}") for s in CAT5], axis=2)

    nd = nodes.set_index("territory_id").loc[tids]
    pop = nd["pop_2023"].to_numpy(np.float64)
    region = nd["region_code"].to_numpy()

    tid2row = pd.Series(np.arange(len(tids)), index=tids)
    ei = tid2row.loc[edges.tid_x.to_numpy()].to_numpy(np.int32)
    ej = tid2row.loc[edges.tid_y.to_numpy()].to_numpy(np.int32)
    dist = edges.dist_km.to_numpy(np.float64)

    return MeasureData(tids=tids, log_all=log_all, sa_all=sa_all, shares=shares,
                       clr=clr, vals=vals, pop=pop, hw=(ei, ej, dist), region=region)


# --- базовые ряды ----------------------------------------------------------------

def residual_returns(data: MeasureData, grw_idx: np.ndarray | None = None) -> np.ndarray:
    """(n, T−1): приросты десезонированных уровней − кросс-секционное среднее
    по месяцу (рыночные остатки, база M2)."""
    d = np.diff(data.sa_all, axis=1)
    if grw_idx is not None:
        d = d[:, grw_idx]
    return d - d.mean(axis=0, keepdims=True)


def _deseasoned_growths(data: MeasureData, grw_idx: np.ndarray | None = None) -> np.ndarray:
    """(n, T−1): сырые log-приросты минус pooled-среднее календарного месяца
    (месячные dummy на приростах, база M1). Рыночный фактор НЕ убирается."""
    lr = np.diff(data.log_all, axis=1)
    if grw_idx is not None:
        lr = lr[:, grw_idx]
    out = lr.copy()
    for m in range(12):
        cols = np.flatnonzero(data.gcal == m)
        out[:, cols] -= lr[:, cols].mean()
    return out


def category_residuals(data: MeasureData, grw_idx: np.ndarray | None = None) -> np.ndarray:
    """(n, T−1, 5): M2-остатки по каждой из 5 чистых категорий (sa per узел →
    diff → кросс-секционный демеанинг)."""
    lv = np.log(np.maximum(data.vals, 1.0))  # value≥10 в данных; пол — страховка
    sa = np.empty_like(lv)
    for m in range(12):
        cols = np.flatnonzero(data.lcal == m)
        sa[:, cols, :] = lv[:, cols, :] - lv[:, cols, :].mean(axis=1, keepdims=True)
    d = np.diff(sa, axis=1)
    if grw_idx is not None:
        d = d[:, grw_idx]
    return d - d.mean(axis=0, keepdims=True)


def _corr(X: np.ndarray) -> np.ndarray:
    """Корреляционная матрица строк X (n, G); константные ряды → 0 вне диагонали."""
    Z = X - X.mean(axis=1, keepdims=True)
    sd = Z.std(axis=1, keepdims=True)
    Z = np.where(sd < 1e-12, 0.0, Z / np.where(sd < 1e-12, 1.0, sd))
    R = Z @ Z.T / X.shape[1]
    np.fill_diagonal(R, 1.0)
    return R


def _corr2(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Кросс-корреляции строк A (n, G) и строк B (n, G): R[i,j] = corr(A_i, B_j)."""

    def _z(X: np.ndarray) -> np.ndarray:
        Z = X - X.mean(axis=1, keepdims=True)
        sd = Z.std(axis=1, keepdims=True)
        return np.where(sd < 1e-12, 0.0, Z / np.where(sd < 1e-12, 1.0, sd))

    return _z(A) @ _z(B).T / A.shape[1]


# --- результат меры ----------------------------------------------------------------

@dataclass
class SimResult:
    """matrix: (n,n) сходство (диагональ игнорируется). edges: (ei, ej, w)
    уникальные пары i<j — для статистик edge-list мер (M8). extra: служебное."""
    matrix: np.ndarray
    edges: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    extra: dict = field(default_factory=dict)


# --- анти-примеры ------------------------------------------------------------------

def sim_X1(data: MeasureData, lvl_idx=None, grw_idx=None) -> SimResult:
    """Пирсон уровней log_all (вырождено: mean r ≈ 0.89, 24 §0)."""
    X = data.log_all if lvl_idx is None else data.log_all[:, lvl_idx]
    return SimResult(_corr(X))


def sim_X2(data: MeasureData, lvl_idx=None, grw_idx=None) -> SimResult:
    """Пирсон сырых log-приростов (вырождено общим фактором, 24 §0)."""
    d = np.diff(data.log_all, axis=1)
    if grw_idx is not None:
        d = d[:, grw_idx]
    return SimResult(_corr(d))


def sim_X3(data: MeasureData, lvl_idx=None, grw_idx=None) -> SimResult:
    """Косинус средних сырых долей (6 компонентов; вырождено, 24 §0)."""
    sh = data.shares if lvl_idx is None else data.shares[:, lvl_idx]
    p = sh.mean(axis=1)
    U = p / np.linalg.norm(p, axis=1, keepdims=True)
    S = U @ U.T
    np.fill_diagonal(S, 1.0)
    return SimResult(S)


# --- M1–M4: динамика уровня ---------------------------------------------------------

def sim_M1(data: MeasureData, lvl_idx=None, grw_idx=None) -> SimResult:
    """Пирсон приростов с календарной десезонизацией (месячные dummy)."""
    return SimResult(_corr(_deseasoned_growths(data, grw_idx)))


def sim_M2(data: MeasureData, lvl_idx=None, grw_idx=None) -> SimResult:
    """Пирсон рыночных остатков приростов (десезон + кросс-секционный демеанинг)."""
    return SimResult(_corr(residual_returns(data, grw_idx)))


def sim_M3(data: MeasureData, lvl_idx=None, grw_idx=None) -> SimResult:
    """Мультиканал: Fisher-z среднее корреляций M2-остатков по 5 категориям."""
    res = category_residuals(data, grw_idx)
    zs = [np.arctanh(np.clip(_corr(res[:, :, c]), -0.999999, 0.999999))
          for c in range(res.shape[2])]
    S = np.tanh(np.mean(zs, axis=0))
    np.fill_diagonal(S, 1.0)
    return SimResult(S)


def sim_M4(data: MeasureData, lvl_idx=None, grw_idx=None, max_lag: int = 3) -> SimResult:
    """max |r| по лагам −3..+3 на M2-рядах. extra["lead_lag"][i,j] = ℓ означает:
    j ЛИДИРУЕТ i на ℓ мес (ℓ>0: максимум достигнут при corr(res_i[t], res_j[t−ℓ]);
    ℓ<0 — i лидирует j). Раньше знак был инвертирован — исправлено."""
    R0 = residual_returns(data, grw_idx)
    n = R0.shape[0]
    best = np.abs(_corr(R0))
    lag_map = np.zeros((n, n), dtype=np.int8)
    for lag in range(1, max_lag + 1):
        # Rl[i,j] = |corr(res_i[t], res_j[t+lag])|: будущее j против настоящего i
        # → высокое значение означает, что i ЛИДИРУЕТ j на lag → j лидирует на −lag.
        Rl = np.abs(_corr2(R0[:, :-lag], R0[:, lag:]))
        for M, sgn in ((Rl, -lag), (Rl.T, lag)):
            upd = M > best
            best[upd] = M[upd]
            lag_map[upd] = sgn
    return SimResult(best, extra={"lead_lag": lag_map})


# --- M5: композиция -----------------------------------------------------------------

def clr_profiles(data: MeasureData, lvl_idx=None) -> np.ndarray:
    """(n, 6): CLR-средние профили долей (среднее CLR по месяцам)."""
    C = data.clr if lvl_idx is None else data.clr[:, lvl_idx]
    return C.mean(axis=1)


def sim_M5(data: MeasureData, lvl_idx=None, grw_idx=None) -> SimResult:
    """Aitchison-близость: −евклидово расстояние на CLR-средних профилях."""
    P = clr_profiles(data, lvl_idx)
    G = (P * P).sum(axis=1)
    d2 = G[:, None] + G[None, :] - 2.0 * (P @ P.T)
    np.clip(d2, 0.0, None, out=d2)
    return SimResult(-np.sqrt(d2))


# --- M6/M7: DTW (numba) ---------------------------------------------------------------

@njit(cache=True, fastmath=True)
def _dtw_1d(a: np.ndarray, b: np.ndarray, w: int) -> float:
    """DTW с окном Сакое–Тибы |t−s| ≤ w, локальная стоимость — квадрат разности."""
    T = a.shape[0]
    D = np.full((T + 1, T + 1), np.inf)
    D[0, 0] = 0.0
    for t in range(1, T + 1):
        lo = t - w if t - w > 1 else 1
        hi = t + w if t + w < T else T
        for s in range(lo, hi + 1):
            c = (a[t - 1] - b[s - 1]) ** 2
            m = D[t - 1, s]
            if D[t, s - 1] < m:
                m = D[t, s - 1]
            if D[t - 1, s - 1] < m:
                m = D[t - 1, s - 1]
            D[t, s] = c + m
    return D[T, T]


@njit(cache=True, fastmath=True)
def _dtwd_1d(a: np.ndarray, b: np.ndarray, w: int) -> float:
    """DTW-D (синхронный warp): локальная стоимость — квадрат евклидова по каналам."""
    T = a.shape[0]
    C = a.shape[1]
    D = np.full((T + 1, T + 1), np.inf)
    D[0, 0] = 0.0
    for t in range(1, T + 1):
        lo = t - w if t - w > 1 else 1
        hi = t + w if t + w < T else T
        for s in range(lo, hi + 1):
            c = 0.0
            for ch in range(C):
                c += (a[t - 1, ch] - b[s - 1, ch]) ** 2
            m = D[t - 1, s]
            if D[t, s - 1] < m:
                m = D[t, s - 1]
            if D[t - 1, s - 1] < m:
                m = D[t - 1, s - 1]
            D[t, s] = c + m
    return D[T, T]


@njit(parallel=True, cache=True, fastmath=True)
def dtw_matrix(X: np.ndarray, w: int) -> np.ndarray:
    """(n,n) матрица DTW-дистанций между строками X (n, T) float64."""
    X = np.ascontiguousarray(X)
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in prange(n - 1):
        for j in range(i + 1, n):
            d = _dtw_1d(X[i], X[j], w)
            D[i, j] = d
            D[j, i] = d
    return D


@njit(parallel=True, cache=True, fastmath=True)
def dtwd_matrix(X: np.ndarray, w: int) -> np.ndarray:
    """(n,n) матрица DTW-D дистанций между траекториями X (n, T, C) float64."""
    X = np.ascontiguousarray(X)
    n = X.shape[0]
    D = np.zeros((n, n))
    for i in prange(n - 1):
        for j in range(i + 1, n):
            d = _dtwd_1d(X[i], X[j], w)
            D[i, j] = d
            D[j, i] = d
    return D


def _znorm(X: np.ndarray) -> np.ndarray:
    Z = X - X.mean(axis=1, keepdims=True)
    sd = Z.std(axis=1, keepdims=True)
    return Z / np.where(sd < 1e-12, 1.0, sd)


def sim_M6(data: MeasureData, lvl_idx=None, grw_idx=None,
           window: int = DTW_WINDOW) -> SimResult:
    """DTW уровней: z-нормированные ряды log_all, окно Сакое–Тибы w=3."""
    X = data.log_all if lvl_idx is None else data.log_all[:, lvl_idx]
    D = dtw_matrix(_znorm(np.asarray(X, dtype=np.float64)), window)
    return SimResult(-D)


def sim_M7(data: MeasureData, lvl_idx=None, grw_idx=None,
           window: int = DTW_WINDOW) -> SimResult:
    """DTW-D на CLR-траекториях 5 чистых категорий (proch — линейно зависим),
    синхронный warp, w=3. CLR без масштаба — z-нормировка не нужна."""
    C = data.clr[:, :, :5] if lvl_idx is None else data.clr[:, lvl_idx, :5]
    D = dtwd_matrix(np.asarray(C, dtype=np.float64), window)
    return SimResult(-D)


# --- M8: априорная структура -----------------------------------------------------------

def sim_M8(data: MeasureData, lvl_idx=None, grw_idx=None,
           gamma: float = M8_GAMMA) -> SimResult:
    """Дорожная гравитация: w = pop_i·pop_j / max(d, 1 км)^1.5 на highway kNN(10)
    (островные fallback-рёбра уже в edges). Нормировка на max. Статична —
    бутстреп рядов её не меняет (24 §4.6: честное свойство, не баг)."""
    assert data.hw is not None and data.pop is not None, "M8 нужны pop и hw-рёбра"
    ei, ej, dist = data.hw
    n = data.n
    lo = np.minimum(ei, ej).astype(np.int64)
    hi = np.maximum(ei, ej).astype(np.int64)
    key = lo * n + hi
    uk, inv = np.unique(key, return_inverse=True)
    dmin = np.full(len(uk), np.inf)
    np.minimum.at(dmin, inv, dist)
    ui = (uk // n).astype(np.int32)
    uj = (uk % n).astype(np.int32)
    d_eff = np.maximum(dmin, M8_DIST_FLOOR_KM)
    w = data.pop[ui] * data.pop[uj] / d_eff**gamma
    w = w / w.max()
    W = np.zeros((n, n), dtype=np.float64)
    W[ui, uj] = w
    W[uj, ui] = w
    return SimResult(W, edges=(ui, uj, w))


# --- общий графоформат -----------------------------------------------------------------

def knn_lists(sim, k: int) -> np.ndarray:
    """(n, k) индексы топ-k соседей по строке (без self), по убыванию сходства.
    Детерминировано: argpartition + stable sort. Принимает SimResult или матрицу."""
    W = sim.matrix if isinstance(sim, SimResult) else np.asarray(sim)
    W = W.copy()
    n = W.shape[0]
    k = min(k, n - 1)
    np.fill_diagonal(W, -np.inf)
    part = np.argpartition(-W, kth=k - 1, axis=1)[:, :k]
    w = np.take_along_axis(W, part, axis=1)
    order = np.argsort(-w, axis=1, kind="stable")
    return np.take_along_axis(part, order, axis=1)


def knn_union_graph(sim, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Единый формат: per-node kNN(k) + union. Возвращает edge_index (E, 2) int32
    (i < j, отсортировано) и weights (E,) float32 = сходство меры."""
    res = sim if isinstance(sim, SimResult) else SimResult(np.asarray(sim))
    W = res.matrix
    n = W.shape[0]
    lists = knn_lists(res, k)
    kk = lists.shape[1]
    src = np.repeat(np.arange(n), kk)
    dst = lists.ravel()
    lo = np.minimum(src, dst).astype(np.int64)
    hi = np.maximum(src, dst).astype(np.int64)
    key = np.unique(lo * n + hi)
    lo, hi = key // n, key % n
    w = W[lo, hi].astype(np.float32)
    return np.column_stack([lo, hi]).astype(np.int32), w


def edge_set_jaccard(e1: np.ndarray, e2: np.ndarray, n: int) -> float:
    """Jaccard двух edge set'ов (edge_index E×2, i<j)."""
    k1 = e1[:, 0].astype(np.int64) * n + e1[:, 1].astype(np.int64)
    k2 = e2[:, 0].astype(np.int64) * n + e2[:, 1].astype(np.int64)
    inter = np.intersect1d(k1, k2, assume_unique=True).size
    union = k1.size + k2.size - inter
    return float(inter / union) if union else 0.0


def jaccard_lists(A: np.ndarray, B: np.ndarray) -> float:
    """Mean Jaccard per-node kNN-списков (n,k) × (n,k)."""
    inter = (A[:, :, None] == B[:, None, :]).any(axis=2).sum(axis=1)
    union = A.shape[1] + B.shape[1] - inter
    return float((inter / union).mean())


# --- гейт приемлемости -------------------------------------------------------------------

def stratified_resample(T: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Стратифицированный по календарному месяцу бутстреп: для каждой позиции t —
    замена из пула того же календарного месяца. Возвращает (lvl_idx (T,),
    grw_idx (T−1,)) — индексы колонок уровней и приростов."""
    lcal = np.arange(T) % 12
    lvl = np.empty(T, dtype=np.int64)
    for t in range(T):
        pool = np.flatnonzero(lcal == lcal[t])
        lvl[t] = pool[rng.integers(len(pool))]
    gcal = np.arange(1, T) % 12
    grw = np.empty(T - 1, dtype=np.int64)
    for t in range(T - 1):
        pool = np.flatnonzero(gcal == gcal[t])
        grw[t] = pool[rng.integers(len(pool))]
    return lvl, grw


def bootstrap_jaccard(fn, data: MeasureData, k: int, B: int, seed: int
                      ) -> tuple[float, float, np.ndarray]:
    """Гейт: mean/sd Jaccard kNN(k)-списков по B стратифицированным бутстрепам.
    fn(data, lvl_idx=..., grw_idx=...) → SimResult. Детерминировано по seed."""
    base = knn_lists(fn(data), k)
    rng = np.random.default_rng(seed)
    js = np.empty(B)
    for b in range(B):
        lvl, grw = stratified_resample(data.T, rng)
        js[b] = jaccard_lists(base, knn_lists(fn(data, lvl_idx=lvl, grw_idx=grw), k))
    return float(js.mean()), float(js.std(ddof=1)) if B > 1 else 0.0, js


def gate_verdict(j_mean: float, threshold: float) -> str:
    return "pass" if j_mean >= threshold else "fail"


def measure_seed(seed: int, name: str) -> int:
    """Детерминированный per-measure seed гейта."""
    return seed + zlib.crc32(name.encode()) % (2**31)


# --- M9: выпуклая смесь ------------------------------------------------------------------

def percentile_sim(S: np.ndarray, positive_only: bool = False) -> np.ndarray:
    """Ранг-нормализация сходства в (0,1]: перцентиль по верхнему треугольнику.
    positive_only — для edge-мер (M8): ранжируются только ненулевые пары."""
    n = S.shape[0]
    iu = np.triu_indices(n, 1)
    v = np.asarray(S[iu], dtype=np.float64)
    if positive_only:
        mask = v > 0
        r = np.zeros_like(v)
        r[mask] = rankdata(v[mask]) / mask.sum()
    else:
        r = rankdata(v) / v.size
    P = np.zeros((n, n), dtype=np.float64)
    P[iu] = r
    P[(iu[1], iu[0])] = r
    np.fill_diagonal(P, 1.0)
    return P


def sim_M9(data: MeasureData, lvl_idx=None, grw_idx=None, alpha: float = 0.5) -> SimResult:
    """α·M5 + (1−α)·M2 на перцентиль-нормализованных сходствах."""
    p2 = percentile_sim(sim_M2(data, grw_idx=grw_idx).matrix)
    p5 = percentile_sim(sim_M5(data, lvl_idx=lvl_idx).matrix)
    return SimResult(alpha * p5 + (1.0 - alpha) * p2)


def m9_alpha_split_half(data: MeasureData, grid: list[float], k: int
                        ) -> tuple[float, dict[float, float]]:
    """Выбор α по ноге T (НЕ по композиту): odd/even месяцы (пререг secondary),
    мера пересчитывается на половинах позиций, Jaccard kNN-списков половин;
    max → α. Ничьи разрешаются в пользу меньшего α (оккам).
    Первичный split-half 1–12/13–24 непригоден для M2-компоненты: при T=24 и
    month-of-year десезонизации d[12+j] ≡ −d[j] → corr-матрицы половин совпадают
    точно, J=1.0 у любого α (см. шапку модуля)."""
    T = data.T
    halves = [(np.arange(0, T, 2), np.arange(0, T - 1, 2)),
              (np.arange(1, T, 2), np.arange(1, T - 1, 2))]
    pcts = [(percentile_sim(sim_M2(data, grw_idx=g).matrix),
             percentile_sim(sim_M5(data, lvl_idx=l).matrix))
            for l, g in halves]
    js: dict[float, float] = {}
    for a in grid:
        lists = [knn_lists(a * p5 + (1.0 - a) * p2, k) for p2, p5 in pcts]
        js[float(a)] = jaccard_lists(lists[0], lists[1])
    best = max(grid, key=lambda a: (js[float(a)], -a))
    return float(best), js


# --- M10: SNF ------------------------------------------------------------------------------

def _snf_norm(W: np.ndarray) -> np.ndarray:
    """SNF-нормировка (Wang 2014): P_ii = 1/2, P_ij = w_ij / (2 Σ_{k≠i} w_ik)."""
    W = np.array(W, dtype=np.float32, copy=True)
    np.fill_diagonal(W, 0.0)
    rs = W.sum(axis=1, keepdims=True)
    P = W / np.where(rs > 1e-12, 2.0 * rs, 1.0)
    np.fill_diagonal(P, 0.5)
    return P


def _snf_kernel(P: np.ndarray, k: int) -> sp.csr_matrix:
    """kNN-ядро S: строки нормированы по сумме k ближайших (без self)."""
    n = P.shape[0]
    k = min(k, n - 1)
    W = P.copy()
    np.fill_diagonal(W, 0.0)
    part = np.argpartition(-W, kth=k - 1, axis=1)[:, :k]
    w = np.take_along_axis(W, part, axis=1)
    rows = np.repeat(np.arange(n), k)
    vals = w.ravel()
    rsum = vals.reshape(n, k).sum(axis=1)
    vals = vals / np.where(np.repeat(rsum, k) < 1e-12, 1.0, np.repeat(rsum, k))
    return sp.csr_matrix((vals, (rows, part.ravel())), shape=(n, n))


def snf(mats: list[np.ndarray], k: int = 10, t: int = SNF_T
        ) -> tuple[np.ndarray, list[float]]:
    """Similarity Network Fusion (Wang 2014): P_m ← S_m · (среднее остальных) · S_mᵀ,
    t итераций. Возвращает (fused, errs) — errs = max|Δfused| по итерациям
    (для проверки сходимости)."""
    M = len(mats)
    Ps = [_snf_norm(m) for m in mats]
    Ss = [_snf_kernel(P, k) for P in Ps]
    fused = sum(Ps) / M
    errs: list[float] = []
    for _ in range(t):
        tot = sum(Ps)
        new = []
        for m in range(M):
            others = (tot - Ps[m]) / (M - 1) if M > 1 else Ps[m]
            P = Ss[m] @ others @ Ss[m].T
            P = (P + P.T) / 2.0
            new.append(_snf_norm(P))
        Ps = new
        prev, fused = fused, sum(Ps) / M
        errs.append(float(np.abs(fused - prev).max()))
    # _snf_norm не сохраняет симметрию (строковые суммы разные) — симметризуем
    # финальное слияние; это similarity-матрица, не стохастическая.
    fused = (fused + fused.T) / 2.0
    np.fill_diagonal(fused, 0.0)
    return fused, errs


def sim_M10(data: MeasureData, lvl_idx=None, grw_idx=None, k: int = 10,
            t: int = SNF_T) -> SimResult:
    """SNF-слияние {M2, M5, M8} — по представителю от семейства (24 §1.2);
    входы перцентиль-нормализованы, k=10, t=20 (фикс)."""
    p2 = percentile_sim(sim_M2(data, grw_idx=grw_idx).matrix)
    p5 = percentile_sim(sim_M5(data, lvl_idx=lvl_idx).matrix)
    p8 = percentile_sim(sim_M8(data).matrix, positive_only=True)
    W, errs = snf([p2, p5, p8], k=k, t=t)
    return SimResult(W, extra={"snf_err": errs})


# --- M11': learned graph (своя реализация Kalofolias 2016 log-модели) ----------------------

def m11_log_model(Z2: np.ndarray, alpha: float = M11_ALPHA, beta: float = M11_BETA,
                  iters: int = M11_ITERS, tol: float = 1e-7
                  ) -> tuple[np.ndarray, float]:
    """min_W Σ_ij w_ij z_ij − α Σ_i log(deg_i) + β||W||_F²,  W ≥ 0, симметричный,
    нулевая диагональ. Проекционный градиентный спуск, полная матрица.
    z_ij = ||x_i − x_j||² (отмасштабирован к mean 1 для устойчивости шага)."""
    n = Z2.shape[0]
    iu = np.triu_indices(n, 1)
    scale = Z2[iu].mean()
    Z = np.asarray(Z2 / max(scale, 1e-12), dtype=np.float64)
    W = alpha / (Z + 0.1)  # старт — точное решение без Frobenius-члена
    np.fill_diagonal(W, 0.0)
    step = 0.25 / (1.0 + 2.0 * beta)
    prev = W.copy()
    for it in range(iters):
        deg = W.sum(axis=1) + 1e-12
        inv = alpha / deg
        G = Z - (inv[:, None] + inv[None, :]) + 2.0 * beta * W
        W = W - step * G
        np.maximum(W, 0.0, out=W)
        np.fill_diagonal(W, 0.0)
        if it % 25 == 24:
            rel = float(np.abs(W - prev).max() / (np.abs(prev).max() + 1e-12))
            if rel < tol:
                break
            prev = W.copy()
    deg = W.sum(axis=1) + 1e-12
    obj = float((W * Z).sum() - alpha * np.log(deg).sum() + beta * (W**2).sum())
    return W, obj


def sim_M11(data: MeasureData, lvl_idx=None, grw_idx=None,
            alpha: float = M11_ALPHA, beta: float = M11_BETA,
            iters: int = M11_ITERS) -> SimResult:
    """Learned graph на z-нормированных рядах log_all (M11': своя реализация
    log-модели Kalofolias 2016 — см. шапку модуля)."""
    X = data.log_all if lvl_idx is None else data.log_all[:, lvl_idx]
    X = _znorm(np.asarray(X, dtype=np.float64))
    G = (X * X).sum(axis=1)
    Z2 = G[:, None] + G[None, :] - 2.0 * (X @ X.T)
    np.clip(Z2, 0.0, None, out=Z2)
    np.fill_diagonal(Z2, 0.0)
    W, obj = m11_log_model(Z2, alpha=alpha, beta=beta, iters=iters)
    return SimResult(W, extra={"m11_obj": obj})


# --- топология (таблицы этапа 06a) ----------------------------------------------------------

def topology_stats(edge_index: np.ndarray, n: int,
                   region: np.ndarray | None = None) -> dict:
    """Рёбра, степени, изоляты, % гиганта, доля рёбер внутри субъекта РФ."""
    E = len(edge_index)
    A = sp.csr_matrix((np.ones(E), (edge_index[:, 0], edge_index[:, 1])),
                      shape=(n, n))
    A = A + A.T
    deg = np.diff(A.indptr)
    n_comp, labels = connected_components(A, directed=False)
    giant = float(np.bincount(labels).max() / n)
    out = {
        "edges": int(E),
        "deg_mean": float(deg.mean()),
        "deg_min": int(deg.min()),
        "deg_med": float(np.median(deg)),
        "deg_max": int(deg.max()),
        "isolates": int((deg == 0).sum()),
        "giant_pct": giant,
    }
    if region is not None:
        same = region[edge_index[:, 0]] == region[edge_index[:, 1]]
        out["within_region_share"] = float(same.mean())
    return out
