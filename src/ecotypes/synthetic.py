"""Синтетическая валидация (спека research/28_synthetic_validation.md, 04.10.2026).

Два генератора с известной истиной:
- PP-Dir (§2): K типов → прототипы μ_k на 6-компонентном симплексе → базы узлов
  → месячные наблюдения Дирихле + двухслойный граф (SBM-ядро ∪ атрибутно-гео шум);
  доля δ узлов меняет тип в случайный месяц.
- LFR+Dir (§3): структурный эталон networkx LFR, атрибуты сшиваются той же
  Дирихле-цепочкой (коммьюнити → прототип → базы → месяцы).

Калибровочные константы измерены 04.10.2026 по consumption.parquet
(2190 МО × 24 мес) и зашиты литералами; см. комментарии у констант.

Зафиксированные при сборке отступления/уточнения спеки (все остальное — дословно):
1. Прототипы после Dir(α_proto·p̄) доводятся IPF-рейкингом до взвешенного
   арифметического среднего = p̄. Без этого acceptance-чек §6.2 (|Δсредних|<0.02)
   невыполним при малых K: sd среднего K=4 Dir(2·p̄)-прототипов ≈ 0.13 на «Прочее».
   Рейкинг не меняет порядок отбраковки (min CLR проверяется после рейкинга).
2. Центры месячных наблюдений обрезаны снизу на MONTH_FLOOR=0.01 (ре-нормировка).
   Основание — реализм, не подгонка: минимальная наблюдаемая доля компонента
   в средних профилях МО ≈ 0.004–0.02 (consumption.parquet), а Dir(α_btw·μ) при
   граничных прототипах Dir(2·p̄) даёт базы с долями 1e-6..1e-9, где гамма-шум
   месяца вырождается и помесячная CLR-дистанция перестаёт калиброваться одним α.
3. Параметр alpha_within задан в 5-компонентной шкале спеки §1 (медиана 764 ≈ 750,
   MoM внутри суммы5). В 6-компонентном генераторе применяется эффективная
   концентрация alpha_within · ALPHA_WITHIN_6_FACTOR (фактор 17.0 подобран
   04.10.2026 на полной цепочке генератора под таргет медианной помесячной
   CLR-дистанции 0.18; прямое применение 750 даёт ≈ 0.42 — «ловушка MoM» §1).
4. Сезонный вектор s(t) в формуле §2.4 — скаляр объёма: в параметрах Дирихле он
   менял бы только концентрацию, не композицию. Применён к V; сезонность
   композиции целиком несёт МП-кривая m(t).
5. Потолок статической NMI при δ=0.075 по спеке = 0.815 [checked]; прямой пересчёт
   по механике §2.6 (модальная метка vs помесячная истина) даёт ≈ 0.91
   (K=6, Dir(1_K) размеры типов). Расхождение зафиксировано в PREREG_DEVIATIONS
   и в tests/test_synthetic.py.
6. Размеры типов ограничены снизу: π ~ Dir(1_K) пересэмплируется, пока
   min размер типа < max(2, ⌈n/100⌉). Пустой тип вырождает ячейку K в K−1
   (при seed=0, K=6, n=1000 без защиты: размеры [154,257,7,0,143,439]); аналог
   min_community у LFR. Распределение размеров остаётся Dir(1_K), условным на
   не-вырождение.
"""
from __future__ import annotations

import logging

import networkx as nx
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

log = logging.getLogger(__name__)

# --- Калибровка (измерено 04.10.2026, consumption.parquet) --------------------

# Средний 6-компонентный профиль: среднее по МО-месяцам долей
# (Продовольствие, Здоровье, Маркетплейсы, Общепит, Транспорт, Прочее=итог−сумма5).
P_BAR = np.array([0.438, 0.049, 0.138, 0.033, 0.058, 0.283])

# Сезонность итога: средний объём календарного месяца / средний по году (12 значений).
SEASONAL_12 = np.array([0.8477, 0.9158, 1.0071, 0.9646, 0.9829, 0.9982,
                        1.0332, 1.0424, 0.9721, 1.0101, 1.0205, 1.2053])

# Помесячная доля Маркетплейсов в ИТОГЕ (24 мес, немонотонна — декабрьские провалы
# января; в сумме5 та же кривая 0.134→0.266).
MP_CURVE_24 = np.array([0.0937, 0.0997, 0.1025, 0.1002, 0.1016, 0.1028,
                        0.1075, 0.1128, 0.1227, 0.1367, 0.1619, 0.1542,
                        0.1374, 0.1494, 0.1498, 0.1486, 0.1451, 0.1456,
                        0.1495, 0.1610, 0.1670, 0.1783, 0.1938, 0.1948])

TREND_YEAR = 1.131  # рост итога 2024/2023 = +13.1% (измерено 04.10.2026)

# Переводной фактор 5-комп → 6-комп для внутри-узловой концентрации (п.3 выше).
ALPHA_WITHIN_6_FACTOR = 17.0

MONTH_FLOOR = 0.01   # нижний пол долей в центрах месячных наблюдений (п.2 выше)
CLR_EPS = 1e-9       # пол для логов в CLR (нули Дирихле-гамм → андерфлоу float64)
IDX_MP = 2           # позиция «Маркетплейсы» в профиле

# Граф: целевой средний degree SBM-ядра и ширины атрибутно-гео слоя.
AVG_DEGREE = 20.0
GEO_MEDIAN_KM = 1816.0  # медиана попарных дистанций connection.parquet [checked §1]
GEO_L = 1800.0          # шкала гео-затухания ℓ [checked §1]
NOISE_KNN_K = 10        # kNN-прореживание шум-слоя (наш прод-граф тоже kNN)
NOISE_SIGMA = 1.0       # ширина CLR-ядра; порядок внутритипового CLR-разброса баз ≈ 1.3

# LFR: фиксированный retry-лист (провал «Could not assign communities» логируем,
# параметры не переподбираем — спека §3). Все seed'ы проверены на n=2016, mu=0.3,
# networkx 3.7: генерация успешна.
LFR_RETRY_SEEDS = (0, 2, 4, 10, 20)
LFR_MAX_ATTEMPTS = 5

_CATS = ["Продовольствие", "Здоровье", "Маркетплейсы", "Общественное питание",
         "Транспорт", "Прочее"]


# --- Общие куски цепочки ------------------------------------------------------

def _clr(x: np.ndarray) -> np.ndarray:
    """CLR-преобразование по последней оси с полом CLR_EPS."""
    lx = np.log(np.clip(x, CLR_EPS, None))
    return lx - lx.mean(axis=-1, keepdims=True)


def _rdirichlet(rng: np.random.Generator, a: np.ndarray) -> np.ndarray:
    """Векторный Dirichlet по строкам через гаммы; пол CLR_EPS против андерфлоу
    (np.random.dirichlet принимает только 1-D вектор концентраций)."""
    g = rng.gamma(np.clip(a, 1e-300, None), 1.0)
    g = np.clip(g, CLR_EPS, None)
    return g / g.sum(axis=-1, keepdims=True)


def _rake(mu: np.ndarray, w: np.ndarray, target: np.ndarray,
          iters: int = 200, tol: float = 1e-12) -> np.ndarray:
    """IPF-рейкинг прототипов: взвешенное арифметическое среднее строк → target.
    Мультипликативные поправки по компонентам + ре-нормировка строк."""
    mu = mu.copy()
    for _ in range(iters):
        m = w @ mu
        mu *= (target / np.clip(m, 1e-300, None))[None, :]
        mu /= mu.sum(axis=1, keepdims=True)
        if np.abs(w @ mu - target).max() < tol:
            break
    return mu


def _min_pairwise_clr(mu: np.ndarray) -> float:
    c = _clr(mu)
    d = [np.linalg.norm(c[i] - c[j]) for i in range(len(mu)) for j in range(i)]
    return min(d) if d else np.inf


def _mp_multipliers(T: int) -> np.ndarray:
    """Множители m(t) (T,6): доля МП следует MP_CURVE_24, остальные компоненты
    ре-нормируются пропорционально; временнОе среднее каждого множителя = 1."""
    if T > 24:
        raise ValueError("T > 24 не поддержан: МП-кривая откалибрована на 24 мес")
    mp = MP_CURVE_24[:T]
    p0 = P_BAR[IDX_MP]
    R = np.ones((T, 6))
    R[:, IDX_MP] = mp / p0
    other = (1.0 - mp) / (1.0 - p0)
    for j in range(6):
        if j != IDX_MP:
            R[:, j] = other
    return R / R.mean(axis=0, keepdims=True)


def _draw_prototypes(rng: np.random.Generator, K: int, alpha_proto: float,
                     w: np.ndarray, reject: bool = True) -> np.ndarray:
    """μ_k ~ Dir(α_proto·p̄) → рейкинг на p̄ → отбраковка min CLR < 1.0
    (пересэмпл тем же RNG, спека §2.2). Для LFR-режима reject=False."""
    while True:
        mu = _rdirichlet(rng, np.repeat((alpha_proto * P_BAR)[None, :], K, axis=0))
        mu = _rake(mu, w, P_BAR)
        if not reject or _min_pairwise_clr(mu) >= 1.0:
            return mu
        log.info("прототипы пересэмплированы: min CLR < 1.0")


def _draw_bases(rng: np.random.Generator, mu: np.ndarray, z: np.ndarray,
                alpha_btw: float) -> np.ndarray:
    return _rdirichlet(rng, alpha_btw * mu[z])


def _draw_months(rng: np.random.Generator, bases_seq: np.ndarray, T: int,
                 alpha_within_eff: float) -> np.ndarray:
    """X (T,n,6): M_i(t) ~ Dir(α_eff · norm(пол(B_i(t)) ∘ m(t)));
    bases_seq (T,n,6) — помесячные базы (у дрейфующих меняются с месяца смены)."""
    R = _mp_multipliers(T)
    Bf = np.clip(bases_seq, MONTH_FLOOR, None)
    Bf /= Bf.sum(axis=2, keepdims=True)
    C = Bf * R[:, None, :]
    a = alpha_within_eff * C / C.sum(axis=2, keepdims=True)
    g = rng.gamma(np.clip(a, 1e-300, None), 1.0)
    g = np.clip(g, CLR_EPS, None)
    return (g / g.sum(axis=2, keepdims=True)).astype(np.float64)


def _draw_volumes(rng: np.random.Generator, n: int, T: int) -> np.ndarray:
    """V (T,n): L_i · trend(год) · сезонность(месяц) · лог-нормальный шум.
    Тренд годовым шагом (помесячный тренд от сезонности отдельно не калиброван)."""
    L = rng.lognormal(0.0, 1.0, n)
    years = np.arange(T) // 12
    months = np.arange(T) % 12
    mult = (TREND_YEAR ** years) * SEASONAL_12[months]
    eps = rng.lognormal(0.0, 0.2, (T, n))
    return (L[None, :] * mult[:, None] * eps).astype(np.float64)


def _draw_geo(rng: np.random.Generator, n: int) -> np.ndarray:
    """Латентные 2D-координаты, отмасштабированные так, чтобы медиана попарных
    евклидовых дистанций = 1816 км (медиана connection.parquet, §1)."""
    g = rng.uniform(0.0, 1.0, (n, 2))
    d2 = ((g[:, None, :] - g[None, :, :]) ** 2).sum(axis=2)
    med = np.median(np.sqrt(d2[np.triu_indices(n, 1)]))
    return g * (GEO_MEDIAN_KM / med)


def _noise_layer(rng: np.random.Generator, bases_eff: np.ndarray,
                 geo: np.ndarray) -> sp.csr_matrix:
    """Атрибутно-гео слой: w_ij = exp(−d_CLR(B_i,B_j)²/2σ²)·exp(−geo_ij/ℓ),
    прореженный kNN (k=NOISE_KNN_K, объединение) — та же природа, что прод-граф."""
    c = _clr(bases_eff)
    d2 = (c * c).sum(1)[:, None] + (c * c).sum(1)[None, :] - 2.0 * (c @ c.T)
    np.clip(d2, 0.0, None, out=d2)
    g2 = (geo * geo).sum(1)[:, None] + (geo * geo).sum(1)[None, :] - 2.0 * (geo @ geo.T)
    np.clip(g2, 0.0, None, out=g2)
    W = np.exp(-d2 / (2.0 * NOISE_SIGMA**2)) * np.exp(-np.sqrt(g2) / GEO_L)
    np.fill_diagonal(W, 0.0)
    n = W.shape[0]
    k = min(NOISE_KNN_K, n - 1)
    top = np.argpartition(-W, kth=k, axis=1)[:, :k]
    rows = np.repeat(np.arange(n), k)
    cols = top.ravel()
    vals = W[rows, cols]
    A = sp.csr_matrix((vals, (rows, cols)), shape=(n, n))
    A = A.maximum(A.T)  # объединение kNN
    return A


def _sbm_layer(rng: np.random.Generator, z: np.ndarray, mu_edge: float) -> sp.csr_matrix:
    """Пуассоновский SBM: w_ij ~ Poisson(λ·похожесть); λ из целевого avg_degree=20
    и доли межтиповых рёбер mu_edge (приближение сбалансированных блоков по
    размерам строки)."""
    n = len(z)
    counts = np.bincount(z).astype(float)
    lam_in = AVG_DEGREE * (1.0 - mu_edge) / np.clip(counts[z] - 1.0, 1.0, None)
    lam_out = AVG_DEGREE * mu_edge / np.clip(n - counts[z], 1.0, None)
    same = z[:, None] == z[None, :]
    lam = np.where(same, lam_in[:, None], lam_out[:, None])
    np.fill_diagonal(lam, 0.0)
    W = rng.poisson(lam).astype(np.float64)
    W = np.maximum(W, W.T)
    return sp.csr_matrix(W)


# --- Генераторы ---------------------------------------------------------------

def make_ppdataset(seed: int, K: int, n: int = 2016, T: int = 24,
                   alpha_btw: float = 80.0, alpha_within: float = 750.0,
                   alpha_proto: float = 2.0, delta: float = 0.0,
                   mu_edge: float = 0.25) -> dict:
    """PP-Dir генератор (спека §2). Один seed реплики → вся случайность.

    Возвращает dict: X (T,n,6) float64 — месячные доли; V (T,n) — объёмы;
    A (n,n) csr — объединение SBM-ядра и атрибутно-гео слоя (max весов);
    z (T,n) int — истинные метки по месяцам (с дрейфом); geo (n,2) — латентные
    координаты; meta — параметры и служебная информация реплики.
    """
    rng = np.random.default_rng(seed)
    alpha_eff = ALPHA_WITHIN_6_FACTOR * alpha_within

    # 1. Типы: размеры неровные, π ~ Dir(1_K) (спека §2.1), с защитой от
    # вырождения: пересэмпл, пока min размер типа < max(2, ⌈n/100⌉) — п.6 шапки.
    min_size = max(2, int(np.ceil(0.01 * n)))
    while True:
        pi = rng.dirichlet(np.ones(K))
        z0 = rng.choice(K, n, p=pi)
        if np.bincount(z0, minlength=K).min() >= min_size:
            break
    w = np.bincount(z0, minlength=K) / n

    # 2. Прототипы с отбраковкой min CLR < 1.0 (после рейкинга на p̄).
    mu = _draw_prototypes(rng, K, alpha_proto, w, reject=(K > 1))

    # 3. Базы узлов.
    bases = _draw_bases(rng, mu, z0, alpha_btw)

    # 4. Дрейф: δ узлов меняют тип в τ_i ~ Uniform{4..21} (окно спеки для T=24,
    # при другом T масштабируется), новый тип ~ Uniform из остальных K−1.
    z = np.tile(z0, (T, 1))
    n_drift = int(round(delta * n))
    taus = np.zeros(0, dtype=int)
    drift_idx = np.zeros(0, dtype=int)
    if n_drift > 0 and K > 1:
        drift_idx = rng.choice(n, n_drift, replace=False)
        lo = max(1, round(4 * T / 24))
        hi = max(lo + 1, round(21 * T / 24) + 1)
        taus = rng.integers(lo, hi, n_drift)
        off = rng.integers(0, K - 1, n_drift)
        new = np.where(off < z0[drift_idx], off, off + 1)
        for i, t, nl in zip(drift_idx, taus, new):
            z[t:, i] = nl
        # база пересэмплируется из нового прототипа с месяца смены:
        # помесячная последовательность баз (T,n,6), до τ — старая, после — новая
        bases_post = _draw_bases(rng, mu, z[T - 1], alpha_btw)
        bases_seq = np.repeat(bases[None, :, :], T, axis=0)
        for i, t in zip(drift_idx, taus):
            bases_seq[t:, i, :] = bases_post[i]
    else:
        bases_seq = np.repeat(bases[None, :, :], T, axis=0)

    # 5. Месячные наблюдения вокруг помесячных баз.
    X = _draw_months(rng, bases_seq, T, alpha_eff)

    # 6. Объёмы, гео, граф.
    V = _draw_volumes(rng, n, T)
    geo = _draw_geo(rng, n)

    # Эффективная база для шум-слоя: среднее по месяцам (у дрейфующих —
    # взвешенное pre/post), спека §2.5 использует B_i напрямую.
    bases_eff = bases_seq.mean(axis=0)
    A_noise = _noise_layer(rng, bases_eff, geo)
    A_sbm = _sbm_layer(rng, z0, mu_edge)
    A = A_sbm.maximum(A_noise).tocsr()
    A.setdiag(0.0)
    A.eliminate_zeros()

    return {
        "X": X, "V": V, "A": A, "z": z.astype(np.int64), "geo": geo,
        "meta": dict(
            generator="ppdir", seed=seed, K=K, n=n, T=T,
            alpha_btw=alpha_btw, alpha_within=alpha_within,
            alpha_within_eff=alpha_eff, alpha_proto=alpha_proto,
            delta=delta, mu_edge=mu_edge, n_drift=n_drift,
            type_sizes=np.bincount(z0, minlength=K),
            min_proto_clr=_min_pairwise_clr(mu),
            prototypes=mu, avg_degree_target=AVG_DEGREE,
            categories=list(_CATS),
        ),
    }


def make_lfr_dir(seed: int, n: int = 2016, mu: float = 0.3, T: int = 24,
                 alpha_btw: float = 80.0, alpha_within: float = 750.0,
                 alpha_proto: float = 2.0) -> dict:
    """LFR+Dir (спека §3): структура networkx LFR, атрибуты — та же Дирихле-цепочка
    (коммьюнити → прототип Dir(α_proto·p̄) без отбраковки → базы → месяцы).

    mu < 0.3 не предлагается вообще [checked §3: нестабильно]. Провалы
    «Could not assign communities» ретраятся по фиксированному списку
    LFR_RETRY_SEEDS (≤ LFR_MAX_ATTEMPTS попыток), провалы логируются.
    """
    if mu < 0.3:
        raise ValueError(f"mu={mu} < 0.3: LFR при mu≤0.25 нестабилен (спека §3), "
                         "в протокол идут только mu ∈ {0.3, 0.4}")

    graph, seed_used, tried = None, None, []
    for s in [seed] + [x for x in LFR_RETRY_SEEDS if x != seed]:
        if len(tried) >= LFR_MAX_ATTEMPTS:
            break
        tried.append(s)
        try:
            graph = nx.generators.community.LFR_benchmark_graph(
                n, 2.5, 1.5, mu, average_degree=int(AVG_DEGREE),
                min_community=30, seed=s)
            seed_used = s
            break
        except nx.ExceededMaxIterations:
            log.warning("LFR: seed=%s провал (Could not assign communities), retry", s)
    if graph is None:
        raise RuntimeError(f"LFR не сгенерировался за {LFR_MAX_ATTEMPTS} попыток, "
                           f"seed'ы {tried}; провалы залогированы")

    comms = sorted({frozenset(graph.nodes[v]["community"]) for v in graph},
                   key=min)
    node2c = {}
    for cid, nodes in enumerate(comms):
        for v in nodes:
            node2c[v] = cid
    z0 = np.array([node2c[v] for v in range(n)], dtype=np.int64)
    C = len(comms)
    w = np.bincount(z0, minlength=C) / n

    rng = np.random.default_rng(seed)
    alpha_eff = ALPHA_WITHIN_6_FACTOR * alpha_within
    muc = _draw_prototypes(rng, C, alpha_proto, w, reject=False)  # §3: без отбраковки
    bases = _draw_bases(rng, muc, z0, alpha_btw)
    X = _draw_months(rng, np.repeat(bases[None, :, :], T, axis=0), T, alpha_eff)
    V = _draw_volumes(rng, n, T)
    geo = _draw_geo(rng, n)

    A = nx.to_scipy_sparse_array(graph, dtype=np.float64, format="csr")

    return {
        "X": X, "V": V, "A": A, "z": np.tile(z0, (T, 1)), "geo": geo,
        "meta": dict(
            generator="lfr_dir", seed=seed, seed_used=seed_used, seeds_tried=tried,
            n=n, T=T, mu=mu, n_communities=C,
            community_sizes=np.bincount(z0, minlength=C),
            alpha_btw=alpha_btw, alpha_within=alpha_within,
            alpha_within_eff=alpha_eff, alpha_proto=alpha_proto,
            avg_degree_target=AVG_DEGREE, categories=list(_CATS),
        ),
    }


# --- Метрики против истины (спека §4) -----------------------------------------

def nmi(labels_true, labels_pred) -> float:
    """NMI, average_method='arithmetic' (предрег §4.1)."""
    return float(normalized_mutual_info_score(labels_true, labels_pred,
                                              average_method="arithmetic"))


def ari(labels_true, labels_pred) -> float:
    return float(adjusted_rand_score(labels_true, labels_pred))


def per_snapshot_nmi(z_true: np.ndarray, z_pred: np.ndarray) -> np.ndarray:
    """Помесячная NMI: z_true/z_pred (T,n) → массив (T,)."""
    z_true = np.asarray(z_true)
    z_pred = np.asarray(z_pred)
    return np.array([nmi(z_true[t], z_pred[t]) for t in range(z_true.shape[0])])


def f1_type_switch(z_true: np.ndarray, z_pred: np.ndarray, window: int = 1) -> float:
    """F1 детекции смены типа (спека §4.3): истинное событие (узел, месяц смены)
    детектировано, если у того же узла есть предсказанная смена в окне ±window.
    Сопоставление жадное по времени, каждое событие используется ≤ 1 раза."""
    z_true = np.asarray(z_true)
    z_pred = np.asarray(z_pred)

    def _switches(zz: np.ndarray) -> dict[int, list[int]]:
        sw: dict[int, list[int]] = {}
        t_idx, i_idx = np.argwhere(zz[1:] != zz[:-1]).T
        for t, i in zip(t_idx.tolist(), i_idx.tolist()):
            sw.setdefault(i, []).append(t + 1)
        return sw

    st, sp_ = _switches(z_true), _switches(z_pred)
    tp = fp = fn = 0
    for i in set(st) | set(sp_):
        tt = sorted(st.get(i, []))
        pp = sorted(sp_.get(i, []))
        used = [False] * len(pp)
        for t in tt:
            cand = [j for j, p in enumerate(pp) if not used[j] and abs(p - t) <= window]
            if cand:
                used[min(cand, key=lambda j: abs(pp[j] - t))] = True
                tp += 1
            else:
                fn += 1
        fp += len(pp) - sum(used)
    return 2.0 * tp / max(2 * tp + fp + fn, 1)
