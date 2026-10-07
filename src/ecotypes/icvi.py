"""ICVI — панель индексов качества кластеризации (критерий СберИндекс 2026, 15%).

Формулы: research/32_icvi_implementation.md. Сетевые индексы (AVI/AVU/ANUI/Q)
сверены с эталонной библиотекой Pattern (Sorooshi, github.com/Sorooshi/Pattern)
до 1e-16; MQ в Pattern отсутствует и берётся из первоисточников.

Направления: SW ↑, CH_over_N ↑, S_Dbw ↓, AVI ↑, AVU ↓, ANUI ↑, MQ ↑.
Предпосылки: A симметрична, diag(A) = 0; признаки X стандартизованы.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
from scipy import sparse
from sklearn.metrics import calinski_harabasz_score, silhouette_score

__all__ = [
    "sw",
    "ch_over_n",
    "sdbw",
    "avi",
    "avu",
    "anui",
    "mq",
    "q_modularity",
    "compute_all",
    "permutation_baseline",
]

# Дефолты зеркалят configs/default.yaml (секция icvi) и fixed_decisions prereg.yaml.
_DEFAULT_PANEL = ("SW", "CH_over_N", "S_Dbw", "AVI", "AVU", "MQ")
_DEFAULT_MQ_VARIANT = "mancoridis"
_DEFAULT_SDBW_ALG_NOISE = "bind"


# ---------------------------------------------------------------- attribute-space

def sw(X: np.ndarray, labels: np.ndarray) -> float:
    """Silhouette width (Rousseeuw 1987), обёртка sklearn.metrics.silhouette_score.

    SW = mean_i (b_i − a_i) / max(a_i, b_i), ∈ [−1, 1], больше = лучше.
    Требует 2 ≤ k ≤ n−1. Источник: sklearn; в статьях жюри — атрибутивный индекс.
    """
    return float(silhouette_score(X, labels))


def ch_over_n(X: np.ndarray, labels: np.ndarray) -> float:
    """Calinski–Harabasz, нормированный на размер выборки: CH/n.

    CH = (B/(k−1)) / (W/(n−k)), B/W — меж/внутрикластерная дисперсия; ∈ [0, ∞), ↑.
    Деление на n — по наблюдению Doklady-2025 (авторы критерия) о механическом
    росте CH с размером выборки; в отчёт и сравнение месяцев идёт только CH/n.
    """
    X = np.asarray(X)
    return float(calinski_harabasz_score(X, labels) / X.shape[0])


def sdbw(X: np.ndarray, labels: np.ndarray, alg_noise: str = "bind") -> float:
    """S_Dbw (Halkidi & Vazirgiannis), пакет s-dbw; ∈ [0, ∞), меньше = лучше.

    Вызов зафиксирован пререгом: S_Dbw(X, labels, centers_id=None, method='Tong',
    alg_noise='bind', centr='mean', nearest_centr=True, metric='euclidean').
    Дефолт пакета alg_noise='comb' осознанно переопределён на 'bind'
    (fixed_decisions.sdbw_alg_noise в prereg.yaml). k = 1 невозможен — пакет
    бросает ValueError, не подавляем.
    """
    from s_dbw import S_Dbw  # локальный импорт: сетевые индексы не должны тянуть пакет

    return float(S_Dbw(np.asarray(X), np.asarray(labels), alg_noise=alg_noise))


# ---------------------------------------------------------------- network (Pattern)

def _block_sums(A: np.ndarray | sparse.spmatrix, labels: np.ndarray):
    """S = M·A·Mᵀ: S[i,j] — сумма весов между кластерами i и j.

    Для симметричной A с нулевой диагональю: S[i,i] = 2·(внутренний вес кластера i),
    S[i,j] (i≠j) = межкластерный вес. Дословно соответствует построению в Pattern.
    """
    labels = np.asarray(labels)
    classes = np.unique(labels)
    k = len(classes)
    M = (labels[None, :] == classes[:, None]).astype(float)  # k×n индикаторная
    if sparse.issparse(A):
        Ms = sparse.csr_matrix(M)
        S = (Ms @ A @ Ms.T).toarray()
    else:
        S = M @ np.asarray(A, dtype=float) @ M.T
    return S, M.sum(axis=1), k


def avi(A: np.ndarray | sparse.spmatrix, labels: np.ndarray) -> float:
    """Average Isolability (Biswas & Biswas 2017; реализация Pattern).

    AVI = (1/k)·Σᵢ Sᵢᵢ / ΣⱼSᵢⱼ, ∈ [0, 1], больше = лучше.
    Термы с нулевым знаменателем (изолированные кластеры без рёбер) = 0, как в Pattern.

    Сноска про двойной счёт: Sᵢᵢ = 2·intᵢ, а Sᵢⱼ (i≠j) = 1·extᵢⱼ, поэтому
    AVI = 2·intᵢ/(2·intᵢ + extᵢ) — разбиение «половина рёбер внутри» даёт
    AVI = 2/3, а не 1/2. Значения систематически выше наивного ожидания;
    это свойство определения, в отчёте проговаривается.
    """
    S, _, k = _block_sums(A, labels)
    total = S.sum(axis=1)
    iso = np.zeros(k)
    np.divide(np.diag(S), total, out=iso, where=total != 0)
    return float(iso.sum() / k)


def avu(A: np.ndarray | sparse.spmatrix, labels: np.ndarray) -> float:
    """Average Unifiability (Biswas & Biswas 2017; реализация Pattern).

    AVU = (1/k)·ΣᵢΣⱼ≠ᵢ Sᵢⱼ / (outᵢ + inⱼ − Sᵢⱼ), outᵢ = Σⱼ≠ᵢSᵢⱼ, inⱼ = Σᵢ≠ⱼSᵢⱼ;
    ∈ [0, k−1] (каждый вклад ≤ 1), меньше = лучше. Нулевые знаменатели = 0.
    Смысл слагаемого: доля веса пары (i,j) во внешнем весе объединения i∪j.

    Ловушка: на n ≈ 2000 нулевое (перестановочное) распределение AVU имеет почти
    нулевую дисперсию → z-score обманчиво мал по модулю; репортить z + percentile
    + null_mean ± null_std (см. permutation_baseline).
    """
    S, _, k = _block_sums(A, labels)
    d = np.diag(S)
    out_i = S.sum(axis=1) - d
    in_j = S.sum(axis=0) - d
    num = S.copy()
    np.fill_diagonal(num, 0.0)
    den = out_i[:, None] + in_j[None, :] - S
    frac = np.zeros_like(S)
    np.divide(num, den, out=frac, where=den != 0)
    return float(frac.sum() / k)


def anui(A: np.ndarray | sparse.spmatrix, labels: np.ndarray) -> float:
    """Average Normalized Unifiability-Isolability (Pattern), бонусный индекс панели.

    ANUI = 1 / (AVU + 1/AVI), ∈ (0, min(AVI, 1/AVU)], больше = лучше.
    AVI = 0 или знаменатель 0 → 0.0 (как в Pattern).
    """
    avi_v = avi(A, labels)
    avu_v = avu(A, labels)
    # Дословно Pattern: denom = avu + 1/avi при avi != 0, иначе 0 (→ anui = 0).
    denom = (avu_v + 1.0 / avi_v) if avi_v != 0 else 0.0
    return float(1.0 / denom if denom != 0 else 0.0)


def mq(
    A: np.ndarray | sparse.spmatrix,
    labels: np.ndarray,
    variant: str = "mancoridis",
) -> float:
    """Modularization Quality. В Pattern отсутствует — формулы из первоисточников.

    variant='mancoridis' (дефолт, prereg.yaml fixed_decisions.mq_variant;
    Mancoridis et al., IWPC 1998, первоисточник понятия):
        MQ = (1/k)·Σᵢ μᵢ/Nᵢ² − (2/(k(k−1)))·Σᵢ<ⱼ εᵢⱼ/(2NᵢNⱼ),
        Nᵢ — число узлов кластера i, εᵢⱼ = Sᵢⱼ. ∈ [−1, 1], ↑; сопоставим между K.
        При k = 1: MQ = A₁ = μ₁/N₁².
    variant='turbomq' (только абляция; в отчёт не идёт — механически растёт с k):
        TurboMQ (Bunch): MQ = Σᵢ CFQᵢ, CFQᵢ = 2μᵢ/(2μᵢ + εᵢ),
        μᵢ = Sᵢᵢ/2 (внутренний вес), εᵢ = внешний вес кластера i.
        ∈ [0, k], ↑; сравнивать только при фиксированном k.
    """
    if variant not in ("turbomq", "mancoridis"):
        raise ValueError(f"mq variant должен быть 'turbomq'|'mancoridis', получено {variant!r}")
    S, N, k = _block_sums(A, labels)
    mu = np.diag(S) / 2.0
    out = S.sum(axis=1) - np.diag(S)
    if variant == "turbomq":
        den = 2.0 * mu + out
        cfq = np.zeros(k)
        np.divide(2.0 * mu, den, out=cfq, where=den != 0)
        return float(cfq.sum())
    a_i = mu / N**2
    if k == 1:
        return float(a_i[0])
    ii, jj = np.triu_indices(k, k=1)
    inter = (S[ii, jj] / (2.0 * N[ii] * N[jj])).sum()
    return float(a_i.mean() - inter / (k * (k - 1) / 2.0))


def q_modularity(A: np.ndarray | sparse.spmatrix, labels: np.ndarray) -> float:
    """Модульность Ньюмана (Pattern, modularity) — ТОЛЬКО референс, не доказательство.

    Q = Σᵢ [Sᵢᵢ/(2m) − (dᵢ/(2m))²], m = ΣA/2, dᵢ — сумма степеней кластера i.
    Циркулярна для методов, её оптимизирующих (Leiden/Louvain) — в отчёте с †.
    Граф без рёбер (m = 0) → 0.0.
    """
    S, _, _ = _block_sums(A, labels)
    m2 = S.sum()  # 2m
    if m2 == 0:
        return 0.0
    d = S.sum(axis=1)
    return float((np.diag(S) / m2 - (d / m2) ** 2).sum())


# ---------------------------------------------------------------- panel & baseline

def _cfg_get(cfg: Any, name: str, default: Any) -> Any:
    """Читает поле из IcviCfg (pydantic), dict или None."""
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def compute_all(
    X: np.ndarray | None,
    A: np.ndarray | sparse.spmatrix | None,
    labels: np.ndarray,
    cfg_icvi: Any = None,
) -> dict[str, float]:
    """Полная панель ICVI из конфига (IcviCfg / dict / None → дефолты prereg).

    cfg_icvi.panel — подмножество {SW, CH_over_N, S_Dbw, AVI, AVU, MQ, ANUI, Q};
    ANUI и Q только по явному запросу (бонус / референс с † соответственно).
    cfg_icvi.mq_variant — 'turbomq'|'mancoridis'; cfg_icvi.sdbw_alg_noise — 'bind'|'comb'.
    X=None пропускает атрибутивные индексы, A=None — сетевые (помесячный прогон
    может не иметь одной из модальностей).
    """
    panel = list(_cfg_get(cfg_icvi, "panel", _DEFAULT_PANEL))
    mq_variant = _cfg_get(cfg_icvi, "mq_variant", _DEFAULT_MQ_VARIANT)
    alg_noise = _cfg_get(cfg_icvi, "sdbw_alg_noise", _DEFAULT_SDBW_ALG_NOISE)

    attr = {"SW": sw, "CH_over_N": ch_over_n}
    net = {"AVI": avi, "AVU": avu, "ANUI": anui, "Q": q_modularity}

    out: dict[str, float] = {}
    for name in panel:
        if name in attr:
            if X is not None:
                out[name] = attr[name](X, labels)
        elif name == "S_Dbw":
            if X is not None:
                out[name] = sdbw(X, labels, alg_noise=alg_noise)
        elif name == "MQ":
            if A is not None:
                out[name] = mq(A, labels, variant=mq_variant)
        elif name in net:
            if A is not None:
                out[name] = net[name](A, labels)
        else:
            raise ValueError(f"неизвестный индекс в icvi.panel: {name!r}")
    return out


def permutation_baseline(
    func: Callable[..., float],
    *args: Any,
    labels: np.ndarray,
    runs: int = 100,
    rng: np.random.Generator | int | None = None,
    higher_better: bool = True,
) -> dict[str, float]:
    """Случайный baseline индекса: runs перестановок меток (кардинальности сохраняются).

    func вызывается как func(*args, labels_perm). Протокол статей жюри (ESWA-2026):
    z = (obs − mean_null)/std_null; percentile — доля null не хуже obs
    (для ↑: P(null ≤ obs), для ↓: P(null ≥ obs); передать higher_better=False).

    Возвращает z, percentile, null_mean, null_std — null_mean/null_std ОБЯЗАТЕЛЬНЫ
    в репорте: у AVU нулевое распределение имеет почти нулевую дисперсию при
    n ≈ 2000, и z без разброса null читается неверно.
    std_null = 0 → z = 0 при obs == mean, иначе ±inf (знак obs − mean).
    """
    labels = np.asarray(labels)
    if not isinstance(rng, np.random.Generator):
        rng = np.random.default_rng(rng)
    obs = float(func(*args, labels))
    n = labels.shape[0]
    nulls = np.empty(runs)
    for r in range(runs):
        nulls[r] = func(*args, labels[rng.permutation(n)])
    null_mean = float(nulls.mean())
    null_std = float(nulls.std(ddof=1)) if runs > 1 else 0.0
    if null_std > 0:
        z = (obs - null_mean) / null_std
    elif obs == null_mean:
        z = 0.0
    else:
        z = float(np.copysign(np.inf, obs - null_mean))
    if higher_better:
        percentile = float((nulls <= obs).mean())
    else:
        percentile = float((nulls >= obs).mean())
    return {"z": float(z), "percentile": percentile, "null_mean": null_mean, "null_std": null_std}
