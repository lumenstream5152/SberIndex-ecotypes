"""Динамический слой (research/29, сборка 05.10.2026): эволюция типов во времени —
ядро-дифференциация против статичного эталона.

Схема two-stage (спека §2): per-snapshot Leiden (RBConfiguration, γ из конфига,
25 seeds из stage_seed(seed, 'dynamics')) → CSPA-консенсус (cluster.py, не
дублируем) → согласование меток с t−1 Hungarian'ом на (1 − Jaccard),
τ = cfg.dynamics.tau_inherit → реестр типов (birth/death/merge/split) →
anti-flicker smoothing окном-3 (публичные переходы только на smoothed, §10.4) →
стабильность: ARI(t,t+1) vs seed-null vs 5%-пертурбация рёбер (§5; flagged-пары
в событийный слой не допускаются).

Multislice сознательно отвергнут (спека §4): сила временной связи у нас
ИЗМЕРЯЕТСЯ (тройка ARI), а не предполагается приором.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment
from scipy.stats import chi2

from .cluster import leiden_consensus_cspa, run_leiden
from .config import Config
from .seeds import stage_seed
from .synthetic import ari

log = logging.getLogger(__name__)

__all__ = [
    "TAU_SPLIT", "PERTURB_FRAC", "PERTURB_REPS", "PAGERANK_ALPHA",
    "load_snapshots", "snapshot_labels", "snapshot_labels_with_agreement",
    "labels_from_snapshots", "match_labels", "smooth_labels",
    "flicker_share", "transition_share", "perturb_graph", "stability_report",
    "transitions_monthly", "transitions_quarterly", "p_bar",
    "stationary", "goodman_order_test", "spatial_markov", "events_table",
    "run_dynamics",
]

# Дефолты спеки 29 §10, которых нет в strict-конфиге (DynamicsCfg — только
# tau_inherit/metric/smoothing_window): зафиксированы здесь, не размазаны.
TAU_SPLIT = 0.15        # порог merge/split (спека §3: калибровать на финальном графе)
PERTURB_FRAC = 0.05     # доля пертурбируемых рёбер в нуль-модели (§5)
PERTURB_REPS = 10       # реплик пертурбации на снимок (§5)
PAGERANK_ALPHA = 0.99   # регуляризация π при неэргодичной P̄ (§6.2)
BOOTSTRAP_B = 50        # node bootstrap для OR spatial Markov (§6.4)
BOOTSTRAP_FRAC = 0.8    # сабсемпл узлов без возвращения (§5, тот же протокол)
CONTEXT_MIN_SHARE = 0.5  # доля доминирующего типа среди соседей, иначе «mixed» (§6.4)
CLR_COLS = ("clr_prod", "clr_health", "clr_market",
            "clr_food", "clr_transp", "clr_proch")


# ---------------------------------------------------------------- вход

def load_snapshots(graphs_dir: str | Path) -> tuple[list[sp.csr_matrix], pd.DataFrame]:
    """snap_t*.npz (общий edge-set, месячные веса — сломанное допущение волны 1 №2)
    + node_index.parquet, отсортированный по row_idx. Состав узлов общий (ядро
    2016 МО с полной историей), поэтому все переходные статистики — на всех узлах."""
    gdir = Path(graphs_dir)
    files = sorted(gdir.glob("snap_t*.npz"))
    if not files:
        raise FileNotFoundError(f"в {gdir} нет snap_t*.npz")
    snaps = [sp.load_npz(f).tocsr() for f in files]
    n = snaps[0].shape[0]
    if any(A.shape != (n, n) for A in snaps):
        raise ValueError("снимки разного размера — контракт общего node-set нарушен")
    nidx = pd.read_parquet(gdir / "node_index.parquet")
    nidx = nidx.sort_values("row_idx").reset_index(drop=True)
    if len(nidx) != n:
        raise ValueError(f"node_index ({len(nidx)}) != размер снимков ({n})")
    return snaps, nidx


def _snapshot_seeds(cfg: Config, t: int) -> list[int]:
    """R = cfg.cluster.leiden.seeds_per_snapshot seeds снимка t из
    stage_seed(seed, 'dynamics') — свои на каждый снимок (seed-артефакты не
    протекают в кросс-месячный ARI общим seed'ом)."""
    base = stage_seed(cfg.seed, "dynamics")
    return [stage_seed(base, f"{t:02d}:{s:02d}")
            for s in range(cfg.cluster.leiden.seeds_per_snapshot)]


def _align_to(ref: np.ndarray, lab: np.ndarray) -> np.ndarray:
    """Перевод меток lab в нумерацию ref по максимальному пересечению кластеров
    (id у отдельных прогонов произвольны — без выравнивания сравнение
    «прогон == консенсус» бессмысленно занижено)."""
    ids_r, ids_l = np.unique(ref), np.unique(lab)
    _, inter = _jaccard(ref, lab, ids_r, ids_l)
    return ids_r[inter.argmax(0)[np.searchsorted(ids_l, lab)]]


def labels_from_snapshots(snaps: list[sp.csr_matrix], cfg: Config):
    """Leiden-консенсус каждого снимка: CSPA по R seeds (γ из конфига, без
    перенастройки по месяцам — пульсация K не смешивается с эволюцией, §2).
    Возвращает (Z (T,n) консенсус, agree (T,n) seed_agreement,
    seed_ari (T,R) ARI прогона к консенсусу, z_single (T,n) прогон seed[0] —
    база пертурб-нулевой §5). Одиночные прогоны теми же seeds, что внутри CSPA
    (детерминизм leidenalg), поэтому agreement — точный по спеке §2."""
    gamma = cfg.cluster.leiden.gamma
    T, n = len(snaps), snaps[0].shape[0]
    R = cfg.cluster.leiden.seeds_per_snapshot
    Z = np.zeros((T, n), dtype=np.int64)
    z_single = np.zeros((T, n), dtype=np.int64)
    agree = np.zeros((T, n))
    seed_ari = np.zeros((T, R))
    for t, A in enumerate(snaps):
        seeds = _snapshot_seeds(cfg, t)
        cons = leiden_consensus_cspa(A, gamma=gamma, seeds=seeds)
        singles = [run_leiden(A, gamma=gamma, seed=s) for s in seeds]
        Z[t], z_single[t] = cons, singles[0]
        agree[t] = np.mean([_align_to(cons, s) == cons for s in singles], axis=0)
        seed_ari[t] = [ari(s, cons) for s in singles]
        log.info("снимок %d/%d: K=%d, seed-ARI med=%.3f, agreement=%.3f",
                 t + 1, T, int(cons.max()) + 1, float(np.median(seed_ari[t])),
                 float(agree[t].mean()))
    return Z, agree, seed_ari, z_single


def snapshot_labels_with_agreement(graphs_dir: str | Path, cfg: Config):
    snaps, _ = load_snapshots(graphs_dir)
    return labels_from_snapshots(snaps, cfg)


def snapshot_labels(graphs_dir: str | Path, cfg: Config) -> np.ndarray:
    """(T, n) int — CSPA-консенсусные метки снимков (контракт соседних модулей)."""
    return snapshot_labels_with_agreement(graphs_dir, cfg)[0]


# ---------------------------------------------------------------- матчинг

def _jaccard(za: np.ndarray, zb: np.ndarray, ids_a: np.ndarray, ids_b: np.ndarray):
    """Jaccard-матрица составов (K_a × K_b) + матрица пересечений. Узлы общие."""
    K1, K2 = len(ids_a), len(ids_b)
    pa, pb = np.searchsorted(ids_a, za), np.searchsorted(ids_b, zb)
    inter = np.bincount(pa * K2 + pb, minlength=K1 * K2).reshape(K1, K2).astype(float)
    union = inter.sum(1, keepdims=True) + inter.sum(0, keepdims=True) - inter
    J = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
    return J, inter


def match_labels(z: np.ndarray, months: list[str] | None = None, *,
                 tau: float = 0.3, tau_split: float = TAU_SPLIT
                 ) -> tuple[np.ndarray, pd.DataFrame]:
    """Hungarian (scipy linear_sum_assignment, прямоугольные матрицы нативно —
    без паддинга, спека §1) на cost = 1 − Jaccard между живыми типами t−1 и
    кластерами t. Наследование id при J ≥ τ, иначе birth; death/merge/split —
    по спеке §3 (merge: второй родитель с J ≥ τ_split без своего persist-партнёра
    растворяется в большем, merged_into; split: не-persist кластер t, чей макс.
    пересечения тип a жив и имеет persist-партнёра, J(a,·) ≥ τ_split → parent_id=a;
    тай-брейк размера родителя — меньший type_id).

    Возвращает (z_matched (T,n) глобальные id типов, реестр DataFrame:
    type_id, birth_month, death_month, parent_id, merged_into,
    n_nodes_by_month (json), lifetime). Мёртвые id не воскресают."""
    z = np.asarray(z, dtype=np.int64)
    T, _ = z.shape
    months = [str(m) for m in (months if months is not None else range(T))]
    zm = np.empty_like(z)
    reg: dict[int, dict] = {}
    next_id = 0

    def _birth(month: str, parent: int | None) -> int:
        nonlocal next_id
        tid = next_id
        reg[tid] = dict(birth_month=month, death_month=None,
                        parent_id=parent, merged_into=None)
        next_id += 1
        return tid

    first = np.unique(z[0])
    zm[0] = np.array([_birth(months[0], None) for _ in first],
                     dtype=np.int64)[np.searchsorted(first, z[0])]

    for t in range(1, T):
        prev_ids = np.unique(zm[t - 1])
        cur = np.unique(z[t])
        J, inter = _jaccard(zm[t - 1], z[t], prev_ids, cur)
        r, c = linear_sum_assignment(1.0 - J)
        persist: dict[int, int] = {}   # кластер t -> наследуемый type_id
        claimed: set[int] = set()      # типы t−1 с persist-партнёром
        for ri, ci in zip(r.tolist(), c.tolist()):
            if J[ri, ci] >= tau:
                persist[int(cur[ci])] = int(prev_ids[ri])
                claimed.add(int(prev_ids[ri]))
        sizes = {int(p): int((zm[t - 1] == p).sum()) for p in prev_ids}
        # merge (§3): у persist-пары (a → b) второй родитель a' без своего
        # persist с J(a', b) ≥ τ_split растворяется в большем родителе.
        for cl, tid in list(persist.items()):
            ci = int(np.searchsorted(cur, cl))
            second = [int(prev_ids[ri]) for ri in range(len(prev_ids))
                      if int(prev_ids[ri]) not in claimed and J[ri, ci] >= tau_split]
            if not second:
                continue
            big = max([tid] + second, key=lambda p: (sizes[p], -p))
            for p in [tid] + second:
                if p != big:
                    reg[p]["death_month"] = months[t - 1]
                    reg[p]["merged_into"] = big
                    claimed.discard(p)
            persist[cl] = big
            claimed.add(big)
        # death: живые в t−1 без persist и не поглощённые merge
        for p in prev_ids.tolist():
            p = int(p)
            if p not in claimed and reg[p]["death_month"] is None:
                reg[p]["death_month"] = months[t - 1]
        # split / birth для кластеров t без наследования (§3)
        assign = dict(persist)
        for cl in cur.tolist():
            cl = int(cl)
            if cl in assign:
                continue
            ci = int(np.searchsorted(cur, cl))
            parent = None
            for ri in np.argsort(-inter[:, ci]).tolist():  # по массе пересечения
                p = int(prev_ids[ri])
                if (p in claimed and reg[p]["death_month"] is None
                        and J[ri, ci] >= tau_split):
                    parent = p
                    break
            assign[cl] = _birth(months[t], parent)
        ids_t = np.array([assign[int(cl)] for cl in cur.tolist()], dtype=np.int64)
        zm[t] = ids_t[np.searchsorted(cur, z[t])]

    rows = []
    for tid in sorted(reg):
        d = reg[tid]
        counts = {months[t]: int((zm[t] == tid).sum())
                  for t in range(T) if bool((zm[t] == tid).any())}
        rows.append(dict(type_id=tid, birth_month=d["birth_month"],
                         death_month=d["death_month"], parent_id=d["parent_id"],
                         merged_into=d["merged_into"],
                         n_nodes_by_month=json.dumps(counts, ensure_ascii=False),
                         lifetime=len(counts)))  # id не воскресают → цепочка подряд
    df = pd.DataFrame(rows).astype({"parent_id": "Int64", "merged_into": "Int64"})
    return zm, df


# ---------------------------------------------------------------- smoothing

def smooth_labels(z: np.ndarray, window: int = 3) -> np.ndarray:
    """Majority vote в центрированном окне (на краях усечённом). Метки уже
    глобальные (реестр) — приведение не нужно. Ничья → текущая метка узла
    (детерминированно), среди прочих равных — меньший type_id. Публичные
    заявления о переходах — только по выходу этой функции (спека §10.4)."""
    z = np.asarray(z, dtype=np.int64)
    T, n = z.shape
    ids = np.unique(z)
    zp = np.searchsorted(ids, z)
    out, ar, half = z.copy(), np.arange(n), window // 2
    for t in range(T):
        sl = zp[max(0, t - half): min(T, t + half + 1)]
        cnt = np.zeros((len(ids), n), dtype=np.int64)
        np.add.at(cnt, (sl.ravel(), np.tile(ar, sl.shape[0])), 1)
        mx = cnt.max(0)
        win = cnt.argmax(0)                      # ничья → меньший type_id
        keep = cnt[zp[t], ar] == mx              # ничья с текущей → оставить
        win[keep] = zp[t][keep]
        out[t] = ids[win]
    return out


def flicker_share(z: np.ndarray) -> float:
    """Доля «морганий» A≠B=C по внутренним (узел, месяц) — бенч 10.8% (§9)."""
    z = np.asarray(z)
    if z.shape[0] < 3:
        return 0.0
    a, b, c = z[:-2], z[1:-1], z[2:]
    return float(np.mean((a == c) & (b != a)))


def transition_share(z: np.ndarray) -> float:
    """Средняя по парам доля узлов со сменой метки (бенч 22.7% → 13.1%, §9)."""
    z = np.asarray(z)
    return float(np.mean(z[1:] != z[:-1]))


# ---------------------------------------------------------------- стабильность

def perturb_graph(A: sp.csr_matrix, frac: float,
                  rng: np.random.Generator) -> sp.csr_matrix:
    """Нуль-модель §5: удалить frac рёбер + добавить столько же случайных
    отсутствующих пар с весом = медиана весов (без веса leidenalg падает —
    ловушка, подтверждённая в бенче). Граф симметричный, верхний треугольник."""
    up = sp.triu(A, k=1).tocoo()
    m = up.nnz
    k = int(round(frac * m))
    keep = np.ones(m, dtype=bool)
    keep[rng.choice(m, k, replace=False)] = False
    existing = set(zip(up.row.tolist(), up.col.tolist()))
    med = float(np.median(up.data))
    add_r, add_c = [], []
    while len(add_r) < k:
        i, j = int(rng.integers(A.shape[0])), int(rng.integers(A.shape[0]))
        a, b = (i, j) if i < j else (j, i)
        if a == b or (a, b) in existing:
            continue
        existing.add((a, b))
        add_r.append(a)
        add_c.append(b)
    rows = np.concatenate([up.row[keep], add_r])
    cols = np.concatenate([up.col[keep], add_c])
    data = np.concatenate([up.data[keep], np.full(k, med)])
    B = sp.csr_matrix((data, (rows, cols)), shape=A.shape, dtype=np.float64)
    return B + B.T  # диагональ нулевая — удвоения нет


def _iqr(x: np.ndarray) -> float:
    return float(np.percentile(x, 75) - np.percentile(x, 25))


def stability_report(snaps: list[sp.csr_matrix], z: np.ndarray,
                     seed_ari: np.ndarray, z_single: np.ndarray,
                     cfg: Config, months: list[str]) -> dict:
    """Тройка ARI на каждую пару месяцев (§5): ari_cross консенсусов,
    seed-null (пул 2R ARI прогонов к консенсусу), edge-perturb null
    (пул 2×PERTURB_REPS, база — одиночный прогон seed[0] снимка, Leiden тем же
    seed — как в бенче 0.605; дополнительно perturb-vs-консенсус — бенч 0.781).
    Критерий §5 дословно: эволюция реальна, если ari_cross < ari_perturb_med
    (месяцы различаются СИЛЬНЕЕ шума метода; бенч 0.519 < 0.605 «выполнено»).
    Нарушение — ari_cross >= ari_perturb_med (различие месяцев ниже шумового
    порога) → flagged: такие пары в событийный слой не допускаются."""
    gamma = cfg.cluster.leiden.gamma
    base = stage_seed(cfg.seed, "dynamics")
    T, n = z.shape
    p_single, p_cons = [], []
    for t, A in enumerate(snaps):
        seed0 = _snapshot_seeds(cfg, t)[0]
        vs, vc = [], []
        for rep in range(PERTURB_REPS):
            rng = np.random.default_rng(stage_seed(base, f"perturb:{t:02d}:{rep:02d}"))
            lab = run_leiden(perturb_graph(A, PERTURB_FRAC, rng), gamma=gamma,
                             seed=seed0)
            vs.append(ari(z_single[t], lab))
            vc.append(ari(z[t], lab))
        p_single.append(vs)
        p_cons.append(vc)
        log.info("пертурб-нуль снимка %d/%d: ARI med=%.3f (vs консенсус %.3f)",
                 t + 1, T, float(np.median(vs)), float(np.median(vc)))
    pairs = []
    for t in range(T - 1):
        cross = ari(z[t], z[t + 1])
        pool_s = np.concatenate([seed_ari[t], seed_ari[t + 1]])
        pool_p = np.concatenate([p_single[t], p_single[t + 1]])
        pool_c = np.concatenate([p_cons[t], p_cons[t + 1]])
        pmed = float(np.median(pool_p))
        pairs.append(dict(
            month_from=months[t], month_to=months[t + 1], ari_cross=float(cross),
            ari_seed_med=float(np.median(pool_s)), ari_seed_iqr=_iqr(pool_s),
            ari_perturb_med=pmed, ari_perturb_iqr=_iqr(pool_p),
            ari_perturb_cons_med=float(np.median(pool_c)), n_common=int(n),
            flagged=bool(cross >= pmed)))
    return {
        "pairs": pairs,
        "protocol": {
            "gamma": gamma,
            "seeds_per_snapshot": cfg.cluster.leiden.seeds_per_snapshot,
            "perturb_frac": PERTURB_FRAC, "perturb_reps": PERTURB_REPS,
            "rule": "flagged = ari_cross >= ari_perturb_med (нарушение критерия "
                    "§5 «эволюция реальна, если ari_cross < ari_perturb_med»); "
                    "perturb: −5% рёбер +столько же случайных пар с медианным "
                    "весом, Leiden seed[0] снимка, база — одиночный прогон",
        },
        "summary": {
            "ari_cross_mean": float(np.mean([p["ari_cross"] for p in pairs])),
            "ari_perturb_med_mean": float(np.mean([p["ari_perturb_med"] for p in pairs])),
            "n_flagged": int(sum(p["flagged"] for p in pairs)),
        },
    }


# ---------------------------------------------------------------- переходы

def _transitions(z: np.ndarray, times: list[str],
                 pairs_idx: list[tuple[int, int]],
                 col_a: str, col_b: str) -> pd.DataFrame:
    """Длинный формат K×K матриц: type_from, type_to, n_movers, n_base
    (знаменатель — узлы type_from, присутствующие в обоих месяцах; у нас —
    все). Нулевые ячейки тоже пишем — матрица восстанавливается дословно."""
    rows = []
    for a, b in pairs_idx:
        ids_a, ids_b = np.unique(z[a]), np.unique(z[b])
        _, inter = _jaccard(z[a], z[b], ids_a, ids_b)
        base = inter.sum(1).astype(np.int64)
        for i, ta in enumerate(ids_a.tolist()):
            for j, tb in enumerate(ids_b.tolist()):
                rows.append((times[a], times[b], int(ta), int(tb),
                             int(inter[i, j]), int(base[i])))
    return pd.DataFrame(rows, columns=[col_a, col_b, "type_from", "type_to",
                                       "n_movers", "n_base"])


def transitions_monthly(z: np.ndarray, months: list[str]) -> pd.DataFrame:
    return _transitions(z, months, [(t, t + 1) for t in range(len(months) - 1)],
                        "month_from", "month_to")


def _quarter(m: str) -> str:
    y, mm = m.split("-")
    return f"{y}Q{(int(mm) - 1) // 3 + 1}"


def transitions_quarterly(z: np.ndarray, months: list[str]) -> pd.DataFrame:
    """Переходы через границы кварталов (последний месяц Q → первый Q+1):
    7 пар для 24 месяцев. Расхождение с контрактом §7 («4 среза») зафиксировано:
    24 мес = 8 кварталов, граничные пары — единственные межквартальные стыки."""
    qlabels = [_quarter(m) for m in months]
    pairs = [(t, t + 1) for t in range(len(months) - 1)
             if qlabels[t] != qlabels[t + 1]]
    return _transitions(z, qlabels, pairs, "quarter_from", "quarter_to")


def p_bar(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """P̄ — взвешенное среднее P_t с весами n_t (число общих узлов пары);
    строки с нулевой базой → NaN и из усреднения исключаются (спека §6.1)."""
    ids = np.unique(z)
    K, n = len(ids), z.shape[1]
    num, den = np.zeros((K, K)), np.zeros(K)
    for t in range(z.shape[0] - 1):
        _, inter = _jaccard(z[t], z[t + 1], ids, ids)
        baser = inter.sum(1)
        mask = baser > 0
        P = np.divide(inter, baser[:, None],
                      out=np.full_like(inter, np.nan), where=mask[:, None])
        num[mask] += n * P[mask]
        den[mask] += n
    Pbar = np.divide(num, den[:, None], out=np.full((K, K), np.nan),
                     where=den[:, None] > 0)
    return ids, Pbar


# ---------------------------------------------------------------- стационарность

def stationary(Pbar: np.ndarray, ids: np.ndarray, alpha: float = PAGERANK_ALPHA):
    """π — левый собственный вектор P̄ (NaN-строки выкинуты, row-renorm);
    неэргодичность (нулевые столбцы-«ловушки» от birth/death, абсорбирующие
    состояния, сингулярная фундаментальная матрица, нулевой π) → PageRank-
    регуляризация α=0.99 с униформ-рестартом, факт фиксируется (спека §6.2).
    MFPT: Z = (I − P + 1πᵀ)⁻¹, M_ij = (Z_jj − Z_ij)/π_j, M_ii = 1/π_i.
    Возвращает (ids2, π, M, regularized)."""
    ok = ~np.isnan(Pbar).all(1)
    ids2 = ids[ok]
    P = Pbar[np.ix_(ok, ok)].copy()
    P /= P.sum(1, keepdims=True)

    def _solve(Q: np.ndarray) -> np.ndarray:
        w, V = np.linalg.eig(Q.T)
        pi = np.abs(V[:, int(np.argmax(w.real))].real)
        return pi / pi.sum()

    def _fund(Q: np.ndarray, pi: np.ndarray) -> np.ndarray | None:
        try:
            return np.linalg.inv(np.eye(len(Q)) - Q + np.outer(np.ones(len(Q)), pi))
        except np.linalg.LinAlgError:
            return None

    pi = _solve(P)
    Z = _fund(P, pi)
    regularized = bool((P.sum(0) == 0).any())
    if regularized or Z is None or (pi <= 1e-12).any():
        regularized = True
        P = alpha * P + (1.0 - alpha) / len(P)
        pi = _solve(P)
        Z = _fund(P, pi)
    M = (Z.diagonal()[None, :] - Z) / pi[None, :]
    np.fill_diagonal(M, 1.0 / pi)
    return ids2, pi, M, regularized


def goodman_order_test(z: np.ndarray) -> float | None:
    """χ²-тест Гудмана: 1-й vs 2-й порядок цепи по траекториям узлов (§6.2).
    LR = 2·Σ n_ijk·ln[(n_ijk/n_ij·)/(n_·jk/n_·j·)], df = K(K−1)². Значимый →
    «переходы имеют память» — одна фраза в отчёте, модель не усложняем."""
    z = np.asarray(z)
    T, _ = z.shape
    ids = np.unique(z)
    K = len(ids)
    if T < 3 or K < 2:
        return None
    zp = np.searchsorted(ids, z)
    i, j, k = zp[:-2].ravel(), zp[1:-1].ravel(), zp[2:].ravel()
    n3 = np.bincount((i * K + j) * K + k, minlength=K**3).reshape(K, K, K).astype(float)
    nij, njk = n3.sum(2, keepdims=True), n3.sum(0, keepdims=True)
    nj = n3.sum((0, 2), keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (n3 / nij) / (njk / nj)
    mask = (n3 > 0) & np.isfinite(ratio) & (ratio > 0)
    lr = float(2.0 * (n3[mask] * np.log(ratio[mask])).sum())
    return float(chi2.sf(lr, K * (K - 1) ** 2))


# ---------------------------------------------------------------- spatial Markov

def spatial_markov(z: np.ndarray, knn: pd.DataFrame, nidx: pd.DataFrame, *,
                   seed: int, B: int = BOOTSTRAP_B, frac: float = BOOTSTRAP_FRAC,
                   min_share: float = CONTEXT_MIN_SHARE) -> dict:
    """Условные переходы по контексту соседей (§6.4): соседи = дорожный kNN из
    edges_highway_knn.parquet (как есть, включая island-fallback); контекст
    c(i,t) = доминирующий тип соседей в t при доле ≥ 0.5, иначе «mixed» —
    исключается из сравнений. OR(i,j) = P(i→j | c=j) / P(i→j | c≠j),
    CI — node bootstrap (B=50, сабсемпл 80% без возвращения)."""
    nidx = nidx.sort_values("row_idx").reset_index(drop=True)
    tid2pos = {int(t): p for p, t in enumerate(nidx["territory_id"])}
    n, T = len(nidx), z.shape[0]
    neigh: list[np.ndarray] = [np.zeros(0, dtype=np.int64) for _ in range(n)]
    for tx, grp in knn.groupby("tid_x"):
        p = tid2pos.get(int(tx))
        if p is not None:
            neigh[p] = np.asarray([tid2pos[int(v)] for v in grp["tid_y"]
                                   if int(v) in tid2pos], dtype=np.int64)
    coverage = float(np.mean([v.size > 0 for v in neigh]))
    ids = np.unique(z)
    K, maxid = len(ids), int(ids.max()) + 1
    oa, ob, oc, on = [], [], [], []   # наблюдения: from, to, контекст, узел
    for t in range(T - 1):
        lab = z[t]
        for i in range(n):
            nb = neigh[i]
            if nb.size == 0:
                continue
            cnt = np.bincount(lab[nb], minlength=maxid)
            top = int(cnt.argmax())   # ничья → меньший type_id
            if cnt[top] / nb.size < min_share:
                continue              # mixed
            oa.append(lab[i])
            ob.append(z[t + 1, i])
            oc.append(top)
            on.append(i)
    oa, ob, oc, on = (np.asarray(v, dtype=np.int64) for v in (oa, ob, oc, on))

    ctx_mat = {}
    for c in ids.tolist():
        m = oc == c
        M = np.full((K, K), np.nan)
        if m.any():
            pa, pb = np.searchsorted(ids, oa[m]), np.searchsorted(ids, ob[m])
            C = np.bincount(pa * K + pb, minlength=K * K).reshape(K, K).astype(float)
            M = np.divide(C, C.sum(1, keepdims=True), out=M,
                          where=C.sum(1, keepdims=True) > 0)
        ctx_mat[str(c)] = {"matrix": M.tolist(), "n_obs": int(m.sum())}

    def _or(a, b, c, i, j):
        m1, m0 = (a == i) & (c == j), (a == i) & (c != j)
        n1, n0 = int(m1.sum()), int(m0.sum())
        if n1 == 0 or n0 == 0:
            return np.nan, n1, n0
        p1, p0 = float((b[m1] == j).mean()), float((b[m0] == j).mean())
        return (p1 / p0 if p0 > 0 else (np.inf if p1 > 0 else np.nan)), n1, n0

    pairs_ij = [(int(i), int(j)) for i in ids.tolist() for j in ids.tolist() if i != j]
    boot: dict[tuple[int, int], list[float]] = {p: [] for p in pairs_ij}
    for rep in range(B):
        rng = np.random.default_rng(stage_seed(seed, f"spatial_boot:{rep:02d}"))
        mask = np.isin(on, rng.choice(n, int(round(frac * n)), replace=False))
        for p in pairs_ij:
            v, _, _ = _or(oa[mask], ob[mask], oc[mask], *p)
            boot[p].append(v)
    ors = {}
    for p in pairs_ij:
        v, n1, n0 = _or(oa, ob, oc, *p)
        fin = np.asarray([x for x in boot[p] if np.isfinite(x)])
        ci = (np.percentile(fin, [2.5, 97.5]).tolist() if len(fin) >= 10 else None)
        ors[f"{p[0]}->{p[1]}"] = {"or": v, "ci": ci, "n_ctx": n1, "n_other": n0}
    return {"neighbors": "highway kNN из edges_highway_knn.parquet (с island-fallback)",
            "min_share": min_share, "coverage": coverage,
            "type_ids": [int(x) for x in ids], "context_matrices": ctx_mat,
            "or": ors, "bootstrap": {"B": B, "frac": frac}}


# ---------------------------------------------------------------- события

def events_table(z_smooth: np.ndarray, nidx: pd.DataFrame, months: list[str],
                 flagged_pairs=(), panel: pd.DataFrame | None = None) -> pd.DataFrame:
    """МО-переходчики на smoothed-метках (§6.5): month (месяц новой метки),
    territory_id, type_from, type_to, top3_delta_clr — топ-3 CLR-компонента по
    |Δ| за 3 месяца до перехода (clr[t] − clr[t−3]; окно слева обрезано, для
    перехода в первом месяце истории нет → None). Переходы flagged-пар
    (стабильность ниже шума метода) НЕ допускаются (§5)."""
    flagged = set(flagged_pairs)
    X = None
    if panel is not None:
        X = np.stack([panel.pivot(index="territory_id", columns="month", values=c)
                      .reindex(index=nidx["territory_id"].to_numpy(), columns=months)
                      .to_numpy() for c in CLR_COLS])  # (6, n, T)
    rows = []
    for t in range(len(months) - 1):
        if t in flagged:
            continue
        for i in np.flatnonzero(z_smooth[t] != z_smooth[t + 1]).tolist():
            top3 = None
            if X is not None and t > 0:
                d = X[:, i, t] - X[:, i, max(0, t - 3)]
                order = np.argsort(-np.abs(np.nan_to_num(d, nan=0.0)))[:3]
                top3 = "; ".join(f"{CLR_COLS[j]}={d[j]:+.3f}" for j in order)
            rows.append((months[t + 1], int(nidx["territory_id"].iloc[i]),
                         int(z_smooth[t, i]), int(z_smooth[t + 1, i]), top3))
    return pd.DataFrame(rows, columns=["month", "territory_id", "type_from",
                                       "type_to", "top3_delta_clr"])


# ---------------------------------------------------------------- оркестратор

def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return _jsonable(o.tolist())
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return float(o) if np.isfinite(o) else None
    return o


def _write_json(path: Path, obj) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_jsonable(obj), f, ensure_ascii=False, indent=2)


def run_dynamics(graphs_dir: str | Path, knn_path: str | Path,
                 panel_path: str | Path | None, months: list[str], cfg: Config,
                 out_dir: str | Path) -> dict:
    """Полный прогон спеки 29 → контракт §7 (8 файлов) в out_dir.
    transitions_monthly.csv — smoothed (публичная), сырая версия — отдельным
    файлом-приложением (§6.1); events — parquet (задача сборки, не csv из §7)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    months = [str(m) for m in months]
    snaps, nidx = load_snapshots(graphs_dir)
    if len(snaps) != len(months):
        raise ValueError(f"снимков {len(snaps)} != месяцев {len(months)}")
    Z, agree, seed_ari, z_single = labels_from_snapshots(snaps, cfg)
    zm, registry = match_labels(Z, months, tau=cfg.dynamics.tau_inherit)
    zs = smooth_labels(zm, window=cfg.dynamics.smoothing_window)
    stab = stability_report(snaps, Z, seed_ari, z_single, cfg, months)
    flagged = [t for t, p in enumerate(stab["pairs"]) if p["flagged"]]
    ids, Pbar = p_bar(zs)
    ids2, pi, M, regularized = stationary(Pbar, ids)
    order_p = goodman_order_test(zs)
    sm = spatial_markov(zs, pd.read_parquet(knn_path), nidx,
                        seed=stage_seed(cfg.seed, "dynamics"))
    panel = pd.read_parquet(panel_path) if panel_path is not None else None
    ev = events_table(zs, nidx, months, flagged_pairs=flagged, panel=panel)

    labels = pd.concat([pd.DataFrame({
        "territory_id": nidx["territory_id"].to_numpy(), "month": m,
        "type_id_raw": Z[t], "type_id": zm[t], "type_id_smooth": zs[t],
        "seed_agreement": agree[t], "present": True})
        for t, m in enumerate(months)], ignore_index=True)
    labels.to_parquet(out_dir / "labels.parquet", index=False)
    registry.to_parquet(out_dir / "type_registry.parquet", index=False)
    transitions_monthly(zs, months).to_csv(out_dir / "transitions_monthly.csv",
                                           index=False)
    transitions_monthly(zm, months).to_csv(
        out_dir / "transitions_monthly_raw.csv", index=False)  # приложение §6.1
    transitions_quarterly(zs, months).to_csv(out_dir / "transitions_quarterly.csv",
                                             index=False)
    _write_json(out_dir / "stability.json", stab)
    _write_json(out_dir / "spatial_markov.json", sm)
    ev.to_parquet(out_dir / "events.parquet", index=False)
    _write_json(out_dir / "stationary.json", {
        "pi": {str(int(t)): float(p) for t, p in zip(ids2, pi)},
        "mfpt": {str(int(a)): {str(int(b)): M[x, y] for y, b in enumerate(ids2)}
                 for x, a in enumerate(ids2)},
        "regularized": regularized, "order_test_p": order_p,
        "p_bar": {str(int(a)): {str(int(b)): Pbar[x, y] for y, b in enumerate(ids)}
                  for x, a in enumerate(ids)},
        "note": "π — левый собственный вектор P̄ (взвешенной по n_t, §6.1); "
                "при нулевых столбцах — PageRank α=0.99 (§6.2)",
    })
    metrics = {
        "k_by_month": {m: int(len(np.unique(zm[t]))) for t, m in enumerate(months)},
        "k_total_types": int(len(registry)),
        "registry_events": {
            "birth": int(((registry["birth_month"] != months[0])
                          & registry["parent_id"].isna()).sum()),
            "split": int(registry["parent_id"].notna().sum()),
            "death": int(registry["death_month"].notna().sum()),
            "merge": int(registry["merged_into"].notna().sum()),
        },
        "mover_share_raw": transition_share(zm),
        "mover_share_smoothed": transition_share(zs),
        "flicker_raw": flicker_share(zm), "flicker_smoothed": flicker_share(zs),
        "seed_agreement_mean": float(agree.mean()),
        "ari_cross_mean": stab["summary"]["ari_cross_mean"],
        "ari_perturb_med_mean": stab["summary"]["ari_perturb_med_mean"],
        "n_flagged": stab["summary"]["n_flagged"], "n_pairs": len(stab["pairs"]),
        "n_events": int(len(ev)),
        "order_test_p": order_p, "regularized_pi": regularized,
    }
    return {"metrics": metrics}
