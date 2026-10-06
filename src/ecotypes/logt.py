"""Клубы сходимости Phillips–Sul без R: порт ConvergenceClubs 1:1, numpy-only
(research/30_convergence_clubs.md). Семантика эталона (ps_andrews_hac.R,
estimateMod.R, coreG.R, club.R, findClubs.R, mergeClubs.R):
- log-t регрессия: log(H1/Ht) − 2·log(log t) = a + b·log t + u, окно
  t = round(trim·T)+1…T (1-based), H1 — первый период ВСЕГО ряда;
- HAC: QS-ядро Эндрюса, FQSB: ρ̂ = AR(1) остатков, α = 4ρ̂²/(1−ρ̂)⁴,
  B = 1.3221·(α·t)^0.2, k_l = (sin z/z − cos z)·3/z² при z = 1.2πl/B,
  автоковариации с делителем t−1 (как в GAUSS-коде PS — не «чинить»);
- клубы: сортировка по последнему периоду → ядро (первая проходящая пара →
  рост, k* = argmax |t_k| при проходящих меньших префиксах) → сито
  (t(ядро+u) > cstar=0) → рекурсия → слияние PS-2009 (накопительное,
  соседние клубы). mergeDivergent=False: дивергенты — слой left-behind.
Отклонения от R только в вырожденных точках (нулевые остатки, B→0):
R там даёт NaN/падает, мы — детерминированные ±inf / eps-полы."""
from __future__ import annotations

import numpy as np

H_FLOOR = 1e-12  # пол для H_t: идеальная сходимость ломает log (спека п.3)
THRESHOLD = -1.65  # односторонний 5% порог PS


def _H_of(Y: np.ndarray) -> np.ndarray:
    """H_t = (1/n)Σ(h_it − 1)², h_it = y_it/среднее по сечению. Y: (T, n)."""
    h = Y / Y.mean(axis=1, keepdims=True)
    return np.maximum(((h - 1.0) ** 2).mean(axis=1), H_FLOOR)


def _andrews_lrv_batch(U: np.ndarray) -> np.ndarray:
    """LRV остатков по столбцам U (Tw, m). Дословный порт ps_andrews_hac.R."""
    t = U.shape[0]
    x1, x2 = U[:t - 1], U[1:t]
    den = (x1 * x1).sum(axis=0)
    b1 = np.divide((x1 * x2).sum(axis=0), den, out=np.zeros(U.shape[1]), where=den > 0)
    alpha = np.clip(4.0 * b1 * b1 / np.maximum((1.0 - b1) ** 4, 1e-300), 0.0, 1e300)
    B = np.maximum(1.3221 * (alpha * t) ** 0.2, 1e-8)
    l = np.arange(1, t, dtype=np.float64)[:, None]  # lags 1..t-1
    z = 1.2 * np.pi * l / B[None, :]
    k = (np.sin(z) / z - np.cos(z)) * 3.0 / (z * z)
    t2 = t - 1
    xt = U[:t2]  # в автоковариациях GAUSS-код берёт первые t−1 остатков
    out = (U * U).sum(axis=0) / t2  # γ₀ — по всем t остаткам, делитель t−1
    for i in range(1, t2):
        g = (xt[:t2 - i] * xt[i:t2]).sum(axis=0) / t2
        out += 2.0 * k[i - 1] * g
    return out


def _logt_from_Hmat(H: np.ndarray, trim: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """log-t регрессия для m H-рядов (столбцы H: (T, m)) → (beta, se, t)."""
    T = H.shape[0]
    rT = np.arange(int(round(T * trim)) + 1, T + 1)  # 1-based окно (как R seq)
    logt = np.log(rT.astype(np.float64))
    rH = np.log(H[0:1, :] / H[rT - 1, :]) - 2.0 * np.log(logt)[:, None]
    xc = logt - logt.mean()
    sxx = float(xc @ xc)
    rHc = rH - rH.mean(axis=0, keepdims=True)
    b = (xc[:, None] * rHc).sum(axis=0) / sxx
    U = rHc - xc[:, None] * b[None, :]  # остатки (a впитан центрированием)
    lrv = _andrews_lrv_batch(U)
    se = np.sqrt(np.maximum(lrv, 0.0) / sxx)
    t = np.divide(b, se, out=np.sign(b) * np.inf * np.ones_like(b), where=se > 0)
    return b, se, t


def _prefix_H(Ys: np.ndarray) -> np.ndarray:
    """H_t для всех префиксов k=1..n упорядоченных столбцов Ys (T, n) за O(Tn):
    H_t(k) = (Q_t/m_t² − 2S_t/m_t + k)/k, S/Q — cumsum по y и y², m_t = S_t/k."""
    S = np.cumsum(Ys, axis=1)
    Q = np.cumsum(Ys * Ys, axis=1)
    k = np.arange(1, Ys.shape[1] + 1, dtype=np.float64)[None, :]
    m = S / k
    return np.maximum((Q / (m * m) - 2.0 * S / m + k) / k, H_FLOOR)


def logt_test(series: np.ndarray, trim: float = 1.0 / 3.0,
              threshold: float = THRESHOLD) -> dict:
    """Log-t тест PS по панели (T, n): t-стат b̂ и вердикт t > threshold."""
    Y = np.ascontiguousarray(series, dtype=np.float64)
    b, se, t = _logt_from_Hmat(_H_of(Y)[:, None], trim)
    return {"t_stat": float(t[0]), "beta": float(b[0]), "std_err": float(se[0]),
            "converges": bool(t[0] > threshold)}


def _find_core(Ysub: np.ndarray, trim: float, threshold: float) -> list[int] | None:
    """Ядро по coreG.R: первая проходящая соседняя пара → рост префикса,
    k* = argmax |t_k| среди непрерывно проходящих префиксов (type='max')."""
    r = Ysub.shape[1]
    Y1, Y2 = Ysub[:, :-1], Ysub[:, 1:]
    m = (Y1 + Y2) / 2.0
    Hp = np.maximum((((Y1 / m - 1.0) ** 2 + (Y2 / m - 1.0) ** 2) / 2.0), H_FLOOR)
    tp = _logt_from_Hmat(Hp, trim)[2]
    ok = np.nonzero(tp > threshold)[0]
    if len(ok) == 0:
        return None
    s = int(ok[0])
    if s == r - 2:
        return [s, s + 1]
    ts = _logt_from_Hmat(_prefix_H(Ysub[:, s:]), trim)[2][1:]  # длины 2..r−s
    passing = []
    for j in range(len(ts)):
        if ts[j] > threshold:
            passing.append(j)
        else:
            break
    best = max(passing, key=lambda j: abs(ts[j]))
    return list(range(s, s + best + 2))


def _sieve_tstats(Ysub: np.ndarray, core: list[int], cand: list[int],
                  trim: float) -> np.ndarray:
    """t(ядро + u) для каждого кандидата u за O(T·|cand|): меняется только
    m_u = (S_c + y_u)/k в разложении Σ(y/m − 1)² = Q/m² − 2S/m + k."""
    k = len(core) + 1
    S_c = Ysub[:, core].sum(axis=1)
    Q_c = (Ysub[:, core] ** 2).sum(axis=1)
    Yd = Ysub[:, cand]
    m = (S_c[:, None] + Yd) / k
    H = ((Q_c[:, None] / (m * m) - 2.0 * S_c[:, None] / m + len(core))
         + (Yd / m - 1.0) ** 2) / k
    return _logt_from_Hmat(np.maximum(H, H_FLOOR), trim)[2]


def _logt_subset(Y: np.ndarray, idx: list[int], trim: float) -> float:
    return float(_logt_from_Hmat(_H_of(Y[:, idx])[:, None], trim)[2][0])


def _merge_ps2009(Y: np.ndarray, clubs: list[list[int]], trim: float,
                  threshold: float, merge_log: list[dict]) -> list[list[int]]:
    """Слияние PS-2009 (mergeClubs.R, method='PS'): накопительный обход
    соседних клубов; сливаем при t > threshold, после отказа рестарт
    со следующего клуба. merge_log получает каждую попытку."""
    ll = len(clubs)
    if ll < 2:
        return clubs
    out, i = [], 0
    while i < ll - 1:
        units, append_last, k = list(clubs[i]), False, i + 1
        while k < ll:
            t = _logt_subset(Y, units + clubs[k], trim)
            merged = t > threshold
            merge_log.append({"left_size": len(units), "right_size": len(clubs[k]),
                              "t_stat": t, "merged": bool(merged)})
            if merged:
                units += clubs[k]
                k += 1
            else:
                if k == ll - 1:
                    append_last = True
                break
        i = k
        out.append(units)
        if append_last:
            out.append(list(clubs[ll - 1]))
    return out


def club_classification(series: np.ndarray, trim: float = 1.0 / 3.0,
                        threshold: float = THRESHOLD, cstar: float = 0.0,
                        do_merge: bool = True) -> dict:
    """Классификация PS (findClubs + mergeClubs): clubs — метка клуба на
    единицу (1..K, −1 = дивергент), divergent — индексы дивергентов,
    merge_log — журнал слияний + попарная merge-матрица финальных клубов."""
    Y = np.ascontiguousarray(series, dtype=np.float64)
    T, n = Y.shape
    if T < 3 or n < 2:
        raise ValueError("нужны T>=3 и n>=2")
    remaining = np.argsort(-Y[T - 1], kind="stable").tolist()  # убывание последнего
    clubs: list[list[int]] = []
    divergent: list[int] = []
    while remaining:
        if len(remaining) == 1:
            divergent.append(remaining.pop())
            break
        Ysub = Y[:, remaining]
        if _logt_subset(Y, remaining, trim) > threshold:
            clubs.append(list(remaining))
            remaining = []
            break
        core = _find_core(Ysub, trim, threshold)
        if core is None:
            divergent.extend(remaining)
            remaining = []
            break
        core_set = set(core)
        cand = [p for p in range(len(remaining)) if p not in core_set]
        members = list(core)
        if cand:
            tc = _sieve_tstats(Ysub, core, cand, trim)
            members += [cand[j] for j in range(len(cand)) if tc[j] > cstar]
        mset = set(members)
        clubs.append([remaining[p] for p in sorted(members)])
        remaining = [remaining[p] for p in range(len(remaining)) if p not in mset]
    merge_log: list[dict] = []
    if do_merge:
        clubs = _merge_ps2009(Y, clubs, trim, threshold, merge_log)
    labels = np.full(n, -1, dtype=np.int64)
    for cid, members in enumerate(clubs, start=1):
        labels[np.asarray(members, dtype=np.int64)] = cid
    club_t = [_logt_subset(Y, c, trim) for c in clubs]
    K = len(clubs)
    pair_t = np.full((K, K), np.nan)
    for a in range(K):
        for b in range(a + 1, K):
            pair_t[a, b] = pair_t[b, a] = _logt_subset(Y, clubs[a] + clubs[b], trim)
    merge_log.append({"pairwise_t_final": pair_t.tolist()})
    return {"clubs": labels, "divergent": np.asarray(sorted(divergent), dtype=np.int64),
            "merge_log": merge_log, "n_clubs": K,
            "club_sizes": [len(c) for c in clubs], "club_tstats": club_t}


def sigma_convergence(series: np.ndarray) -> dict:
    """σ-критерий: CV_t по сечению + тренд log H_t на t (наклон/мес, OLS)."""
    Y = np.ascontiguousarray(series, dtype=np.float64)
    cv = Y.std(axis=1, ddof=1) / Y.mean(axis=1)
    H = _H_of(Y)
    x = np.arange(1, len(H) + 1, dtype=np.float64)
    xc = x - x.mean()
    ly = np.log(H)
    slope = float(xc @ (ly - ly.mean()) / (xc @ xc))
    resid = ly - (ly.mean() + slope * xc)
    se = float(np.sqrt(resid @ resid / (len(x) - 2) / (xc @ xc)))
    return {"cv_t": cv, "trend": slope, "trend_t": slope / se if se > 0 else np.inf,
            "cv_first": float(cv[0]), "cv_last": float(cv[-1]),
            "converges": bool(slope < 0)}


def beta_convergence(series: np.ndarray) -> dict:
    """β-критерий: рост log(y_T/y_1)/(T−1) на log y_1, OLS по сечению."""
    Y = np.ascontiguousarray(series, dtype=np.float64)
    T, n = Y.shape
    g = np.log(Y[T - 1] / Y[0]) / (T - 1)
    x = np.log(Y[0])
    xc = x - x.mean()
    beta = float(xc @ (g - g.mean()) / (xc @ xc))
    resid = g - (g.mean() + beta * xc)
    se = float(np.sqrt(resid @ resid / (n - 2) / (xc @ xc)))
    return {"beta": beta, "t": beta / se if se > 0 else np.inf,
            "converges": bool(beta < 0)}


def verdict_2of3(logt_res: dict, sigma_res: dict, beta_res: dict) -> dict:
    """Правило «≥2 из 3» (спека п.5): вывод при согласии минимум двух критериев."""
    votes = {"logt": bool(logt_res["converges"]), "sigma": bool(sigma_res["converges"]),
             "beta": bool(beta_res["converges"])}
    n = sum(votes.values())
    return {"votes": votes, "n_votes": n, "verdict": "convergence" if n >= 2 else "divergence"}
