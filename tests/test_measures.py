"""Тесты модуля мер сходства (research/24). Контракты:
(a) M1–M10 на mini_panel (60 МО) → валидный граф (симметрия, без самопетель,
    разумные степени); (б) X1/X2/X3 вырождаются на синтетике с общим фактором,
    M2 — нет; (в) гейт детерминирован и возвращает вердикт; (г) SNF сходится;
    (д) M9: α=0 ≡ M2, α=1 ≡ M5 по kNN-спискам; (е) DTW: известный сдвиг во
    времени монотонно увеличивает дистанцию."""
from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from ecotypes import measures as ms

K = 10


def _mini_data(pp: dict) -> ms.MeasureData:
    """MeasureData из PP-Dir мини-панели: V → уровни, X → доли/CLR, geo →
    псевдо-highway kNN, pop = средний объём."""
    X, V, geo = pp["X"], pp["V"], pp["geo"]
    T, n, _ = X.shape
    log_all = np.log(V.T)                                # (n, T)
    sa_all = ms.sa_from_log(log_all)
    shares = np.transpose(X, (1, 0, 2))                  # (n, T, 6)
    clr = ms.clr_transform(shares)
    vals = shares[:, :, :5] * V.T[:, :, None]            # (n, T, 5)
    pop = V.mean(axis=0)

    # псевдо-highway: kNN(10) по латентным координатам, симметризованный
    G = (geo * geo).sum(axis=1)
    d2 = np.clip(G[:, None] + G[None, :] - 2.0 * geo @ geo.T, 0.0, None)
    np.fill_diagonal(d2, np.inf)
    kk = min(K, n - 1)
    nbr = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
    src = np.repeat(np.arange(n), kk)
    dst = nbr.ravel()
    ei = np.concatenate([src, dst]).astype(np.int32)
    ej = np.concatenate([dst, src]).astype(np.int32)
    dist = np.concatenate([np.sqrt(d2[src, dst]), np.sqrt(d2[src, dst])])
    region = (np.arange(n) % 6).astype(np.int8)
    return ms.MeasureData(tids=np.arange(n, dtype=np.int32), log_all=log_all,
                          sa_all=sa_all, shares=shares, clr=clr, vals=vals,
                          pop=pop, hw=(ei, ej, dist), region=region)


@pytest.fixture(scope="module")
def mini_data(mini_panel) -> ms.MeasureData:
    return _mini_data(mini_panel)


def _common_factor_data(n: int = 200, T: int = 24, seed: int = 0) -> ms.MeasureData:
    """Синтетика с доминирующим общим фактором: log = const + f(t) + ε,
    sd шага f = 0.3, sd idio-шума = 0.01 → уровни и сырые приросты вырождены,
    рыночные остатки чисты."""
    rng = np.random.default_rng(seed)
    f = np.cumsum(rng.normal(0.0, 0.3, T))
    log_all = 10.0 + f[None, :] + rng.normal(0.0, 0.01, (n, T))
    base = np.array([0.44, 0.05, 0.14, 0.03, 0.06, 0.28])
    shares = np.clip(base[None, None, :] + rng.normal(0, 1e-4, (n, T, 6)), 1e-6, None)
    shares /= shares.sum(axis=2, keepdims=True)
    return ms.MeasureData(
        tids=np.arange(n, dtype=np.int32), log_all=log_all,
        sa_all=ms.sa_from_log(log_all), shares=shares, clr=ms.clr_transform(shares),
        vals=shares[:, :, :5] * 1000.0, pop=None, hw=None, region=None)


def _graph_checks(edge_index: np.ndarray, weights: np.ndarray, n: int) -> None:
    E = len(edge_index)
    assert E > 0
    assert (edge_index[:, 0] < edge_index[:, 1]).all(), "нужны уникальные пары i<j"
    assert weights.shape == (E,)
    assert np.isfinite(weights).all()
    A = sp.csr_matrix((weights, (edge_index[:, 0], edge_index[:, 1])), shape=(n, n))
    A = A + A.T
    deg = np.diff(A.indptr)
    assert deg.min() >= 1, "есть изоляты"
    assert deg.max() < n, "самопетли или дубли"
    assert 8 <= deg.mean() <= 40, f"средняя степень {deg.mean():.1f} вне разумного"


# (a) валидность графов M1–M10 на мини-панели ---------------------------------------

def test_all_measures_valid_graph(mini_data):
    fns = {
        "M1": ms.sim_M1, "M2": ms.sim_M2, "M3": ms.sim_M3, "M4": ms.sim_M4,
        "M5": ms.sim_M5, "M6": ms.sim_M6, "M7": ms.sim_M7, "M8": ms.sim_M8,
        "M9": lambda d, **kw: ms.sim_M9(d, alpha=0.5),
        "M10": lambda d, **kw: ms.sim_M10(d, k=K),
    }
    for name, fn in fns.items():
        res = fn(mini_data)
        ei, w = ms.knn_union_graph(res, K)
        _graph_checks(ei, w, mini_data.n)


def test_m4_leadlag_map(mini_data):
    res = ms.sim_M4(mini_data)
    ll = res.extra["lead_lag"]
    assert ll.shape == (mini_data.n, mini_data.n)
    assert set(np.unique(ll)) <= set(range(-3, 4))
    assert np.all(res.matrix >= 0.0) and np.all(res.matrix <= 1.0 + 1e-9)


def test_m4_leadlag_sign():
    """lag_map[i,j] = ℓ ⟺ j лидирует i на ℓ мес. Узел 1 = узел 0, сдвинутый на
    +2 (последователь) → lag_map[0,1] = −2 (0 лидирует 1), lag_map[1,0] = +2."""
    rng = np.random.default_rng(0)
    n, T = 5, 24
    d0 = rng.normal(0.0, 1.0, T - 1)
    sa = np.zeros((n, T))
    sa[0, 1:] = np.cumsum(d0)                       # лидер
    sa[1, 2:] = sa[0, :-2]                          # последователь: lag 2
    sa[2:, :] = rng.normal(0.0, 0.01, (n - 2, T))   # фоновый шум
    data = ms.MeasureData(
        tids=np.arange(n, dtype=np.int32), log_all=sa.copy(), sa_all=sa,
        shares=np.full((n, T, 6), 1 / 6), clr=np.zeros((n, T, 6)),
        vals=np.ones((n, T, 5)))
    ll = ms.sim_M4(data).extra["lead_lag"]
    assert ll[0, 1] == -2, f"i=0 лидирует j=1 на 2 → ожидалось −2, got {ll[0, 1]}"
    assert ll[1, 0] == 2, f"j=0 лидирует i=1 на 2 → ожидалось +2, got {ll[1, 0]}"


def test_sa_split_half_negation_identity():
    """Документация вырождения первичной T-ноги: при T=24 и month-of-year sa
    (по 2 точки на календарный месяц) d[12+j] ≡ −d[j] — corr-меры половин
    совпадают точно, split-half Jaccard = 1.0. Поэтому M9 α — по odd/even."""
    rng = np.random.default_rng(1)
    log_all = np.cumsum(rng.normal(0, 1, (30, 24)), axis=1)
    sa = ms.sa_from_log(log_all)
    assert np.allclose(sa[:, 12:], -sa[:, :12])
    d = np.diff(sa, axis=1)
    assert np.allclose(d[:, 12:], -d[:, :11])


def test_m9_alpha_oddeven_not_degenerate(mini_data):
    """odd/even-половины не имеют тождества негации → Jaccard < 1 хотя бы для
    части решётки, α из решётки, выбор детерминирован."""
    grid = [0.0, 0.25, 0.5, 0.75, 1.0]
    a1, js1 = ms.m9_alpha_split_half(mini_data, grid, K)
    a2, js2 = ms.m9_alpha_split_half(mini_data, grid, K)
    assert a1 == a2 and js1 == js2
    assert a1 in grid and set(js1) == set(grid)
    assert all(0.0 <= j <= 1.0 for j in js1.values())
    assert min(js1.values()) < 1.0, "odd/even вырожден — критерий снова пуст"


# (б) анти-примеры вырождаются, M2 — нет ---------------------------------------------

def test_anti_examples_degenerate():
    data = _common_factor_data()
    iu = np.triu_indices(data.n, 1)
    for name, fn in [("X1", ms.sim_X1), ("X2", ms.sim_X2), ("X3", ms.sim_X3)]:
        m = float(fn(data).matrix[iu].mean())
        assert m > 0.85, f"{name}: mean sim {m:.3f} — ожидалось вырождение > 0.85"
    m2 = float(ms.sim_M2(data).matrix[iu].mean())
    assert abs(m2) < 0.15, f"M2: mean r {m2:.3f} — рыночный фактор не убран"


# (в) гейт детерминирован и возвращает вердикт ----------------------------------------

def test_gate_deterministic():
    data = _common_factor_data(n=40, seed=1)
    a = ms.bootstrap_jaccard(ms.sim_M2, data, k=5, B=3, seed=7)
    b = ms.bootstrap_jaccard(ms.sim_M2, data, k=5, B=3, seed=7)
    assert a[0] == b[0] and a[1] == b[1] and (a[2] == b[2]).all()
    assert ms.gate_verdict(a[0], 0.25) in ("pass", "fail")
    assert ms.gate_verdict(0.3, 0.25) == "pass"
    assert ms.gate_verdict(0.2, 0.25) == "fail"
    # статическая мера не зависит от бутстрепа рядов → J = 1 (24 §4.6)
    d2 = _mini_data_static(data.n)
    j_static, _, _ = ms.bootstrap_jaccard(ms.sim_M8, d2, k=5, B=2, seed=3)
    assert j_static == 1.0


def _mini_data_static(n: int) -> ms.MeasureData:
    rng = np.random.default_rng(0)
    pos = rng.uniform(0, 1, (n, 2))
    G = (pos * pos).sum(1)
    d2 = np.clip(G[:, None] + G[None, :] - 2 * pos @ pos.T, 0, None)
    np.fill_diagonal(d2, np.inf)
    kk = 5
    nbr = np.argpartition(d2, kth=kk - 1, axis=1)[:, :kk]
    src = np.repeat(np.arange(n), kk)
    ei = np.concatenate([src, nbr.ravel()]).astype(np.int32)
    ej = np.concatenate([nbr.ravel(), src]).astype(np.int32)
    dist = np.concatenate([np.sqrt(d2[src, nbr.ravel()])] * 2)
    T = 24
    log_all = rng.normal(0, 1, (n, T))
    return ms.MeasureData(tids=np.arange(n, dtype=np.int32), log_all=log_all,
                          sa_all=ms.sa_from_log(log_all),
                          shares=np.full((n, T, 6), 1 / 6),
                          clr=np.zeros((n, T, 6)), vals=np.ones((n, T, 5)),
                          pop=rng.lognormal(0, 1, n), hw=(ei, ej, dist), region=None)


# (г) SNF сходится ---------------------------------------------------------------------

def test_snf_converges():
    rng = np.random.default_rng(0)
    n = 40
    mats = []
    for _ in range(3):
        M = rng.uniform(0, 1, (n, n))
        M = (M + M.T) / 2
        np.fill_diagonal(M, 1.0)
        mats.append(M)
    fused, errs = ms.snf(mats, k=5, t=20)
    assert len(errs) == 20
    assert errs[-1] < errs[0], "ошибка не убывает"
    assert errs[-1] < 1e-3, "не сошлось за 20 итераций"
    assert np.all(fused >= 0.0) and np.all(np.diag(fused) == 0.0)
    assert np.allclose(fused, fused.T)


# (д) M9: α=0 ≡ M2, α=1 ≡ M5 -------------------------------------------------------------

def test_m9_alpha_endpoints(mini_data):
    l_m2 = ms.knn_lists(ms.sim_M2(mini_data), K)
    l_m5 = ms.knn_lists(ms.sim_M5(mini_data), K)
    l_a0 = ms.knn_lists(ms.sim_M9(mini_data, alpha=0.0), K)
    l_a1 = ms.knn_lists(ms.sim_M9(mini_data, alpha=1.0), K)
    assert ms.jaccard_lists(l_a0, l_m2) == 1.0
    assert ms.jaccard_lists(l_a1, l_m5) == 1.0


# (е) DTW: известный сдвиг монотонно увеличивает дистанцию --------------------------------

def test_dtw_monotone_in_shift():
    T = 16
    t = np.arange(T)
    base = np.sin(2 * np.pi * t / T)
    dists = []
    for s in (1, 2, 3):
        shifted = np.sin(2 * np.pi * (t - s) / T)
        X = np.vstack([base, shifted])
        D = ms.dtw_matrix(np.ascontiguousarray(X, dtype=np.float64), w=3)
        dists.append(D[0, 1])
    assert dists[0] < dists[1] < dists[2], f"DTW не монотонен по сдвигу: {dists}"


def test_dtwd_monotone_in_shift():
    T = 16
    t = np.arange(T)
    base = np.column_stack([np.sin(2 * np.pi * t / T), np.cos(2 * np.pi * t / T)])
    dists = []
    for s in (1, 2, 3):
        shifted = np.column_stack([np.sin(2 * np.pi * (t - s) / T),
                                   np.cos(2 * np.pi * (t - s) / T)])
        X = np.stack([base, shifted])
        D = ms.dtwd_matrix(np.ascontiguousarray(X, dtype=np.float64), w=3)
        dists.append(D[0, 1])
    assert dists[0] < dists[1] < dists[2], f"DTW-D не монотонен по сдвигу: {dists}"


def test_dtw_window_respected():
    # сдвиг больше окна не может быть «разwarpлен»: дистанция > 0 даже при w=3
    T = 16
    t = np.arange(T)
    a = np.sin(2 * np.pi * t / T)
    b = np.sin(2 * np.pi * (t - 8) / T)
    X = np.ascontiguousarray(np.vstack([a, b]), dtype=np.float64)
    assert ms.dtw_matrix(X, w=3)[0, 1] > 1.0
