"""Тесты лаг-лидерства (research/22, модуль ecotypes.laglead). Контракты:
(а) синтетика с известным лагом k=2, SNR≥3 → точное восстановление ≥0.9 на 100 парах;
(б) знак лага не инвертирован: lag_map[i,j] = ℓ > 0 ⟺ j лидирует i (как sim_M4);
(в) суррогатный нуль: на независимых AR(1) доля FDR-открытий ≤ 5% ± 2%;
(г) phase-randomization сохраняет спектр точно и mean-ACF, а circular-shift как
    перестановочный тест структурно бессилен (p_min ≈ 0.04, малые сдвиги
    сохраняют истинное выравнивание → нуль загрязнён);
(д) детерминизм по seed.
"""
from __future__ import annotations

import numpy as np
import pytest

from ecotypes import laglead as ll

T = 23  # как в данных: 24 уровня → 23 прироста


def _ar1(rng: np.random.Generator, rho: float, n: int) -> np.ndarray:
    e = rng.normal(0.0, 1.0, n)
    x = np.empty(n)
    x[0] = e[0]
    for t in range(1, n):
        x[t] = rho * x[t - 1] + e[t]
    return x


def _planted_pairs(n_pairs: int = 100, k: int = 2, snr: float = 4.0,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """leader = AR(1); follower = leader, сдвинутый на k мес назад во времени
    (F[t] = L[t−k]) + независимый шум с SNR = var(сигнала)/var(шума)."""
    rng = np.random.default_rng(seed)
    X = np.empty((2 * n_pairs, T))
    for p in range(n_pairs):
        e = _ar1(rng, 0.5, T + k)
        lead = e[k:]
        X[2 * p] = lead
        X[2 * p + 1] = e[:T] + rng.normal(0.0, lead.std() / np.sqrt(snr), T)
    return X, np.arange(0, 2 * n_pairs, 2), np.arange(1, 2 * n_pairs, 2)


# (а) восстановление известного лага ------------------------------------------------

def test_lag_recovery_known_shift():
    X, li, fi = _planted_pairs(n_pairs=100, k=2, snr=4.0, seed=0)
    _, lag_map, _ = ll.lag_scan(X)
    recovery = (lag_map[li, fi] == -2).mean()  # i=leader лидирует j=follower на 2
    assert recovery >= 0.9, f"восстановление lag=2: {recovery:.2f} < 0.9"


def test_lag_recovery_snr3():
    X, li, fi = _planted_pairs(n_pairs=100, k=2, snr=3.0, seed=7)
    _, lag_map, _ = ll.lag_scan(X)
    assert (lag_map[li, fi] == -2).mean() >= 0.9


# (б) знак не инвертирован -------------------------------------------------------------

def test_sign_convention_not_inverted():
    """Одна пара без шума: L лидирует F на 2 мес → lag_map[L,F] = −2,
    lag_map[F,L] = +2, direction антисимметричен; edge list: leader=L."""
    rng = np.random.default_rng(0)
    e = _ar1(rng, 0.5, T + 2)
    X = np.vstack([e[2:], e[:T]])  # row0 = leader, row1 = follower (lag 2)
    r_max, lag_map, direction = ll.lag_scan(X)
    assert lag_map[0, 1] == -2, f"0 лидирует 1 на 2 → ожидалось −2, got {lag_map[0, 1]}"
    assert lag_map[1, 0] == 2
    assert direction[0, 1] == -1 and direction[1, 0] == 1
    assert (direction == -direction.T).all()
    assert np.isnan(r_max[0, 0]) and lag_map[0, 0] == 0  # диагональ вне игры
    edges = ll.lead_edges_from_scan(r_max, lag_map, np.ones((2, 2), bool))
    row = edges.iloc[0]
    assert (row.leader_row, row.follower_row, row.lag) == (0, 1, 2)
    assert row.r == pytest.approx(1.0, abs=1e-9)


# (в) суррогатный нуль: FPR --------------------------------------------------------------

def test_surrogate_null_fpr():
    """Независимые AR(1) ρ=0.5 (спектр нетривиален → phase-randomization
    содержательна): доля открытий после BH-FDR ≤ 0.05 + 0.02."""
    rng = np.random.default_rng(20)
    n = 150
    X = np.stack([_ar1(rng, 0.5, T) for _ in range(n)])
    null, _, _ = ll.surrogate_null(X, B=40, n_pairs=5000, seed=1)
    r_obs, _, _ = ll.lag_scan(X)
    iu = np.triu_indices(n, 1)
    p = ll.empirical_pvalues(r_obs[iu], np.sort(null))
    q = ll.bh_qvalues(p)
    rate = (q <= ll.FDR_Q).mean()
    assert rate <= 0.07, f"FPR {rate:.3f} > 0.07 — нуль не калиброван"


# (г) phase-randomization: спектр/ACF vs circular-shift ---------------------------------

def _acf(x: np.ndarray, hmax: int = 4) -> np.ndarray:
    z = x - x.mean()
    return np.array([np.mean(z[:-h] * z[h:]) / np.mean(z * z)
                     for h in range(1, hmax + 1)])


def test_phase_randomization_preserves_spectrum():
    rng = np.random.default_rng(10)
    x = _ar1(rng, 0.7, T)
    B = 400
    surr = ll.phase_randomize(np.tile(x, (B, 1)), rng)
    # периодограмма сохраняется ТОЧНО (амплитуды не тронуты)
    assert np.allclose(np.abs(np.fft.rfft(surr, axis=1)),
                       np.abs(np.fft.rfft(x)), atol=1e-10)
    # mean-ACF по суррогатам сходится к ACF исходного (автокорреляция жива)
    mean_acf = np.array([_acf(s) for s in surr]).mean(axis=0)
    assert np.abs(mean_acf - _acf(x)).max() < 0.1


def test_phase_randomization_breaks_alignment_circular_shift_powerless():
    """Пара с истинным лагом k=2, SNR=8. Phase-нуль сидит на нулевом уровне,
    а circular-shift: (1) всего T−1=22 допустимых сдвига → p_min = 1/23 ≈ 0.043
    (под FDR структурно бессилен); (2) малые сдвиги |s| ≤ k сохраняют
    выравнивание → нуль загрязнён сигналом."""
    rng = np.random.default_rng(11)
    e = _ar1(rng, 0.5, T + 2)
    lead = e[2:]
    pair = np.vstack([lead, e[:T] + rng.normal(0.0, lead.std() / np.sqrt(8.0), T)])
    r_obs = ll.lag_scan(pair)[0][0, 1]

    rng_ph = np.random.default_rng(12)
    null_ph = np.array([ll.lag_scan(ll.phase_randomize(pair, rng_ph))[0][0, 1]
                        for _ in range(300)])
    assert r_obs > np.quantile(null_ph, 0.99), "сигнал не отличим от phase-нуля"

    n_shifts = T - 1
    assert 1.0 / (n_shifts + 1) > 0.01, "circular-shift может дать малый p?"
    r_small = np.array([ll.lag_scan(np.vstack([pair[0], np.roll(pair[1], s)]))[0][0, 1]
                        for s in (1, 2)])
    assert (r_small > np.quantile(null_ph, 0.99)).all(), (
        f"малые сдвиги не сохранили выравнивание: {r_small}")


# (д) детерминизм ------------------------------------------------------------------------

def test_determinism():
    rng = np.random.default_rng(3)
    X = np.stack([_ar1(rng, 0.5, T) for _ in range(60)])
    n1 = ll.surrogate_null(X, B=5, n_pairs=500, seed=42)
    n2 = ll.surrogate_null(X, B=5, n_pairs=500, seed=42)
    assert np.array_equal(n1[0], n2[0]) and np.array_equal(n1[1], n2[1])
    assert np.array_equal(n1[2][0], n2[2][0])
    r1 = ll.lag_scan(X)
    r2 = ll.lag_scan(X)
    assert np.array_equal(np.nan_to_num(r1[0]), np.nan_to_num(r2[0]))
    assert np.array_equal(r1[1], r2[1])
    s1 = ll.synthetic_verification(seed=5, n_pairs=20, null_n=40, null_B=5)
    s2 = ll.synthetic_verification(seed=5, n_pairs=20, null_n=40, null_B=5)
    assert s1 == s2
