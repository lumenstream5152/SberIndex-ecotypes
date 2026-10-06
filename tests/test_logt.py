"""Тесты log-t модуля (research/30): направление теста и восстановление клубов
на синтетике, глобальная сходимость, случайные блуждания как контроль,
детерминизм, согласие σ/β с log-t (правило ≥2 из 3)."""
import numpy as np

from ecotypes import logt as lt

T = 24
_T = np.arange(T)


def _panel_two_clubs_divergent(seed=7, na=20, nb=20, nc=10):
    """Два сходящихся блока (→2.0 и →1.0) + nc экспоненциальных дивергентов."""
    rng = np.random.default_rng(seed)
    ya = 2.0 + (rng.uniform(1.0, 3.0, na) - 2.0)[:, None] * np.exp(-0.25 * _T)[None, :]
    yb = 1.0 + (rng.uniform(0.5, 1.5, nb) - 1.0)[:, None] * np.exp(-0.25 * _T)[None, :]
    rate = np.linspace(0.12, 0.30, nc)
    yc = rng.uniform(0.8, 1.2, nc)[:, None] * np.exp(rate[:, None] * _T[None, :])
    ya += rng.normal(0, 0.01, ya.shape)
    yb += rng.normal(0, 0.01, yb.shape)
    yc += rng.normal(0, 0.01, yc.shape)
    return np.vstack([ya, yb, yc]).T, na, nb, nc


def test_two_clubs_plus_divergent():
    """(а) два клуба + дивергентный слой; знаки t-стат правильные."""
    Y, na, nb, nc = _panel_two_clubs_divergent()
    assert lt.logt_test(Y[:, :na])["t_stat"] > 0  # клуб A сходится
    assert lt.logt_test(Y[:, na:na + nb])["t_stat"] > 0  # клуб B сходится
    assert lt.logt_test(Y[:, :na + nb])["t_stat"] < -1.65  # A+B нет
    assert lt.logt_test(Y[:, na + nb:])["t_stat"] < -1.65  # дивергенты нет
    res = lt.club_classification(Y)
    assert res["n_clubs"] == 2
    assert sorted(res["club_sizes"]) == [na, nb]
    assert len(res["divergent"]) == nc
    lab = res["clubs"]
    assert len(set(lab[:na])) == 1 and len(set(lab[na:na + nb])) == 1
    assert lab[0] != lab[na]  # разные клубы
    assert (lab[na + nb:] == -1).all()
    assert all(t > -1.65 for t in res["club_tstats"])  # пост-валидация ядер
    pair_t = np.asarray(res["merge_log"][-1]["pairwise_t_final"])
    offdiag = pair_t[~np.eye(pair_t.shape[0], dtype=bool)]
    assert np.all(offdiag < -1.65)  # клубы различимы


def test_global_convergence_one_club():
    """(б) полностью сходящаяся панель → 1 клуб, дивергентов нет."""
    rng = np.random.default_rng(1)
    Y = (1.0 + (rng.uniform(0.5, 2.0, 30) - 1.0)[:, None] * np.exp(-0.3 * _T)[None, :]
         + rng.normal(0, 0.01, (30, T))).T
    assert lt.logt_test(Y)["converges"]
    res = lt.club_classification(Y)
    assert res["n_clubs"] == 1 and len(res["divergent"]) == 0


def test_random_walks_reject():
    """(в) контроль: независимые случайные блуждания → t << −1.65."""
    rng = np.random.default_rng(0)
    Y = 50.0 + np.cumsum(rng.normal(0, 1.0, (T, 50)), axis=0)
    assert lt.logt_test(Y)["t_stat"] < -1.65


def test_determinism():
    """(г) повторный прогон бит-в-бит: метки, размеры, merge_log."""
    Y, _, _, _ = _panel_two_clubs_divergent()
    r1 = lt.club_classification(Y)
    r2 = lt.club_classification(Y)
    assert np.array_equal(r1["clubs"], r2["clubs"])
    assert np.array_equal(r1["divergent"], r2["divergent"])
    assert r1["merge_log"][:-1] == r2["merge_log"][:-1]
    assert np.array_equal(np.asarray(r1["merge_log"][-1]["pairwise_t_final"]),
                          np.asarray(r2["merge_log"][-1]["pairwise_t_final"]),
                          equal_nan=True)


def test_sigma_beta_agree_with_logt():
    """(д) σ и β согласованы с log-t: 3/3 на сходящейся, 0/3 на расходящейся."""
    rng = np.random.default_rng(2)
    conv = (1.0 + (rng.uniform(0.5, 2.0, 40) - 1.0)[:, None]
            * np.exp(-0.3 * _T)[None, :] + rng.normal(0, 0.01, (40, T))).T
    v = lt.verdict_2of3(lt.logt_test(conv), lt.sigma_convergence(conv),
                        lt.beta_convergence(conv))
    assert v["n_votes"] == 3 and v["verdict"] == "convergence"
    # дивергенция с β>0: темп привязан к начальному уровню (богатые растут быстрее)
    c0 = rng.uniform(0.5, 2.0, 40)
    div = (c0[:, None] * np.exp(0.1 * c0[:, None] * _T[None, :])
           + rng.normal(0, 0.01, (40, T))).T
    v = lt.verdict_2of3(lt.logt_test(div), lt.sigma_convergence(div),
                        lt.beta_convergence(div))
    assert v["n_votes"] == 0 and v["verdict"] == "divergence"
    assert lt.beta_convergence(div)["beta"] > 0
    assert lt.sigma_convergence(div)["trend"] > 0


def test_prefix_trick_matches_direct():
    """Страховка O(NT)-трюка: префиксные H совпадают с прямым расчётом."""
    rng = np.random.default_rng(3)
    Y = rng.uniform(0.5, 1.5, (T, 15))
    Hp = lt._prefix_H(Y)
    for k in (2, 5, 15):
        assert np.allclose(Hp[:, k - 1], lt._H_of(Y[:, :k]))
