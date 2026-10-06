"""Тесты динамического слоя (research/29): Hungarian-матчинг (тождество,
прямоугольный birth, merge/split на ручных примерах), детерминизм полного
прогона на мини-графах PP-Dir, smoothing гасит моргание и не трогает устойчивых."""
import numpy as np
import pandas as pd
import scipy.sparse as sp

from ecotypes.config import load_config
from ecotypes.dynamics import (CLR_COLS, flicker_share, match_labels,
                               screen_events, smooth_labels, snapshot_labels,
                               transition_share)
from ecotypes.synthetic import make_ppdataset


def _mini_graphs(tmp_path, n=120, T=3, seed=5):
    """T снимков на общем node-set из PP-Dir (n=120, K=3): лёгкий дрейф весов,
    топология общая — как прод-графы 03."""
    d = make_ppdataset(seed=seed, K=3, n=n, T=24, delta=0.0)
    gdir = tmp_path / "graphs"
    gdir.mkdir()
    A0 = d["A"].tocoo()
    for t in range(T):
        w = np.asarray(A0.data, dtype=np.float64) * (1.0 + 0.02 * t)
        sp.save_npz(gdir / f"snap_t{t + 1:02d}.npz",
                    sp.csr_matrix((w, (A0.row, A0.col)), shape=A0.shape))
    pd.DataFrame({"row_idx": np.arange(n, dtype=np.int32),
                  "territory_id": np.arange(n, dtype=np.int32) + 1}
                 ).to_parquet(gdir / "node_index.parquet", index=False)
    return gdir


# (а) Hungarian на идентичных разбиениях → тождество
def test_hungarian_identity_on_identical_partitions():
    z = np.tile(np.array([0, 0, 1, 1, 2, 2, 2, 3]), (2, 1))
    zm, reg = match_labels(z, months=["m0", "m1"])
    assert np.array_equal(zm[0], zm[1])
    assert len(reg) == 4 and reg["death_month"].isna().all()


# (б) прямоугольный случай K_t ≠ K_{t+1}: новый кластер с J < τ_split → birth
def test_rectangular_birth():
    z0 = np.array([0] * 20 + [1] * 10)
    z1 = np.array([0] * 20 + [1] * 9 + [2])   # узел 29 откололся: J=1/10=0.1
    zm, reg = match_labels(np.vstack([z0, z1]), months=["m0", "m1"])
    new = zm[1, -1]
    row = reg[reg.type_id == new].iloc[0]
    assert row.birth_month == "m1" and pd.isna(row.parent_id)
    assert zm[1, 0] == zm[0, 0] and zm[1, 20] == zm[0, 20]  # старые унаследованы


# (в) split: второй по массе пересечения кластер t с J ≥ τ_split → parent_id
def test_split_registered():
    z0 = np.array([0] * 60 + [1] * 30)
    z1 = np.array([0] * 40 + [1] * 20 + [2] * 30)  # кластер 1 — откол от типа 0
    zm, reg = match_labels(np.vstack([z0, z1]), months=["m0", "m1"])
    a_id = zm[0, 0]
    child = zm[1, 50]
    crow = reg[reg.type_id == child].iloc[0]
    assert crow.parent_id == a_id and crow.birth_month == "m1"
    assert zm[1, 0] == a_id  # основная масса наследует тип


# (в′) merge: два типа t−1 сливаются в один кластер t → меньший merged_into больший
def test_merge_registered():
    z0 = np.array([0] * 40 + [1] * 20 + [2] * 40)
    z1 = np.array([0] * 60 + [1] * 40)  # типы 0 и 1 слились в кластер 0
    zm, reg = match_labels(np.vstack([z0, z1]), months=["m0", "m1"])
    a_id, b_id = zm[0, 0], zm[0, 40]
    brow = reg[reg.type_id == b_id].iloc[0]
    assert brow.merged_into == a_id and brow.death_month == "m0"
    assert (zm[1, :60] == a_id).all()  # больший родитель наследуется


# (г) детерминизм полного прогона на мини-графах: два прогона → те же labels
def test_full_run_deterministic(tmp_path):
    cfg = load_config("configs/default.yaml")
    g = _mini_graphs(tmp_path)
    z1, z2 = snapshot_labels(g, cfg), snapshot_labels(g, cfg)
    assert z1.shape == (3, 120) and z1.dtype == np.int64
    assert np.array_equal(z1, z2)
    zm1, r1 = match_labels(z1)
    zm2, r2 = match_labels(z2)
    assert np.array_equal(zm1, zm2)
    assert np.array_equal(smooth_labels(zm1), smooth_labels(zm2))
    assert r1[["type_id", "birth_month"]].equals(r2[["type_id", "birth_month"]])


# (д) smoothing снижает долю переходов и не меняет метки устойчивых узлов
def test_smoothing_reduces_flicker_keeps_stable():
    zm = np.array([
        [0, 1, 0, 0],   # моргание
        [1, 1, 1, 1],   # устойчивый
        [2, 2, 2, 2],   # устойчивый
        [0, 0, 1, 1],   # честный переход — должен сохраниться
        [1, 2, 1, 2],   # два моргания
    ], dtype=np.int64).T  # (T=4, n=5)
    zs = smooth_labels(zm, window=3)
    assert transition_share(zs) < transition_share(zm)
    assert flicker_share(zs) < flicker_share(zm)
    assert (zs[:, 1] == 1).all() and (zs[:, 2] == 2).all()
    assert zs[1, 0] == 0            # моргание погашено
    assert zs[2, 3] == 1 and zs[3, 3] == 1  # честный переход не стёрт


# ---------------------------------------------------------------- узловой скрин (№12)

def _screen_fixture():
    """5 узлов × 4 месяца. clr_prod несёт профиль (остальные clr_* = 0), поэтому
    CLR-дистанция смежных месяцев = |Δclr_prod|. Дистанции всех узел-месяцев:
    {0.1,5.0,0.0, 4.0,4.0,0.0, 0.1,4.5,0.0, 0.05,0.2,0.0, 0.1,0.0,6.0} →
    q75 = 4.0 (numpy linear). Пороги скрина заданы явно, не из default.yaml."""
    months = ["m0", "m1", "m2", "m3"]
    lab = {1: [0, 0, 1, 1],    # честный переход в m2 — должен пройти
           2: [0, 1, 0, 0],    # моргун: событие в m1 — фликер (m2 вернул 0)
           3: [0, 0, 1, 1],    # переход в m2, но seed_agreement 0.5 < 0.8
           4: [0, 0, 1, 1],    # переход в m2, но сдвиг 0.2 < q75
           5: [0, 0, 0, 1]}    # переход в m3 (последний месяц): форвард-чек снят
    agree = {1: 0.95, 2: 0.95, 3: 0.5, 4: 0.95, 5: 0.95}
    prof = {1: [0.0, 0.1, 5.1, 5.1],     # d(m1→m2) = 5.0 ≥ 4.0
            2: [0.0, 4.0, 0.0, 0.0],     # d(m0→m1) = 4.0 ≥ 4.0 (только фликер)
            3: [0.0, 0.1, 4.6, 4.6],     # d(m1→m2) = 4.5 ≥ 4.0 (только seed)
            4: [0.0, 0.05, 0.25, 0.25],  # d(m1→m2) = 0.2 < 4.0
            5: [0.0, 0.1, 0.1, 6.1]}     # d(m2→m3) = 6.0 ≥ 4.0
    rows_l, rows_p = [], []
    for tid in lab:
        for k, m in enumerate(months):
            rows_l.append({"territory_id": tid, "month": m,
                           "type_id_smooth": lab[tid][k],
                           "seed_agreement": agree[tid]})
            rows_p.append({"territory_id": tid, "month": m,
                           **{c: (prof[tid][k] if c == "clr_prod" else 0.0)
                              for c in CLR_COLS}})
    events = pd.DataFrame(
        [(1, "m2", 0, 1), (2, "m1", 0, 1), (3, "m2", 0, 1),
         (4, "m2", 0, 1), (5, "m3", 0, 1)],
        columns=["territory_id", "month", "type_from", "type_to"])
    cfg = load_config("configs/default.yaml")
    es = cfg.dynamics.event_screen
    es.seed_agreement_min, es.displacement_quantile, es.no_flicker = 0.8, 0.75, True
    return events, pd.DataFrame(rows_l), pd.DataFrame(rows_p), cfg


def _by_node(out):
    return {int(r.territory_id): r for r in out.itertuples()}


# (е) настоящий сдвиг профиля проходит скрин; моргун — нет (no_flicker)
def test_screen_admits_real_shift_rejects_flicker():
    out = screen_events(*_screen_fixture())
    by = _by_node(out)
    assert bool(by[1].admitted) and by[1].reject_reason == ""
    assert not bool(by[2].admitted) and "no_flicker" in by[2].reject_reason
    # граница: переход в последний месяц — форвард-проверка снята, остальное есть
    assert bool(by[5].admitted)


# (ж) низкий seed_agreement и нематериальный сдвиг отсеиваются со своими причинами
def test_screen_rejects_low_seed_and_small_displacement():
    out = screen_events(*_screen_fixture())
    by = _by_node(out)
    assert not bool(by[3].admitted) and "seed_agreement" in by[3].reject_reason
    assert not bool(by[4].admitted) and "displacement" in by[4].reject_reason
    # причины не смешиваются: каждый отсеян ровно своей проверкой
    assert by[2].reject_reason == "no_flicker"
    assert by[3].reject_reason == "seed_agreement"
    assert by[4].reject_reason == "displacement"


# (з) скрин детерминирован
def test_screen_deterministic():
    fx = _screen_fixture()
    pd.testing.assert_frame_equal(screen_events(*fx), screen_events(*fx))


# (и) admitted ⊆ all и счётчики сходятся
def test_screen_admitted_subset_and_counts():
    events, *_ = _screen_fixture()
    out = screen_events(*_screen_fixture())
    adm = out[out["admitted"]]
    key = ["territory_id", "month"]
    assert len(adm.merge(events, on=key)) == len(adm)          # подмножество
    assert len(adm) + int((~out["admitted"]).sum()) == len(out)  # счётчики
    assert len(adm) == 2  # ровно узлы 1 (сдвиг) и 5 (последний месяц)
