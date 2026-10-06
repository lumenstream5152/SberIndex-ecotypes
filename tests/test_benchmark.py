"""Тесты сравнительного слоя (benchmark.py): z-нормировка, композит, margin rule,
LOMO, Wilcoxon/BH, гейт-фильтрация, чтение весов ИЗ prereg.yaml."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import yaml

from ecotypes import benchmark as bm

W = {"Q": 0.30, "S": 0.25, "T": 0.20, "R": 0.15, "H": 0.10}
LEGS = list(W)


def _legs(df_dict: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(df_dict, index=["M1", "M2", "M3"])


# ---------------------------------------------------------------- z-нормировка

def test_zscore_manual():
    s = pd.Series([1.0, 2.0, 3.0])
    z = bm.zscore(s)
    # ddof=0: mean=2, sd=sqrt(2/3)
    sd = np.sqrt(2.0 / 3.0)
    assert np.allclose(z.to_numpy(), [-1 / sd, 0.0, 1 / sd])


def test_zscore_zero_variance():
    z = bm.zscore(pd.Series([5.0, 5.0, 5.0]))
    assert (z == 0).all()


# ---------------------------------------------------------------- композит

def test_composite_manual_weights():
    legs = _legs({"Q": [3.0, 2.0, 1.0], "S": [1.0, 2.0, 3.0],
                  "T": [1.0, 1.0, 1.0], "R": [2.0, 2.0, 2.0],
                  "H": [0.0, 1.0, 2.0]})
    lz = bm.leg_zscores(legs)
    comp = bm.composite(lz, W)
    # ручная проверка: z(Q)=[1,-0? ...] для [3,2,1]: mean=2 sd=sqrt(2/3)
    sd3 = np.sqrt(2.0 / 3.0)
    zq = pd.Series([1 / sd3, 0.0, -1 / sd3], index=legs.index)
    zs = -zq
    zh = pd.Series([-1 / sd3, 0.0, 1 / sd3], index=legs.index)
    manual = 0.30 * zq + 0.25 * zs + 0.10 * zh  # T,R нулевой дисперсии → z=0
    assert np.allclose(comp.to_numpy(), manual.to_numpy())
    # точный знак: M1 = (0.30 − 0.25 − 0.10)/sd3 < 0; M3 — зеркально > 0
    assert comp.idxmax() == "M3" and comp["M3"] > comp["M2"] > comp["M1"]


def test_composite_missing_leg_raises():
    with pytest.raises(KeyError):
        bm.composite(bm.leg_zscores(_legs({"Q": [1, 2, 3]})), W)


def test_composite_reads_weights_from_prereg_file(tmp_path):
    """Веса читаются из prereg.yaml: меняем вес в копии файла → композит меняется."""
    legs = _legs({"Q": [3.0, 1.0, 2.0], "S": [1.0, 3.0, 2.0],
                  "T": [2.0, 2.0, 2.0], "R": [2.0, 2.0, 2.0], "H": [2.0, 2.0, 2.0]})
    lz = bm.leg_zscores(legs)
    prereg = yaml.safe_load(open("configs/prereg.yaml", encoding="utf-8"))
    w0 = {k: float(v) for k, v in prereg["similarity"]["composite_weights"].items()}
    comp0 = bm.composite(lz, w0)
    assert comp0.idxmax() == "M1"  # Q вес 0.30 > S 0.25
    prereg["similarity"]["composite_weights"]["S"] = 0.55
    prereg["similarity"]["composite_weights"]["Q"] = 0.05
    p = tmp_path / "prereg_mod.yaml"
    p.write_text(yaml.safe_dump(prereg, allow_unicode=True), encoding="utf-8")
    w1 = {k: float(v) for k, v in
          yaml.safe_load(p.read_text(encoding="utf-8"))["similarity"
                                                          ]["composite_weights"].items()}
    comp1 = bm.composite(lz, w1)
    assert comp1.idxmax() == "M2"
    assert not np.allclose(comp0.to_numpy(), comp1.to_numpy())


# ---------------------------------------------------------------- margin rule

def _se(series_idx, val):
    return pd.Series({i: val for i in series_idx})


def test_margin_rule_win():
    scores = pd.Series({"M8": 1.0, "M5": 0.5, "M2": 0.0})
    v = bm.margin_verdict(scores, _se(scores.index, 0.1), 1.0,
                          ["M8", "M5", "M2"])
    assert v["verdict"] == "win" and v["winner"] == "M8"


def test_margin_rule_tie_goes_to_tiebreak():
    scores = pd.Series({"M2": 1.0, "M8": 0.95, "M5": 0.0})
    # отрыв 0.05 < 1·SE(0.2) → ничья → tiebreak: M8 раньше M2
    v = bm.margin_verdict(scores, _se(scores.index, 0.2), 1.0,
                          ["M8", "M5", "M2", "M1"])
    assert v["verdict"] == "tie_tiebreak" and v["winner"] == "M8"


# ---------------------------------------------------------------- LOMO

def test_lomo_stable():
    legs = _legs({"Q": [3.0, 1.0, 2.0], "S": [3.0, 1.0, 2.0], "T": [3.0, 1.0, 2.0],
                  "R": [3.0, 1.0, 2.0], "H": [3.0, 1.0, 2.0]})
    out = bm.lomo_winners(bm.leg_zscores(legs), W)
    assert out["full_winner"] == "M1"
    assert out["n_changed"] == 0 and not out["photo_finish"]


def test_lomo_photo_finish():
    # M1 держится на Q и S; M2 — на T,R,H. Удаление Q ИЛИ S смещает топ-1 →
    # меняется 2 из 5 → фотофиниш (правило «>1 из 5», prereg §A).
    legs = _legs({"Q": [10.0, 0.0, 0.0], "S": [10.0, 0.0, 0.0],
                  "T": [0.0, 1.0, 1.0], "R": [0.0, 1.0, 1.0], "H": [0.0, 1.0, 1.0]})
    out = bm.lomo_winners(bm.leg_zscores(legs), W)
    assert out["full_winner"] == "M1"
    assert out["per_drop"]["Q"] == "M2" and out["per_drop"]["S"] == "M2"
    assert out["per_drop"]["T"] == "M1"
    assert out["n_changed"] == 2 and out["photo_finish"]


# ---------------------------------------------------------------- Wilcoxon / BH

def test_wilcoxon_paired_known():
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 0.1, 15)
    y = rng.normal(0.0, 0.1, 15)
    df = bm.wilcoxon_table(pd.DataFrame({"a": x, "b": y, "c": y}))
    row = df[((df.a == "a") & (df.b == "b"))].iloc[0]
    assert row.p < 0.001
    # вырожденная пара (b ≡ c) → p=1.0 с флагом, без падения scipy
    row_bc = df[((df.a == "b") & (df.b == "c"))].iloc[0]
    assert row_bc.degenerate and row_bc.p == 1.0
    assert (df.p_adj >= df.p).all() and (df.p_adj <= 1).all()


def test_bh_adjust_monotone():
    p = np.array([0.001, 0.01, 0.04, 0.5])
    q = bm.bh_adjust(p)
    assert np.allclose(q, [0.004, 0.02, 0.053333, 0.5], atol=1e-6)


# ---------------------------------------------------------------- гейт-фильтрация

def test_scoring_candidates_excludes_anti_and_failed():
    rows = [{"measure": m, "verdict": "pass"} for m in
            ["M1", "M2", "M5", "M8", "X1", "X2", "X3"]]
    rows.append({"measure": "M11", "verdict": "fail"})
    table_a = pd.DataFrame(rows)
    prereg = yaml.safe_load(open("configs/prereg.yaml", encoding="utf-8"))
    cand, excluded = bm.scoring_candidates(table_a, prereg["similarity"])
    assert cand == ["M1", "M2", "M5", "M8"]
    assert "M11" in excluded and "admissibility" in excluded["M11"]
    assert all(x in excluded and excluded[x] == "anti_example"
               for x in ("X1", "X2", "X3"))


# ---------------------------------------------------------------- композит методов

def test_method_composite_two_versions_without_interpretability():
    df = pd.DataFrame({
        "nmi_synth": [0.9, 0.5], "boot_ari_mean": [0.95, 0.8],
        "seed_ari_iqr": [0.01, 0.1], "SW": [0.2, 0.1], "CH_over_N": [0.1, 0.05],
        "S_Dbw": [0.5, 0.8], "MQ": [0.3, 0.1], "timing_s": [10.0, 100.0]},
        index=["m1", "m2"])
    pm = yaml.safe_load(open("configs/prereg.yaml", encoding="utf-8"))["method"]
    out = bm.method_composite(df, pm, interp=None)
    assert not out["interp_present"]
    z = out["composite_with_zero_interp"]
    r = out["composite_renormalized"]
    # обе версии репортятся, ранжирование здесь совпадает, шкалы разные
    assert z.idxmax() == "m1" and r.idxmax() == "m1"
    assert not np.isclose(z["m1"], r["m1"])
    # с заполненной интерпретацией — одна версия
    out2 = bm.method_composite(df, pm,
                               interp=pd.Series({"m1": 4.0, "m2": 2.0}))
    assert out2["interp_present"] and "composite_renormalized" not in out2


def test_load_interpretability_empty(tmp_path):
    p = tmp_path / "rub.yaml"
    p.write_text("scores: {}\n", encoding="utf-8")
    s, status = bm.load_interpretability(p, ["kmeans"])
    assert s is None and "не заполнена" in status
    s, status = bm.load_interpretability(tmp_path / "none.yaml", ["kmeans"])
    assert s is None and "отсутствует" in status
