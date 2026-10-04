"""Тесты панели (research/21 §3): контракт сырых данных, схема выхода,
замкнутость композиции, детерминизм, smoke-подвыборка."""
import numpy as np
import pandas as pd
import pytest

from ecotypes.config import load_config
from ecotypes.panel import PARTS, build_panel

from conftest import requires_raw

RAW = "data/raw"
HL = f"{RAW}/hackathon/hackathonlicence"
EXPECTED_CATS = {"Все категории", "Здоровье", "Маркетплейсы",
                 "Общественное питание", "Продовольствие", "Транспорт"}


@requires_raw
def test_raw_contract():
    cons = pd.read_parquet(f"{HL}/consumption.parquet")
    assert set(cons.columns) == {"date", "territory_id", "category", "value"}
    assert set(cons.category.unique()) == EXPECTED_CATS
    assert cons.date.nunique() == 24
    assert cons.territory_id.nunique() == 2190
    assert not cons.duplicated(["territory_id", "date", "category"]).any()
    assert (cons.value > 0).all()  # нулей нет → CLR без pseudocount


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    out = tmp_path_factory.mktemp("panel")
    cfg = load_config("configs/default.yaml")
    paths = build_panel(cfg, raw_dir=RAW, out_dir=out)
    return {k: pd.read_parquet(p) for k, p in paths.items()}, out, cfg


@requires_raw
def test_shape_contract(built):
    frames, _, _ = built
    panel, nodes = frames["panel_monthly"], frames["nodes_static"]
    assert len(nodes) == 2016
    assert len(panel) == 2016 * 24 == 48384
    assert len(frames["excluded_incomplete"]) == 174
    assert len(frames["regions"]) == 73
    assert panel.territory_id.nunique() == 2016
    assert list(nodes.territory_id) == sorted(nodes.territory_id)
    assert set(panel.month) == {f"2023-{m:02d}" for m in range(1, 13)} | \
        {f"2024-{m:02d}" for m in range(1, 13)}


@requires_raw
def test_composition_closed(built):
    panel = built[0]["panel_monthly"]
    shares = panel[[f"share_{s}" for s in PARTS]].to_numpy(np.float64)
    assert np.abs(shares.sum(axis=1) - 1.0).max() < 1e-6
    assert (shares > 0).all()
    clr = panel[[f"clr_{s}" for s in PARTS]].to_numpy(np.float64)
    assert np.abs(clr.sum(axis=1)).max() < 1e-6


@requires_raw
def test_no_unexpected_nan(built):
    frames, _, _ = built
    panel = frames["panel_monthly"]
    early = panel.month_idx < 12
    assert panel.loc[early, ["yoy_all", "yoy_market"]].isna().all().all()
    late = panel.loc[~early]
    assert not late.isna().any().any()
    assert not frames["nodes_static"].isna().any().any()
    edges = frames["edges_highway_knn"]
    assert not edges.isna().any().any()
    # симметрия: каждое ребро имеет обратное
    pairs = set(zip(edges.tid_x, edges.tid_y))
    assert all((y, x) in pairs for x, y in pairs)
    # каждый остров имеет fallback-ребро (Анадырь↔Анадырский взаимны → без зеркал-дублей)
    isl = set(frames["nodes_static"].loc[frames["nodes_static"].geo_island,
                                         "territory_id"])
    assert len(isl) == 10
    fb = edges[edges.is_island_fallback]
    assert isl <= set(fb.tid_x)
    assert frames["nodes_static"].ma_missing.sum() == 12


@requires_raw
def test_determinism(built, tmp_path):
    frames, _, cfg = built
    paths2 = build_panel(cfg, raw_dir=RAW, out_dir=tmp_path)
    for name, df in frames.items():
        df2 = pd.read_parquet(paths2[name])
        pd.testing.assert_frame_equal(df, df2)


@requires_raw
def test_smoke_sample(tmp_path):
    cfg = load_config("configs/default.yaml", overrides="configs/smoke.yaml")
    assert cfg.data.sample_n == 200
    paths = build_panel(cfg, raw_dir=RAW, out_dir=tmp_path)
    nodes = pd.read_parquet(paths["nodes_static"])
    panel = pd.read_parquet(paths["panel_monthly"])
    assert len(nodes) == 200 and len(panel) == 200 * 24
    assert list(nodes.territory_id) == sorted(nodes.territory_id)
    # детерминизм подвыборки: второй прогон → те же tid
    paths2 = build_panel(cfg, raw_dir=RAW, out_dir=tmp_path / "b")
    assert list(pd.read_parquet(paths2["nodes_static"]).territory_id) == \
        list(nodes.territory_id)
