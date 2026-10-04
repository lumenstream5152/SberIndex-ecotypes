"""Сборка панели и атрибутов узлов по спеке research/21 (§2 решения, §3 схема).

Ядро: МО с полными 24 мес (2023-01…2024-12). CLR по 6 частям без pseudocount
(нулей в данных нет; clr_clip — только страховка), value уже руб/жителя
(на население НЕ делим), десезонирование — month-of-year индексы в логах
(STL запрещён: T=24, statsmodels нет), YoY = log-diff(12).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .config import Config

# suffix → категория consumption.parquet (food = «Общественное питание»)
CATS = {
    "prod": "Продовольствие",
    "health": "Здоровье",
    "market": "Маркетплейсы",
    "food": "Общественное питание",
    "transp": "Транспорт",
}
PARTS = [*CATS, "proch"]  # 6-я часть — остаток «Прочее»
N_MONTHS_EXPECTED = 24
NO_9999_TAKE_SINGLE = {1471, 1487, 1847}  # research/21 §1.3: только строка 2018–2024
# у внутригородских МО Москвы/СПб в справочнике нет координат центра →
# импутация центром региона (сам город); нужно для great-circle fallback и карт
REGION_CENTER = {"Москва": (55.7558, 37.6173), "Санкт-Петербург": (59.9343, 30.3351)}


def _resolve_dict(raw_dir: Path, tids: set[int]) -> pd.DataFrame:
    """Справочник МО → одна строка на territory_id: year_to==9999, иначе единственная
    (для 1471/1487/1847) / самая поздняя по year_to строка."""
    d = pd.read_excel(raw_dir / "t_dict_municipal_districts.xlsx")
    d = d[d.territory_id.notna() & d.territory_id.isin(tids)].copy()
    d["territory_id"] = d.territory_id.astype(np.int32)
    cur = d[d.year_to == 9999]
    rest = d[~d.territory_id.isin(cur.territory_id)]
    rest = rest.sort_values("year_to").groupby("territory_id", as_index=False).tail(1)
    sel = pd.concat([cur, rest], ignore_index=True)
    assert sel.territory_id.is_unique
    sel["ok8"] = sel.oktmo.str.replace("-", "", regex=False).str[:8]
    return sel


def _month_grid(cfg: Config) -> list[str]:
    return [str(p) for p in pd.period_range(cfg.data.start, cfg.data.end, freq="M")]


def _split_core(cons: pd.DataFrame, months: list[str]) -> tuple[list[int], pd.DataFrame]:
    """Ядро = tid с полным набором месяцев; остальным — механизм неполноты."""
    first, last = months[0], months[-1]
    g = cons.groupby("territory_id").date.agg(["nunique", "min", "max", set])
    core = sorted(g[g["set"].apply(lambda s: s == set(months))].index.tolist())
    excl = g[~g.index.isin(core)].reset_index()
    conds = [
        (excl["min"] == first) & (excl["max"] < last),
        (excl["min"] > first) & (excl["max"] == last),
        (excl["min"] == first) & (excl["max"] == last),
    ]
    excl["mech"] = np.select(conds, ["left", "right", "internal"], default="mid")
    return core, excl


def _build_monthly(cons: pd.DataFrame, core: list[int], months: list[str],
                   clr_clip: float) -> pd.DataFrame:
    sub = cons[cons.territory_id.isin(core)]
    wide = sub.pivot(index=["territory_id", "date"], columns="category",
                     values="value").reset_index()
    wide = wide.rename(columns={"Все категории": "val_all",
                                **{c: f"val_{s}" for s, c in CATS.items()}})
    wide["month_idx"] = wide.date.map({m: i for i, m in enumerate(months)})
    wide = wide.sort_values(["territory_id", "month_idx"]).reset_index(drop=True)

    shares = wide[[f"val_{s}" for s in CATS]].to_numpy(np.float64) / \
        wide["val_all"].to_numpy(np.float64)[:, None]
    proch = 1.0 - shares.sum(axis=1)
    shares = np.column_stack([shares, proch])
    for j, s in enumerate(PARTS):
        wide[f"share_{s}"] = shares[:, j].astype(np.float32)
    logp = np.log(np.maximum(shares, clr_clip))
    clr = logp - logp.mean(axis=1, keepdims=True)
    for j, s in enumerate(PARTS):
        wide[f"clr_{s}"] = clr[:, j].astype(np.float32)

    wide["log_all"] = np.log(wide["val_all"].astype(np.float64))
    # month-of-year демеанинг в логах per МО: sa(t) = log v(t) − mean log v(тот же месяц)
    cal_month = wide.date.str[5:7]
    moy_mean = wide.groupby(["territory_id", cal_month])["log_all"].transform("mean")
    wide["sa_all"] = (wide["log_all"] - moy_mean).astype(np.float32)
    g = wide.groupby("territory_id")
    wide["yoy_all"] = g["log_all"].diff(12).astype(np.float32)
    log_market = np.log(wide["val_market"].astype(np.float64))
    wide["yoy_market"] = log_market.groupby(wide.territory_id).diff(12).astype(np.float32)
    wide["log_all"] = wide["log_all"].astype(np.float32)

    cols = (["territory_id", "date", "month_idx", "val_all"]
            + [f"val_{s}" for s in CATS] + [f"share_{s}" for s in PARTS]
            + [f"clr_{s}" for s in PARTS]
            + ["log_all", "sa_all", "yoy_all", "yoy_market"])
    out = wide[cols].rename(columns={"date": "month"})
    out["territory_id"] = out.territory_id.astype(np.int32)
    out["month_idx"] = out.month_idx.astype(np.int8)
    for s in ["all", *CATS]:
        out[f"val_{s}"] = out[f"val_{s}"].astype(np.int32)
    return out


def _dec_amp(vals: np.ndarray) -> float:
    """value(дек)/mean(янв–ноя) по каждому году, среднее по двум. vals: (24,) по month_idx."""
    v = vals.reshape(2, 12)
    return float(np.mean(v[:, 11] / v[:, :11].mean(axis=1)))


def _build_nodes(panel: pd.DataFrame, ref: pd.DataFrame, cfg: Config,
                 raw_dir: Path) -> pd.DataFrame:
    g = panel.groupby("territory_id", sort=True)
    n = g.size().size
    nodes = pd.DataFrame({"territory_id": np.array(sorted(g.groups), np.int32)})
    nodes["level_mean"] = g["log_all"].mean().to_numpy(np.float32)
    last = panel[panel.month_idx == N_MONTHS_EXPECTED - 1].set_index("territory_id")
    nodes["level_end"] = last.loc[nodes.territory_id, "log_all"].to_numpy(np.float32)
    for s in PARTS:
        nodes[f"clr_mean_{s}"] = g[f"clr_{s}"].mean().to_numpy(np.float32)

    yoy = panel.dropna(subset=["yoy_all"]).groupby("territory_id")
    nodes["yoy_all_med"] = yoy["yoy_all"].median().to_numpy(np.float32)
    q = yoy["yoy_all"].quantile([0.25, 0.75]).unstack()
    nodes["yoy_all_iqr"] = (q[0.75] - q[0.25]).to_numpy(np.float32)
    nodes["yoy_market_med"] = yoy["yoy_market"].median().to_numpy(np.float32)

    first_m = panel[panel.month_idx == 0].set_index("territory_id")
    nodes["dclr_market"] = (last["clr_market"] - first_m["clr_market"]) \
        .loc[nodes.territory_id].to_numpy(np.float32)
    nodes["dclr_prod"] = (last["clr_prod"] - first_m["clr_prod"]) \
        .loc[nodes.territory_id].to_numpy(np.float32)

    wide_vals = {
        s: panel.pivot(index="territory_id", columns="month_idx", values=f"val_{s}")
        for s in ["all", *CATS]
    }
    tids = nodes.territory_id.to_numpy()
    for s, col in [("all", "dec_amp_all"), ("market", "dec_amp_market"),
                   ("prod", "dec_amp_prod")]:
        nodes[col] = np.array([_dec_amp(r) for r in
                               wide_vals[s].loc[tids].to_numpy(np.float64)],
                              dtype=np.float32)
    food = wide_vals["food"].loc[tids].to_numpy(np.float64).reshape(n, 2, 12)
    nodes["summer_amp_food"] = (food[:, :, 6:8].mean(axis=2) / food.mean(axis=2)) \
        .mean(axis=1).astype(np.float32)
    # сезонный профиль: mean log_all по календарному месяцу (2 года) → std 12 значений
    la = panel.pivot(index="territory_id", columns="month_idx",
                     values="log_all").loc[tids].to_numpy(np.float64)
    profile = la.reshape(n, 2, 12).mean(axis=1)
    nodes["season_std"] = profile.std(axis=1).astype(np.float32)

    nodes = nodes.merge(ref[["territory_id", "ok8", "region_code", "region_name",
                             "municipal_district_name", "municipal_district_type",
                             "municipal_district_status",
                             "municipal_district_center_lat",
                             "municipal_district_center_lon"]],
                        on="territory_id", how="left", validate="1:1")
    nodes = nodes.rename(columns={"municipal_district_name": "name",
                                  "municipal_district_type": "mun_type",
                                  "municipal_district_center_lat": "lat",
                                  "municipal_district_center_lon": "lon"})
    nodes["is_region_capital"] = (
        nodes.municipal_district_status == "административный_центр_субъекта")
    nodes = nodes.drop(columns=["municipal_district_status"])
    for col in ("lat", "lon"):
        med = nodes.groupby("region_code")[col].transform("median")
        nodes[col] = nodes[col].fillna(med)
    for reg, (la, lo) in REGION_CENTER.items():
        m = nodes.region_name == reg
        nodes.loc[m & nodes.lat.isna(), "lat"] = la
        nodes.loc[m & nodes.lon.isna(), "lon"] = lo
    nodes["lat"] = nodes.lat.astype(np.float32)
    nodes["lon"] = nodes.lon.astype(np.float32)
    nodes["region_code"] = nodes.region_code.astype(np.int8)

    nodes = _add_market_access(nodes, raw_dir)
    nodes = _add_rosstat(nodes, raw_dir)
    return nodes


def _impute_region_median(nodes: pd.DataFrame, col: str) -> None:
    med = nodes.groupby("region_code")[col].transform("median")
    nodes[col] = nodes[col].fillna(med).fillna(nodes[col].median())


def _add_market_access(nodes: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    ma = pd.read_parquet(raw_dir / "hackathon/hackathonlicence/market_access.parquet")
    nodes = nodes.merge(ma.rename(columns={"market_access": "_ma"}),
                        on="territory_id", how="left", validate="1:1")
    nodes["ma_missing"] = nodes._ma.isna()
    nodes["log_ma"] = np.log10(nodes._ma.astype(np.float64))
    _impute_region_median(nodes, "log_ma")  # медиана региона, research/21 §2.7
    obs = ~nodes.ma_missing
    mu, sd = nodes.loc[obs, "log_ma"].mean(), nodes.loc[obs, "log_ma"].std()
    nodes["log_ma"] = ((nodes.log_ma - mu) / sd).astype(np.float32)  # z внутри ядра
    return nodes.drop(columns=["_ma"])


def _rosstat_year(df: pd.DataFrame, col: str, years: tuple[int, ...]) -> pd.Series:
    """ok8 → значение за первый доступный год из years (fallback-цепочка)."""
    out = {}
    for y in years:
        sub = df[df.year == y].set_index("ok8")[col]
        for k, v in sub.items():
            if pd.notna(v):
                out.setdefault(k, v)
    return pd.Series(out)


def _add_rosstat(nodes: pd.DataFrame, raw_dir: Path) -> pd.DataFrame:
    """БД ПМО (research/34): ключ ok8 ↔ oktmo zfill(8). Дыры: флаг + медиана региона."""
    ext = raw_dir.parent / "external"
    cols = ["pop_2023", "pop_2024", "urban_share", "empl_pc", "log_wage"]
    f_pop, f_emp, f_wag = (ext / "rosstat_pmo_population_2022_2024_compact.csv",
                           ext / "rosstat_pmo_employment_okved_annual.csv",
                           ext / "rosstat_pmo_wages_total_annual.csv")
    if not all(p.exists() for p in (f_pop, f_emp, f_wag)):
        nodes[cols] = np.nan
        nodes["rosstat_missing"] = True
    else:
        pop = pd.read_csv(f_pop)
        pop["ok8"] = pop.oktmo.astype(np.int64).astype(str).str.zfill(8)
        # в источнике бывают битые total=0 при ненулевых urban/rural (Краснинский 2023)
        pop.loc[pop.pop_total <= 0, "pop_total"] = np.nan
        emp = pd.read_csv(f_emp)
        emp = emp[emp.okved2.str.startswith("Всего")].copy()
        emp["ok8"] = emp.oktmo.astype(np.int64).astype(str).str.zfill(8)
        wag = pd.read_csv(f_wag)
        wag["ok8"] = wag.oktmo.astype(np.int64).astype(str).str.zfill(8)

        by_year = {y: pop[pop.year == y].set_index("ok8") for y in (2022, 2023, 2024)}
        ok8 = nodes.ok8.to_numpy()
        p23 = _rosstat_year(pop, "pop_total", (2023, 2022)).reindex(ok8).to_numpy()
        p24 = _rosstat_year(pop, "pop_total", (2024, 2023, 2022)).reindex(ok8).to_numpy()
        urban = np.full(len(nodes), np.nan)
        for y in (2024, 2023, 2022):  # urban_share: 2024, fallback 2023→2022
            t = by_year[y].reindex(ok8)
            cand = (t.pop_urban.fillna(0) / t.pop_total.where(t.pop_total > 0)) \
                .clip(0, 1).to_numpy()
            take = np.isnan(urban) & ~np.isnan(cand)
            urban[take] = cand[take]
        empl = _rosstat_year(emp, "value", (2023, 2022)).reindex(ok8).to_numpy()
        wage = _rosstat_year(wag, "value", (2023, 2024, 2022)).reindex(ok8).to_numpy()

        nodes["pop_2023"] = p23
        nodes["pop_2024"] = p24
        nodes["urban_share"] = urban
        nodes["empl_pc"] = empl / np.where(p23 > 0, p23, np.nan)
        nodes["log_wage"] = np.log(wage)
        nodes["rosstat_missing"] = nodes[cols].isna().any(axis=1)

    for c in ["urban_share", "empl_pc", "log_wage"]:
        _impute_region_median(nodes, c)
    for c in ["pop_2023", "pop_2024"]:
        med = nodes.groupby("region_code")[c].transform("median")
        nodes[c] = nodes[c].fillna(med).fillna(nodes[c].median())
        nodes[c] = nodes[c].round().astype(np.int32)
    for c in ["urban_share", "empl_pc", "log_wage"]:
        nodes[c] = nodes[c].astype(np.float32)
    return nodes


def _haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    r = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def _build_edges(nodes: pd.DataFrame, raw_dir: Path, k: int) -> tuple[pd.DataFrame, list[int]]:
    core = set(nodes.territory_id)
    conn = pd.read_parquet(raw_dir / "hackathon/hackathonlicence/connection.parquet")
    hw = conn[conn.type == "highway"]
    hw = hw[hw.territory_id_x.isin(core) & hw.territory_id_y.isin(core)]
    both = pd.concat([
        hw.rename(columns={"territory_id_x": "tid_x", "territory_id_y": "tid_y"}),
        hw.rename(columns={"territory_id_y": "tid_x", "territory_id_x": "tid_y"}),
    ], ignore_index=True)[["tid_x", "tid_y", "distance"]]
    both = both.sort_values(["tid_x", "distance", "tid_y"])
    knn = both.groupby("tid_x", as_index=False).head(k).copy()
    knn["rank"] = knn.groupby("tid_x").cumcount() + 1
    knn["is_island_fallback"] = False

    geo = nodes.set_index("territory_id")[["lat", "lon"]]
    have_hw = set(knn.tid_x)
    fb_rows = []
    for tid in sorted(core - have_hw):  # острова: ближайший по great-circle, §2.7
        lat, lon = geo.loc[tid]
        others = geo.drop(index=tid)
        d = _haversine_km(lat, lon, others.lat.to_numpy(), others.lon.to_numpy())
        j = int(np.argmin(d))
        fb_rows.append({"tid_x": tid, "tid_y": int(others.index[j]),
                        "distance": float(d[j]), "rank": 1, "is_island_fallback": True})
    edges = pd.concat([knn, pd.DataFrame(fb_rows)], ignore_index=True)

    # симметризация: для каждой пары (x,y) гарантируем обратное ребро с тем же rank
    rev = edges.rename(columns={"tid_x": "tid_y", "tid_y": "tid_x"})
    merged = rev.merge(edges[["tid_x", "tid_y"]].assign(_f=1),
                       on=["tid_x", "tid_y"], how="left")
    mirrors = merged[merged._f.isna()].drop(columns=["_f"])
    edges = pd.concat([edges, mirrors], ignore_index=True)
    edges = edges.sort_values(["tid_x", "rank", "tid_y"]).reset_index(drop=True)
    out = edges.astype({"tid_x": np.int32, "tid_y": np.int32,
                        "distance": np.float32, "rank": np.int8})
    out = out.rename(columns={"distance": "dist_km"})
    return out, sorted(core - have_hw)


def build_panel(cfg: Config, raw_dir: str | Path = "data/raw",
                out_dir: str | Path = "data/processed") -> dict[str, Path]:
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    months = _month_grid(cfg)
    cons = pd.read_parquet(raw_dir / "hackathon/hackathonlicence/consumption.parquet")

    core, excl = _split_core(cons, months)
    if cfg.data.sample_n is not None:  # smoke: детерминированная подвыборка ядра
        rng = np.random.default_rng(cfg.seed)
        core = sorted(rng.choice(np.array(core), size=cfg.data.sample_n,
                                 replace=False).tolist())

    ref = _resolve_dict(raw_dir, set(cons.territory_id.unique()))

    panel = _build_monthly(cons, core, months, cfg.panel.clr_clip)
    nodes = _build_nodes(panel, ref[ref.territory_id.isin(core)], cfg, raw_dir)
    edges, island_tids = _build_edges(nodes, raw_dir, cfg.graph.geo.knn_k)
    nodes["geo_island"] = nodes.territory_id.isin(island_tids)

    er = ref.set_index("territory_id")
    excluded = pd.DataFrame({
        "territory_id": excl.territory_id.astype(np.int32),
        "region_name": er.loc[excl.territory_id, "region_name"].to_numpy(),
        "name": er.loc[excl.territory_id, "municipal_district_name"].to_numpy(),
        "mech": excl.mech.to_numpy(),
        "n_months": excl["nunique"].astype(np.int8).to_numpy(),
        "first": excl["min"].to_numpy(),
        "last": excl["max"].to_numpy(),
        "year_from": er.loc[excl.territory_id, "year_from"].astype(np.int16).to_numpy(),
    }).sort_values("territory_id").reset_index(drop=True)

    regions = (nodes.groupby(["region_code", "region_name"], as_index=False)
               .size().rename(columns={"size": "n_core_mo"})
               .sort_values("region_code").reset_index(drop=True))
    regions["n_core_mo"] = regions.n_core_mo.astype(np.int16)
    regions["small_region"] = regions.n_core_mo < 5

    nodes = nodes.sort_values("territory_id").reset_index(drop=True)
    node_cols = (["territory_id", "ok8", "region_code", "region_name", "name",
                  "mun_type", "is_region_capital", "lat", "lon",
                  "level_mean", "level_end"] + [f"clr_mean_{s}" for s in PARTS]
                 + ["yoy_all_med", "yoy_all_iqr", "yoy_market_med", "dclr_market",
                    "dclr_prod", "dec_amp_all", "dec_amp_market", "dec_amp_prod",
                    "summer_amp_food", "season_std", "log_ma", "ma_missing",
                    "geo_island", "pop_2023", "pop_2024", "urban_share", "empl_pc",
                    "log_wage", "rosstat_missing"])
    nodes = nodes[node_cols]
    frames = {"panel_monthly": panel, "nodes_static": nodes,
              "edges_highway_knn": edges, "excluded_incomplete": excluded,
              "regions": regions}
    out: dict[str, Path] = {}
    for name, df in frames.items():
        p = out_dir / f"{name}.parquet"
        df.to_parquet(p, index=False)
        out[name] = p
    return out
