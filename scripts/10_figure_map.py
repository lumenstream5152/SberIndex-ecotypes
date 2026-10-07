"""10: герой-карта макро-типов (F0) для PDF-презентации СберИндекс-2026.

Точечная карта 2016 МО поверх светлой подложки-контуров: цвет =
leiden_consensus (K0/K1/K2), размер точки ∝ sqrt(pop_2024) — в matplotlib
``s`` это площадь маркера, поэтому площадь ∝ pop, а визуальный диаметр
∝ sqrt(pop).

Проекция — коническая равновеликая Альберса (параллели 50°N/70°N,
осевой меридиан 100°E): стандартный вид карты России без внешних
зависимостей (cartopy в venv нет). Долготы Чукотки (−179°) сворачиваются
через dlon в [−180, 180) относительно осевого меридиана.

Подложка: полигоны data/raw/t_dict_municipal_districts_poly.gpkg (71 МБ,
EPSG:4326). Без них точечная карта не читается как силуэт России: 82%
точек сидят в европейской части, Сибирь — редкая пыль. geopandas/shapely
в venv нет и ради одной фигуры не добавляются — GPKG это SQLite + WKB,
парсится stdlib (sqlite3 + struct). Кольца децимируются страйдом до
≤RING_TARGET вершин — подложке хватает, PNG не раздувается. Полигоны
нужны только ради силуэта: дыры-анклавы заливаются цветом подложки (белое
на бумаге читалось бы как «нет данных»), границы МО
не рисуются (на масштабе страны волосяные линии дают муар).

242 МО с импутированными центрами (geo_island и пр.) не выделяются —
по брифу показываются как обычные точки.

Выходы (300 dpi, фон = бумага деки #FFFDF7, без осей):
  report/figures/F0_map.png       — герой слайда типов;
  report/figures/F0_map_mini.png  — мини-карта для титула
                                    (монохромная база + акценты K1/K2).
"""
from __future__ import annotations

import argparse
import logging
import sqlite3
import struct
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

from ecotypes.config import load_config
from ecotypes.seeds import set_all_seeds

NODES = Path("data/processed/nodes_static.parquet")
LABELS = Path("outputs/main/labels.parquet")
GPKG = Path("data/raw/t_dict_municipal_districts_poly.gpkg")
FIG = Path("report/figures")

# ------------------------------------------------------------------ константы стиля
# Семантика типов (профили регионов, outputs/main/passports_macro.parquet):
#   K0 (n=1654) — периферия: вся страна, медианный pop ≈ 20 тыс.
#   K1 (n=196)  — городские ядра: Москва (144) + Московская область (45).
#   K2 (n=166)  — Северо-Запад: СПб (98), Ленобласть, Карелия, Калининград.
COLORS = {
    0: "#C9B585",  # K0 Периферийная Россия — песок (палитра деки)
    1: "#4E7CA6",  # K1 Сервисные городские ядра — стальная синь
    2: "#8E3B4B",  # K2 Северо-Запад — винная
}
TYPE_NAMES = {
    0: "Периферийная Россия",
    1: "Сервисные городские ядра",
    2: "Северо-Запад",
}
ALPHA = {0: 0.60, 1: 0.90, 2: 0.90}
ZORDER = {0: 1, 2: 2, 1: 3}  # периферия снизу, ядра сверху

PAPER = "#FFFDF7"          # фон = панель деки (не чисто белый)
LAND_FACE = "#EBEFF1"      # подложка: холодный светло-серый, читается на бумаге
LAND_FACE_MINI = "#E4E9EC" # в мини карта мельче → подложку чуть плотнее
INK = "#1D2530"
MUTED = "#4C5361"
FAINT = "#8B8FA0"
RING_TARGET = 250          # макс. вершин на кольцо после децимации

S_MAX = 280.0   # площадь маркера (pt²) для max pop (1.63M); cap против каши в ЕЧ
S_MIN = 2.0     # пол того, чтобы малые МО не вырождались в пыль
EDGE_ACCENT = {"edgecolors": "white", "linewidths": 0.25}  # отбивка акцентов в плотных кластерах

# Альберс для России
ALBERS_LAT1, ALBERS_LAT2 = 50.0, 70.0   # стандартные параллели, °N
ALBERS_LON0 = 100.0                     # осевой меридиан, °E
ALBERS_LAT0 = 60.0                      # параллель начала отсчёта, °N

# Шрифт: DejaVu Sans (кириллица есть; системный PT Sans живёт в .ttc, который
# matplotlib рисует молча пусто — не используем)
plt.rcParams.update({"font.family": "DejaVu Sans",
                     "figure.dpi": 150, "text.color": INK})

log = logging.getLogger("figure_map")


# ------------------------------------------------------------------ проекция
def albers_russia(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Коническая равновеликая Альберса, единицы — условные (R=1).

    dlon приводится к [−180, 180) относительно осевого меридиана —
    Чукотка (lon≈−179°) уезжает вправо к Камчатке, а не влево за карту.
    Свёртка поточечная, поэтому кольца полигонов, пересекающие
    антимеридиан, не дают горизонтальных шлейфов через всю карту.
    """
    lam = np.deg2rad(np.asarray(lon, dtype=float))
    phi = np.deg2rad(np.asarray(lat, dtype=float))
    lam0, phi0 = np.deg2rad(ALBERS_LON0), np.deg2rad(ALBERS_LAT0)
    phi1, phi2 = np.deg2rad(ALBERS_LAT1), np.deg2rad(ALBERS_LAT2)

    n = 0.5 * (np.sin(phi1) + np.sin(phi2))
    c = np.cos(phi1) ** 2 + 2.0 * n * np.sin(phi1)
    rho = np.sqrt(np.maximum(c - 2.0 * n * np.sin(phi), 0.0)) / n
    rho0 = np.sqrt(c - 2.0 * n * np.sin(phi0)) / n

    dlam = (lam - lam0 + np.pi) % (2.0 * np.pi) - np.pi
    theta = n * dlam
    return rho * np.sin(theta), rho0 - rho * np.cos(theta)


# ------------------------------------------------------------------ GPKG → кольца (stdlib)
def _wkb_polygon(buf: bytes, off: int, endian: str, stride: int) -> tuple[list[np.ndarray], int]:
    """Polygon: n rings × (n points × stride doubles); возвращает кольца XY."""
    (n_rings,) = struct.unpack_from(endian + "I", buf, off)
    off += 4
    rings = []
    for _ in range(n_rings):
        (n_pts,) = struct.unpack_from(endian + "I", buf, off)
        off += 4
        n = n_pts * stride
        vals = struct.unpack_from(endian + f"{n}d", buf, off)
        off += 8 * n
        rings.append(np.asarray(vals).reshape(n_pts, stride)[:, :2])
    return rings, off


def decode_gpkg_geom(blob: bytes) -> list[list[np.ndarray]]:
    """GeoPackageBinary → список полигонов (каждый = список колец XY).

    Заголовок GPKG: 'GP' + version + flags; длина конверта кодируется
    битами 1–3 flags — сам конверт пропускаем, WKB сам несёт endianness.
    Типы по ISO SQL/MM: 3/6 = XY, 1003/1006 = +Z, 2003/2006 = +M, 3003/3006 = +ZM.
    В слое 2660 геометрий: 2447 MultiPolygon + 213 Polygon — все идут
    в подложку; неполигональные типы (на будущее) дают пустой список.
    """
    if blob[:2] != b"GP":
        raise ValueError("не GeoPackageBinary")
    env_code = (blob[3] >> 1) & 0x07
    off = 8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env_code]

    endian = "<" if blob[off] == 1 else ">"
    (gtype,) = struct.unpack_from(endian + "I", blob, off + 1)
    off += 5
    base, dim = gtype % 1000, gtype // 1000
    stride = 2 + (1 if dim in (1, 2) else 2 if dim == 3 else 0)

    if base == 3:  # Polygon (ISO WKB: 3; 2 = LineString — не подложка)
        rings, _ = _wkb_polygon(blob, off, endian, stride)
        return [rings]
    if base == 6:
        (n_polys,) = struct.unpack_from(endian + "I", blob, off)
        off += 4
        polys = []
        for _ in range(n_polys):
            sub_endian = "<" if blob[off] == 1 else ">"
            (sub_type,) = struct.unpack_from(sub_endian + "I", blob, off + 1)
            sub_stride = 2 + (1 if sub_type // 1000 in (1, 2) else 2 if sub_type // 1000 == 3 else 0)
            rings, off = _wkb_polygon(blob, off + 5, sub_endian, sub_stride)
            polys.append(rings)
        return polys
    return []


def _decimate(ring: np.ndarray, target: int = RING_TARGET) -> np.ndarray:
    if len(ring) <= target:
        return ring
    return ring[:: int(np.ceil(len(ring) / target))]


def load_land() -> tuple[list[np.ndarray], list[np.ndarray], tuple[float, float, float, float]] | None:
    """Проецированные кольца: экстерьеры, дыры, bounds. None при сбое — карта деградирует в точечную."""
    if not GPKG.exists():
        log.warning("нет %s — рендер без подложки", GPKG)
        return None
    con = sqlite3.connect(f"file:{GPKG}?mode=ro", uri=True)
    try:
        (table,) = con.execute(
            "SELECT table_name FROM gpkg_contents WHERE data_type='features'").fetchone()
        (geom_col,) = con.execute(
            "SELECT column_name FROM gpkg_geometry_columns WHERE table_name=?",
            (table,)).fetchone()
        exteriors: list[np.ndarray] = []
        holes: list[np.ndarray] = []
        n_pts = 0
        for (blob,) in con.execute(f"SELECT {geom_col} FROM {table}"):
            for rings in decode_gpkg_geom(blob):
                for i, ring in enumerate(rings):
                    x, y = albers_russia(ring[:, 0], ring[:, 1])
                    pts = np.column_stack([x, y])
                    pts = _decimate(pts)
                    n_pts += len(pts)
                    (exteriors if i == 0 else holes).append(pts)
        all_pts = np.vstack(exteriors)
        bounds = (all_pts[:, 0].min(), all_pts[:, 0].max(),
                  all_pts[:, 1].min(), all_pts[:, 1].max())
        log.info("подложка: %d колец, %d вершин после децимации",
                 len(exteriors) + len(holes), n_pts)
        return exteriors, holes, bounds
    finally:
        con.close()


# ------------------------------------------------------------------ данные
def load_points() -> pd.DataFrame:
    nodes = pd.read_parquet(NODES, columns=["territory_id", "lat", "lon",
                                            "pop_2024", "name", "region_name"])
    labels = pd.read_parquet(LABELS, columns=["territory_id", "leiden_consensus"])
    df = nodes.merge(labels, on="territory_id", validate="one_to_one")
    if df[["lat", "lon", "pop_2024"]].isna().any().any():
        raise ValueError("NaN в lat/lon/pop_2024 — карта по неполным координатам запрещена")
    x, y = albers_russia(df["lon"].to_numpy(float), df["lat"].to_numpy(float))
    df["x"], df["y"] = x, y
    df["s"] = np.maximum(df["pop_2024"] / df["pop_2024"].max() * S_MAX, S_MIN)
    log.info("точек: %d; состав: %s", len(df), df["leiden_consensus"].value_counts().to_dict())
    return df


# ------------------------------------------------------------------ рендер
def _draw_land(ax: plt.Axes, land, face: str, with_holes: bool) -> None:
    exteriors, holes, _ = land
    ax.add_collection(PolyCollection(exteriors, facecolors=face,
                                     edgecolors="none", zorder=0))
    if with_holes and holes:
        # дыры-анклавы заливаем цветом земли, не белым: белое на бумаге
        # читается как «нет данных», а это территория МО-оболочек
        ax.add_collection(PolyCollection(holes, facecolors=face,
                                         edgecolors="none", zorder=0.1))


def _scatter_types(ax: plt.Axes, df: pd.DataFrame, s_scale: float, with_edges: bool) -> None:
    for k in (0, 2, 1):  # порядок = zorder: K0 снизу, K1 сверху
        sub = df[df["leiden_consensus"] == k]
        kw = dict(s=sub["s"] * s_scale, c=COLORS[k], alpha=ALPHA[k],
                  zorder=ZORDER[k], linewidths=0)
        if with_edges and k != 0:
            kw.update(EDGE_ACCENT)
        ax.scatter(sub["x"], sub["y"], **kw)


def _finish(ax: plt.Axes, bounds: tuple[float, float, float, float], pad_frac: float) -> None:
    x0, x1, y0, y1 = bounds
    ax.set_aspect("equal")
    ax.set_xlim(x0 - pad_frac * (x1 - x0), x1 + pad_frac * (x1 - x0))
    ax.set_ylim(y0 - pad_frac * (y1 - y0), y1 + pad_frac * (y1 - y0))
    ax.axis("off")


def _size_legend(ax: plt.Axes, pop_max: float) -> None:
    """Референс-круги канала размера: площадь точки ∝ населению МО.

    Line2D markersize — диаметр в pt, scatter s — площадь в pt²:
    диаметр = 2·sqrt(s/π).
    """
    refs = [(100_000, "100 тыс."), (500_000, "500 тыс."), (1_500_000, "1,5 млн")]
    handles = []
    for pop, lab in refs:
        s = max(pop / pop_max * S_MAX, S_MIN)
        handles.append(Line2D([0], [0], marker="o", ls="none",
                              markersize=2 * np.sqrt(s / np.pi),
                              markerfacecolor="#C4C9CE", markeredgecolor=MUTED,
                              markeredgewidth=0.6, label=lab))
    leg = ax.legend(handles=handles, loc="lower right", frameon=False,
                    fontsize=9.5, title="размер точки ∝ населению",
                    title_fontsize=9.5, handletextpad=0.9, labelspacing=1.0,
                    borderaxespad=0.2, labelcolor=INK)
    leg.get_title().set_color(MUTED)
    ax.add_artist(leg)


def _annotate(ax: plt.Axes, df: pd.DataFrame, mask: pd.Series, text: str,
              dx_pt: float, dy_pt: float, ha: str = "left") -> None:
    """Выноска к крупнейшему (по населению) МО из маски; leader-line 0.5 pt."""
    sub = df[mask]
    if sub.empty:
        log.warning("аннотация «%s»: узел не найден — пропущена", text)
        return
    row = sub.loc[sub["pop_2024"].idxmax()]
    ax.annotate(text, xy=(row["x"], row["y"]), xytext=(dx_pt, dy_pt),
                textcoords="offset points", fontsize=10, color=INK,
                ha=ha, va="center",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.5,
                                shrinkA=2, shrinkB=3),
                zorder=5)


def render_main(df: pd.DataFrame, land) -> Path:
    fig, ax = plt.subplots(figsize=(12.0, 6.9))
    fig.patch.set_facecolor(PAPER)
    if land is not None:
        _draw_land(ax, land, face=LAND_FACE, with_holes=True)
        bounds = land[2]
    else:
        bounds = (df["x"].min(), df["x"].max(), df["y"].min(), df["y"].max())
    _scatter_types(ax, df, s_scale=1.0, with_edges=True)
    _finish(ax, bounds, pad_frac=0.02)

    handles = []
    for k in (0, 1, 2):
        n = int((df["leiden_consensus"] == k).sum())
        handles.append(Line2D([0], [0], marker="o", ls="none", markersize=7,
                              markerfacecolor=COLORS[k], markeredgecolor="none",
                              label=f"{TYPE_NAMES[k]} · n={n}"))
    leg1 = ax.legend(handles=handles, loc="lower left", frameon=False,
                     fontsize=10.5, handletextpad=0.4, borderaxespad=0.2,
                     labelcolor=INK)
    ax.add_artist(leg1)
    _size_legend(ax, float(df["pop_2024"].max()))

    _annotate(ax, df, df["region_name"] == "Москва", "Москва", 16, -30)
    _annotate(ax, df, df["region_name"] == "Санкт-Петербург",
              "Санкт-Петербург", 18, 14)
    _annotate(ax, df, df["name"].str.contains("Азнакаев", case=False, na=False),
              "Азнакаевский район", 14, 20)
    _annotate(ax, df, df["region_name"] == "Калининградская область",
              "Калининградская обл.", -10, -26, ha="right")

    fig.text(0.005, 0.004,
             "Данные: СберИндекс 2023–2024 · границы МО: ОКТМО "
             "(t_dict_municipal_districts_poly.gpkg) · проекция: Альберса "
             "50°/70°N, меридиан 100°E",
             fontsize=7, color=FAINT, ha="left", va="bottom")

    out = FIG / "F0_map.png"
    fig.savefig(out, dpi=300, facecolor=PAPER, bbox_inches="tight",
                pad_inches=0.05)
    plt.close(fig)
    return out


def render_mini(df: pd.DataFrame, land) -> Path:
    """Монохромная база (все типы серым) + акценты K1/K2, без легенды."""
    fig, ax = plt.subplots(figsize=(4.4, 2.55))
    fig.patch.set_facecolor(PAPER)
    if land is not None:
        _draw_land(ax, land, face=LAND_FACE_MINI, with_holes=False)
        bounds = land[2]
    else:
        bounds = (df["x"].min(), df["x"].max(), df["y"].min(), df["y"].max())
    base = df[df["leiden_consensus"] == 0]
    ax.scatter(base["x"], base["y"], s=1.1, c="#B9C0C5", alpha=0.85,
               linewidths=0, zorder=1)
    for k in (2, 1):
        sub = df[df["leiden_consensus"] == k]
        ax.scatter(sub["x"], sub["y"], s=np.maximum(sub["s"] * 0.10, 1.8),
                   c=COLORS[k], alpha=0.95, linewidths=0, zorder=ZORDER[k])
    _finish(ax, bounds, pad_frac=0.03)

    out = FIG / "F0_map_mini.png"
    fig.savefig(out, dpi=300, facecolor=PAPER, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return out


# ------------------------------------------------------------------ entry
def main() -> None:
    ap = argparse.ArgumentParser(description="F0: карта макро-типов (герой + мини для титула)")
    ap.add_argument("--config", default="configs/default.yaml",
                    help="путь к конфигу проекта (нужен только для seed-дисциплины)")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    cfg = load_config(args.config)
    set_all_seeds(cfg.seed)

    FIG.mkdir(parents=True, exist_ok=True)
    df = load_points()
    land = load_land()
    for out in (render_main(df, land), render_mini(df, land)):
        log.info("готово: %s (%.1f КБ)", out, out.stat().st_size / 1024)


if __name__ == "__main__":
    main()
