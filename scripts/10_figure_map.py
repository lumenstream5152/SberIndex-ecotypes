"""10: герой-карта макро-типов (F0) для PDF-презентации СберИндекс-2026.

Точечная карта 2016 МО поверх сплошной подложки-контуров: цвет =
leiden_consensus (K0/K1/K2), размер точки ∝ sqrt(pop_2024).

Проекция — коническая равновеликая Альберса (параллели 50°N/70°N,
осевой меридиан 100°E): стандартный вид карты России.

Выходы:
  report/figures/F0_map.png       — герой слайда типов (3252×1801 px);
  report/figures/F0_map_mini.png  — мини-карта для титула (1035×566 px).
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
from PIL import Image
from matplotlib.collections import PolyCollection
from matplotlib.lines import Line2D

from ecotypes.config import load_config
from ecotypes.seeds import set_all_seeds

NODES = Path("data/processed/nodes_static.parquet")
LABELS = Path("outputs/main/labels.parquet")
GPKG = Path("data/raw/t_dict_municipal_districts_poly.gpkg")
FIG = Path("report/figures")

# Цветовая палитра: гармония высокого контраста на светлой бумаге
COLORS = {
    0: "#C89B38",  # K0 Периферийная Россия — теплая охра (1654 МО, 70.4% населения)
    1: "#2563EB",  # K1 Сервисные городские ядра — королевский синий (196 МО, 19.2% населения)
    2: "#991B1B",  # K2 Северо-Запад — благородный бордовый (166 МО, 10.5% населения)
}
TYPE_NAMES = {
    0: "Периферийная Россия",
    1: "Сервисные городские ядра",
    2: "Северо-Запад",
}
TYPE_POP_PCT = {
    0: "70.4%",
    1: "19.2%",
    2: "10.5%",
}

PAPER = "#FFFDF7"          # фон деки
LAND_FACE = "#EBEFF2"      # бесшовная подложка суши
LAND_EDGE = "#EBEFF2"
INK = "#1E293B"
MUTED = "#475569"
FAINT = "#94A3B8"

S_MAX = 160.0   # площадь маркера (pt²) для max pop; сбалансировано против наложения
S_MIN = 2.5     # минимальный размер малых МО

# Альберс для России
ALBERS_LAT1, ALBERS_LAT2 = 50.0, 70.0   # стандартные параллели, °N
ALBERS_LON0 = 100.0                     # осевой меридиан, °E
ALBERS_LAT0 = 60.0                      # параллель начала отсчёта, °N

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 300,
    "text.color": INK,
})

log = logging.getLogger("figure_map")


def albers_russia(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Коническая равновеликая Альберса, dlon приведен к [-180, 180)."""
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


def _wkb_polygon(buf: bytes, off: int, endian: str, stride: int) -> tuple[list[np.ndarray], int]:
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
    if blob[:2] != b"GP":
        raise ValueError("не GeoPackageBinary")
    env_code = (blob[3] >> 1) & 0x07
    off = 8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[env_code]

    endian = "<" if blob[off] == 1 else ">"
    (gtype,) = struct.unpack_from(endian + "I", blob, off + 1)
    off += 5
    base, dim = gtype % 1000, gtype // 1000
    stride = 2 + (1 if dim in (1, 2) else 2 if dim == 3 else 0)

    if base == 3:  # Polygon
        rings, _ = _wkb_polygon(blob, off, endian, stride)
        return [rings]
    if base == 6:  # MultiPolygon
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


def load_land() -> tuple[list[np.ndarray], tuple[float, float, float, float]] | None:
    """Загрузка геометрий суши без артефактного прореживания (сплошной силуэт)."""
    if not GPKG.exists():
        log.warning("нет %s — рендер без подложки", GPKG)
        return None
    con = sqlite3.connect(f"file:{GPKG}?mode=ro", uri=True)
    try:
        exteriors: list[np.ndarray] = []
        for (blob,) in con.execute("SELECT geom FROM t_dict_municipal_districts_poly"):
            for rings in decode_gpkg_geom(blob):
                if rings:
                    x, y = albers_russia(rings[0][:, 0], rings[0][:, 1])
                    exteriors.append(np.column_stack([x, y]))
        all_pts = np.vstack(exteriors)
        bounds = (all_pts[:, 0].min(), all_pts[:, 0].max(),
                  all_pts[:, 1].min(), all_pts[:, 1].max())
        log.info("подложка загружена: %d полигонов суши", len(exteriors))
        return exteriors, bounds
    finally:
        con.close()


def load_points() -> pd.DataFrame:
    nodes = pd.read_parquet(NODES, columns=["territory_id", "lat", "lon",
                                            "pop_2024", "name", "region_name"])
    labels = pd.read_parquet(LABELS, columns=["territory_id", "leiden_consensus"])
    df = nodes.merge(labels, on="territory_id", validate="one_to_one")
    if df[["lat", "lon", "pop_2024"]].isna().any().any():
        raise ValueError("NaN в lat/lon/pop_2024")
    x, y = albers_russia(df["lon"].to_numpy(float), df["lat"].to_numpy(float))
    df["x"], df["y"] = x, y
    df["s"] = np.maximum(df["pop_2024"] / df["pop_2024"].max() * S_MAX, S_MIN)
    return df


def _annotate_city(ax: plt.Axes, df: pd.DataFrame, mask: pd.Series, text: str,
                   dx_pt: float, dy_pt: float, ha: str = "left") -> None:
    sub = df[mask]
    if sub.empty:
        return
    row = sub.loc[sub["pop_2024"].idxmax()]
    ax.annotate(text, xy=(row["x"], row["y"]), xytext=(dx_pt, dy_pt),
                textcoords="offset points", fontsize=9.5, fontweight="bold", color="#1E293B",
                ha=ha, va="center",
                bbox=dict(boxstyle="round,pad=0.28", fc="#FFFFFF", ec="#CBD5E1", lw=0.7, alpha=0.96),
                arrowprops=dict(arrowstyle="->", color="#475569", lw=0.8, shrinkA=3, shrinkB=4),
                zorder=10)


def render_main(df: pd.DataFrame, land) -> Path:
    fig, ax = plt.subplots(figsize=(12.0, 6.9))
    fig.patch.set_facecolor(PAPER)

    if land is not None:
        exteriors, bounds = land
        # Сплошная заливка с перекрытием границ исключает белые щели между полигонами
        ax.add_collection(PolyCollection(exteriors, facecolors=LAND_FACE, edgecolors=LAND_EDGE, linewidths=1.2, zorder=0))
    else:
        bounds = (df["x"].min(), df["x"].max(), df["y"].min(), df["y"].max())

    # Точки: K0 (периферия), затем K2 (северо-запад), затем K1 (ядра)
    for k in (0, 2, 1):
        sub = df[df["leiden_consensus"] == k]
        if k == 0:
            ax.scatter(sub["x"], sub["y"], s=sub["s"], c=COLORS[k], alpha=0.68, zorder=1, linewidths=0)
        else:
            ax.scatter(sub["x"], sub["y"], s=sub["s"], c=COLORS[k], alpha=0.92, zorder=3 if k == 1 else 2,
                       linewidths=0.4, edgecolors="white")

    x0, x1, y0, y1 = bounds
    ax.set_aspect("equal")
    ax.set_xlim(x0 - 0.02 * (x1 - x0), x1 + 0.02 * (x1 - x0))
    ax.set_ylim(y0 - 0.02 * (y1 - y0), y1 + 0.02 * (y1 - y0))
    ax.axis("off")

    # Аккуратные подписи ключевых центров (без лишних случайных выносок)
    _annotate_city(ax, df, df["region_name"] == "Москва", "Москва", 28, 14, ha="left")
    _annotate_city(ax, df, df["region_name"] == "Санкт-Петербург", "Санкт-Петербург", 26, 16, ha="left")
    _annotate_city(ax, df, df["region_name"] == "Калининградская область", "Калининград", -16, -20, ha="right")

    # Карточка типов (слева снизу)
    handles_types = []
    for k in (0, 1, 2):
        n = int((df["leiden_consensus"] == k).sum())
        pct = TYPE_POP_PCT[k]
        handles_types.append(Line2D([0], [0], marker="o", ls="none", markersize=8.0,
                                    markerfacecolor=COLORS[k], markeredgecolor="white", markeredgewidth=0.5,
                                    label=f"{TYPE_NAMES[k]} · {n} МО ({pct} населения)"))

    leg = ax.legend(handles=handles_types, loc="lower left", bbox_to_anchor=(0.02, 0.05),
                    frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", framealpha=0.96,
                    fontsize=9.5, title="Макро-типы безналичной России (k = 3)",
                    title_fontsize=10.5, handletextpad=0.7, labelspacing=0.6, borderpad=0.8)
    leg.get_title().set_fontweight("bold")
    leg.get_title().set_color("#0F172A")
    ax.add_artist(leg)

    # Карточка масштаба (справа снизу)
    refs = [(100_000, "100 тыс. чел."), (500_000, "500 тыс. чел."), (1_500_000, "1,5 млн чел.")]
    pop_max = float(df["pop_2024"].max())
    handles_size = []
    for pop, lab in refs:
        s = max(pop / pop_max * S_MAX, S_MIN)
        handles_size.append(Line2D([0], [0], marker="o", ls="none",
                                   markersize=2 * np.sqrt(s / np.pi),
                                   markerfacecolor="#CBD5E1", markeredgecolor="#475569",
                                   markeredgewidth=0.7, label=lab))
    leg_size = ax.legend(handles=handles_size, loc="lower right", bbox_to_anchor=(0.98, 0.05),
                         frameon=True, facecolor="#FFFFFF", edgecolor="#CBD5E1", framealpha=0.96,
                         fontsize=9.0, title="Размер точки ∝ населению МО",
                         title_fontsize=10.0, handletextpad=0.8, labelspacing=0.6, borderpad=0.8)
    leg_size.get_title().set_fontweight("bold")
    leg_size.get_title().set_color("#475569")
    ax.add_artist(leg_size)

    fig.text(0.02, 0.012,
             "Данные: СберИндекс 2023–2024 · границы МО: ОКТМО · коническая проекция Альберса (50°/70°N, 100°E)",
             fontsize=7.5, color=FAINT, ha="left", va="bottom")

    out = FIG / "F0_map.png"
    fig.savefig(out, dpi=300, facecolor=PAPER, bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)

    # Нормализация в точный размер (3252×1801 px) для полной совместимости со слайдом PDF
    im = Image.open(out).convert("RGB")
    if im.size != (3252, 1801):
        im = im.resize((3252, 1801), Image.Resampling.LANCZOS)
        im.save(out, dpi=(300, 300))

    return out


def render_mini(df: pd.DataFrame, land) -> Path:
    fig, ax = plt.subplots(figsize=(4.4, 2.55))
    fig.patch.set_facecolor(PAPER)

    if land is not None:
        exteriors, bounds = land
        ax.add_collection(PolyCollection(exteriors, facecolors="#E4E9EC", edgecolors="#E4E9EC", linewidths=0.8, zorder=0))
    else:
        bounds = (df["x"].min(), df["x"].max(), df["y"].min(), df["y"].max())

    base = df[df["leiden_consensus"] == 0]
    ax.scatter(base["x"], base["y"], s=1.1, c="#B9C0C5", alpha=0.85, linewidths=0, zorder=1)
    for k in (2, 1):
        sub = df[df["leiden_consensus"] == k]
        ax.scatter(sub["x"], sub["y"], s=np.maximum(sub["s"] * 0.10, 1.8),
                   c=COLORS[k], alpha=0.95, linewidths=0, zorder=3 if k == 1 else 2)

    x0, x1, y0, y1 = bounds
    ax.set_aspect("equal")
    ax.set_xlim(x0 - 0.03 * (x1 - x0), x1 + 0.03 * (x1 - x0))
    ax.set_ylim(y0 - 0.03 * (y1 - y0), y1 + 0.03 * (y1 - y0))
    ax.axis("off")

    out = FIG / "F0_map_mini.png"
    fig.savefig(out, dpi=300, facecolor=PAPER, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    im = Image.open(out).convert("RGB")
    if im.size != (1035, 566):
        im = im.resize((1035, 566), Image.Resampling.LANCZOS)
        im.save(out, dpi=(300, 300))

    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="F0: карта макро-типов (герой + мини для титула)")
    ap.add_argument("--config", default="configs/default.yaml",
                    help="путь к конфигу проекта (для seed-дисциплины)")
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
