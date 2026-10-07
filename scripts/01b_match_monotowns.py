"""01b: матчинг перечня моногородов 1398-р к справочнику СберИндекса.

Документирует эвристику, ранее существовавшую только как файл (JR1/A8).
Детерминированный алгоритм: внутри региона пробуем (по порядку)
(1) нормализованное имя МО == municipal_district_name;
(2) нормализованный админцентр == municipal_district_center;
(3) нормализованное имя МО оканчивается на имя админцентра (и обратно).
Нормализация: lowercase, ё→е, убраны типы («городской округ», «муниципальный
район», «город», «г.», «пгт»…), знаки препинания, лишние пробелы.
Справочник: только актуальные строки (year_to==9999; исключения-реформы
1471/1487/1847 берутся по последнему year_from — см. panel.py).
Коллизии (n_hits>1) резолвятся точным совпадением по админцентру, иначе —
первым по territory_id с флагом n_hits (честно видимым в выходе).

Вход: data/external/monotowns_1398r_full.csv + data/raw справочник.
Выход: data/external/monotowns_1398r_matched.csv + _unmatched.csv + metrics.
"""
from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from pathlib import Path

import pandas as pd

from ecotypes.config import load_config
from ecotypes.runctx import RunContext
from ecotypes.seeds import set_all_seeds

log = logging.getLogger("match_monotowns")

STOP = ["городской округ", "муниципальный округ", "муниципальный район",
        "городской округ город", "город", "поселок городского типа",
        "пгт", "г.", "г ", "р-н", "район"]


def norm(s: object) -> str:
    s = unicodedata.normalize("NFKC", str(s)).lower().replace("ё", "е")
    s = re.sub(r"[\u00ab\u00bb\"'(),.\-–—/]", " ", s)
    for w in STOP:
        s = s.replace(w, " ")
    return re.sub(r"\s+", " ", s).strip()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overrides", default=None)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(name)s %(message)s")
    cfg = load_config(args.config, overrides=args.overrides)
    set_all_seeds(cfg.seed)
    name = args.config.split("/")[-1].removesuffix(".yaml")
    ctx = RunContext(cfg, config_name=name, stage="01b_match_monotowns")

    ext = Path("data/external")
    full = pd.read_csv(ext / "monotowns_1398r_full.csv")
    ref = pd.read_excel("data/raw/t_dict_municipal_districts.xlsx")
    REF_EXCEPT = {1471, 1487, 1847}
    cur = ref[(ref.year_to == 9999) | (ref.territory_id.isin(REF_EXCEPT))]
    cur = (cur.sort_values("year_from").groupby("territory_id", as_index=False)
              .tail(1))

    by_region: dict[str, pd.DataFrame] = {}
    for region, g in cur.groupby("region_name"):
        g = g.copy()
        g["n_name"] = g.municipal_district_name.map(norm)
        g["n_center"] = g.municipal_district_center.map(norm)
        g["n_short"] = g.municipal_district_name_short.map(norm)
        by_region[str(region)] = g

    matched, unmatched = [], []
    for _, r in full.iterrows():
        g = by_region.get(str(r.region))
        if g is None:
            unmatched.append({**r.to_dict(), "reason": "регион не найден"})
            continue
        n_mo, n_c = norm(r.mo_name_1398), norm(r.admin_center)
        hits = g[g.n_name == n_mo]
        if hits.empty:
            hits = g[g.n_center == n_c]
        if hits.empty:
            hits = g[g.n_name.str.endswith(n_c) | g.n_center.str.endswith(n_c)
                     | g.n_short.map(lambda x: x == n_c)]
        if hits.empty:
            unmatched.append({**r.to_dict(), "reason": "нет совпадения имени"})
            continue
        exact = hits[hits.n_center == n_c]
        use = exact if len(exact) == 1 else hits
        use = use.sort_values("territory_id")
        row = {**r.to_dict(), "territory_id": int(use.iloc[0].territory_id),
               "oktmo": str(use.iloc[0].oktmo),
               "mo_name_sberindex": str(use.iloc[0].municipal_district_name),
               "n_hits": int(len(hits))}
        matched.append(row)

    m = pd.DataFrame(matched)
    u = pd.DataFrame(unmatched)
    m.to_csv(ext / "monotowns_1398r_matched.csv", index=False)
    u.to_csv(ext / "monotowns_1398r_unmatched.csv", index=False)
    ctx.log(f"сматчено {len(m)}/{len(full)} (n_hits>1 у "
            f"{int((m.n_hits > 1).sum())}); не сматчено {len(u)}")
    ctx.write_metrics({"n_full": int(len(full)), "n_matched": int(len(m)),
                       "n_unmatched": int(len(u)),
                       "n_multi_hits": int((m.n_hits > 1).sum())})
    ctx.close()


if __name__ == "__main__":
    main()
