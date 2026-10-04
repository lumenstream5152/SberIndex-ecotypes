"""01: скачать сырые данные в data/raw/ с проверкой SHA256 (см. DATA.md).

Идемпотентен: файл существует и хэш сошёлся → skip. sberbank.com отдаёт
нестандартную цепочку (НУЦ): сначала обычный TLS, при ошибке — curl -k
с предупреждением; целостность всегда гарантирует SHA256.
"""
from __future__ import annotations

import hashlib
import shutil
import ssl
import subprocess
import sys
import urllib.request
import zipfile
from pathlib import Path

RAW = Path("data/raw")
HL = RAW / "hackathon" / "hackathonlicence"

# файл → sha256 (DATA.md, зафиксировано 04.10.2026)
EXPECTED = {
    HL / "consumption.parquet": "9833ddaaee7b2a182ed4cceeed16469031700ea87d508bd976150c6770ef8a61",
    HL / "connection.parquet": "20cbd5213d3ac1d0a867f097b485431811614b283d7366eb5cba9f65dc80493d",
    HL / "market_access.parquet": "434258afe322b7e6e6610b2552d129ae47094613de72dae3d13f6965a28d1dc1",
    RAW / "t_dict_municipal_districts.xlsx": "4150658c3298fbc87ed79838f503f3a5b9a28da33257ff803c7d9231ddb775d6",
    RAW / "t_dict_municipal_districts_poly.gpkg": "e62027630d48e4a13f9b6d173dd074d7358706743fe859ca3b7fa30fefa9813a",
}

URL_ZIP = "https://www.sberbank.com/common/img/uploaded/files/pdf/sberindex/hackathonlicence.zip"
URL_DICT = "https://www.sberbank.com/common/files/t_dict_municipal.rar"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path) -> None:
    print(f"GET {url}")
    try:
        urllib.request.urlretrieve(url, dest)
    except ssl.SSLError:
        print("ВНИМАНИЕ: нестандартная цепочка сертификатов (НУЦ), "
              "переход на curl -k; целостность проверяется SHA256", file=sys.stderr)
        subprocess.run(["curl", "-kfL", url, "-o", str(dest)], check=True)


def main() -> None:
    missing = [p for p in EXPECTED if not p.exists() or sha256(p) != EXPECTED[p]]
    if not missing:
        print("все файлы на месте, хэши сошлись — skip")
        return

    need_hackathon = any("hackathon" in str(p) for p in missing)
    need_dict = any(p.suffix in (".xlsx", ".gpkg") for p in missing)

    if need_hackathon:
        z = RAW / "hackathonlicence.zip"
        if not z.exists():
            RAW.mkdir(parents=True, exist_ok=True)
            download(URL_ZIP, z)
        with zipfile.ZipFile(z) as zf:
            zf.extractall(RAW)
        print("распакован hackathonlicence.zip")

    if need_dict:
        print(
            "Справочник МО: скачайте вручную " + URL_DICT +
            " (rar; нужен unrar/unar), распакуйте t_dict_municipal_districts.xlsx "
            "и t_dict_municipal_districts_poly.gpkg в data/raw/, перезапустите make data",
            file=sys.stderr,
        )

    bad = []
    for p in EXPECTED:
        if not p.exists():
            bad.append(f"{p}: отсутствует")
        elif sha256(p) != EXPECTED[p]:
            bad.append(f"{p}: sha256 не сошёлся")
    if bad:
        print("ПРОБЛЕМА С ДАННЫМИ:\n" + "\n".join(bad), file=sys.stderr)
        sys.exit(1)
    print("данные готовы, хэши сошлись")


if __name__ == "__main__":
    main()
