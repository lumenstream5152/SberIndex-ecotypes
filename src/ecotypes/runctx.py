"""Контекст прогона: outputs/<run_id>/ со снапшотом конфига, run.log, metrics.json.
Связка «цифра в отчёте ↔ конфиг ↔ код» проверяема за 10 секунд."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import Config


class RunContext:
    def __init__(self, cfg: Config, config_name: str, stage: str, out_root: str | Path = "outputs"):
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")  # секунды: параллельные процессы не должны делить каталог
        try:
            git7 = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True, text=True, check=True,
            ).stdout.strip()
        except Exception:
            git7 = "nogit"
        self.run_id = f"{ts}_{config_name}_{git7}"
        self.stage = stage
        self.dir = Path(out_root) / self.run_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self._t0 = time.time()
        # снапшот эффективного конфига
        with open(self.dir / "config.snapshot.yaml", "w", encoding="utf-8") as f:
            f.write(cfg.model_dump_json(indent=2))
        self._log_lines: list[str] = [
            f"run_id: {self.run_id}",
            f"stage: {stage}",
            f"git: {git7}",
            f"python: {sys.version.split()[0]}",
            f"platform: {platform.platform()}",
            f"started: {datetime.now().isoformat()}",
        ]

    def log(self, msg: str) -> None:
        line = f"[{time.time() - self._t0:8.1f}s] {msg}"
        self._log_lines.append(line)
        print(line, flush=True)

    def write_metrics(self, metrics: dict) -> Path:
        p = self.dir / "metrics.json"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
        return p

    def close(self) -> None:
        self._log_lines.append(f"finished: {datetime.now().isoformat()}")
        with open(self.dir / "run.log", "w", encoding="utf-8") as f:
            f.write("\n".join(self._log_lines) + "\n")
