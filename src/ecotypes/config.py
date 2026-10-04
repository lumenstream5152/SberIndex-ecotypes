"""Единственная точка чтения конфигурации. Pydantic strict: опечатка в ключе →
понятная ValidationError на загрузке, а не KeyError в глубине пайплайна."""
from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataCfg(_Strict):
    start: str
    end: str
    min_months: int
    categories: list[str]
    total_category: str
    sample_n: int | None = None  # только smoke-режим


class PanelCfg(_Strict):
    clr_clip: float
    seasonality: Literal["month_of_year_log"]
    market_residual: bool


class GeoCfg(_Strict):
    enabled: bool
    knn_k: int
    lambda_km: float
    lambda_ablation: list[float] = [100.0, 150.0, 250.0]


class DtwCfg(_Strict):
    window: int


class GraphCfg(_Strict):
    similarity: str
    knn_k: int
    knn_sym: Literal["union", "mutual"]
    fdr_q: float
    safety_top: int
    geo: GeoCfg
    dtw: DtwCfg


class LeidenCfg(_Strict):
    partition: Literal["RBConfiguration", "CPM"]
    gamma: float
    gammas_sweep: list[float]
    seeds_per_snapshot: int


class KMeansCfg(_Strict):
    ks: list[int]
    n_init: int


class EvaCfg(_Strict):
    alpha: float


class KefrinCfg(_Strict):
    rho: float
    xi: float
    k: int


class ClusterCfg(_Strict):
    leiden: LeidenCfg
    kmeans: KMeansCfg
    eva: EvaCfg
    kefrin: KefrinCfg


class DynamicsCfg(_Strict):
    tau_inherit: float
    metric: Literal["jaccard"]
    smoothing_window: int


class IcviCfg(_Strict):
    panel: list[str]
    mq_variant: Literal["turbomq", "mancoridis"]
    sdbw_alg_noise: Literal["bind", "comb"]
    random_z_runs: int


class SyntheticCfg(_Strict):
    replicas: int
    grid_K: list[int]
    grid_drift: list[float]
    alpha_btw: float
    alpha_within: float
    alpha_proto: float
    mu_edge: float
    lfr_mu: list[float]


class InterpretCfg(_Strict):
    surrogate_max_depth: int
    mirkin_top_features: int
    shap_samples: int


class SiteCfg(_Strict):
    max_mb: int


class Config(_Strict):
    seed: int
    data: DataCfg
    panel: PanelCfg
    graph: GraphCfg
    cluster: ClusterCfg
    dynamics: DynamicsCfg
    icvi: IcviCfg
    synthetic: SyntheticCfg
    interpret: InterpretCfg
    site: SiteCfg


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | Path, overrides: str | Path | None = None) -> Config:
    """Читает YAML; overrides (например smoke.yaml) сливаются поверх."""
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if overrides is not None:
        with open(overrides, encoding="utf-8") as f:
            raw = _deep_merge(raw, yaml.safe_load(f))
    return Config.model_validate(raw)


def load_prereg(path: str | Path = "configs/prereg.yaml") -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
