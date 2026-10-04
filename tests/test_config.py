"""Тест 1 из спеки (research/36 §5): конфиг валидируется, smoke сливается,
неизвестный ключ и битый enum → читаемая ошибка."""
import pytest
import yaml

from ecotypes.config import load_config


def test_default_config_validates():
    cfg = load_config("configs/default.yaml")
    assert cfg.graph.knn_k == 10
    assert cfg.cluster.leiden.partition == "RBConfiguration"
    assert cfg.icvi.mq_variant == "turbomq"


def test_smoke_merge():
    cfg = load_config("configs/default.yaml", overrides="configs/smoke.yaml")
    assert cfg.data.sample_n == 200
    assert cfg.cluster.leiden.gammas_sweep == [1.0]
    # непереопределённое остаётся из default
    assert cfg.graph.fdr_q == 0.05


def test_unknown_key_rejected(tmp_path):
    base = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
    base["tesko"] = 1
    p = tmp_path / "bad.yaml"
    yaml.safe_dump(base, open(p, "w", encoding="utf-8"))
    with pytest.raises(Exception, match="tesko"):
        load_config(p)


def test_bad_enum_rejected(tmp_path):
    base = yaml.safe_load(open("configs/default.yaml", encoding="utf-8"))
    base["cluster"]["leiden"]["partition"] = "LouvainStyle"
    p = tmp_path / "bad2.yaml"
    yaml.safe_dump(base, open(p, "w", encoding="utf-8"))
    with pytest.raises(Exception):
        load_config(p)


def test_prereg_composite_weights_sum():
    prereg = yaml.safe_load(open("configs/prereg.yaml", encoding="utf-8"))
    w_sim = prereg["similarity"]["composite_weights"]
    assert abs(sum(w_sim.values()) - 1.0) < 1e-9
    w_m = prereg["method"]["composite_weights"]
    assert abs(sum(w_m.values()) - 1.0) < 1e-9
    w_icvi = prereg["method"]["icvi_subweights"]
    assert abs(sum(w_icvi.values()) - 0.30) < 1e-9
