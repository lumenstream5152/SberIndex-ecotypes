"""Adversarial audit tests for sberindex-ecotypes.

Strict invariants:
1. All similarity measures M1..M11 exist in dispatch dictionary and compute valid matrices.
2. Anonymity is strictly enforced in git history (no personal identity leaks).
3. No processed/smoke data is tracked in git.
4. find_run is immune to smoke runs by default.
5. All 17 script runners are present in Makefile.
6. Report and presentation have zero numerical contradictions on noise envelope and driver metrics.
"""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

import numpy as np
import pytest

from ecotypes.measures import MeasureData
import ecotypes.measures as ms
from ecotypes.interpret import find_run


REPO_ROOT = Path(__file__).resolve().parents[1]


# -----------------------------------------------------------------------------
# 1. Dispatcher Completeness (M1..M11)
# -----------------------------------------------------------------------------

def test_measures_dispatcher_completeness():
    """Adversarially verify that every candidate measure M1..M11 can be dispatched
    without KeyError and produces a valid finite similarity matrix."""
    n, t = 20, 24
    rng = np.random.default_rng(42)

    log_all = rng.normal(10.0, 1.0, (n, t))
    sa_all = ms.sa_from_log(log_all)
    shares = rng.dirichlet(np.ones(6), size=(n, t))
    clr = ms.clr_transform(shares)
    vals = rng.normal(5.0, 0.5, (n, t, 5))
    pop = rng.uniform(1000, 100000, size=n)
    hw = (np.array([0, 1], dtype=np.int32), np.array([1, 2], dtype=np.int32), np.array([10.0, 20.0]))
    region = rng.integers(1, 80, size=n)

    data = MeasureData(
        tids=np.arange(n, dtype=np.int32),
        log_all=log_all,
        sa_all=sa_all,
        shares=shares,
        clr=clr,
        vals=vals,
        pop=pop,
        hw=hw,
        region=region,
    )

    alpha9 = 1.0
    k = 5

    fns = {
        "M1": ms.sim_M1,
        "M2": ms.sim_M2,
        "M3": ms.sim_M3,
        "M4": ms.sim_M4,
        "M5": ms.sim_M5,
        "M6": ms.sim_M6,
        "M7": ms.sim_M7,
        "M8": ms.sim_M8,
        "M9": (lambda d, lvl_idx=None, grw_idx=None:
               ms.sim_M9(d, lvl_idx=lvl_idx, grw_idx=grw_idx, alpha=alpha9)),
        "M10": (lambda d, lvl_idx=None, grw_idx=None:
                ms.sim_M10(d, lvl_idx=lvl_idx, grw_idx=grw_idx, k=k)),
        "M11": ms.sim_M11,
    }

    # Verify every M1..M11 is present
    for m_idx in range(1, 12):
        m_name = f"M{m_idx}"
        assert m_name in fns, f"Missing {m_name} in similarity dispatcher!"
        res = fns[m_name](data)
        assert res is not None, f"{m_name} returned None"
        assert res.matrix.shape == (n, n), f"{m_name} shape mismatch: {res.matrix.shape}"
        assert np.all(np.isfinite(res.matrix)), f"{m_name} contains NaN or Inf"


# -----------------------------------------------------------------------------
# 2. Strict Anonymity Check
# -----------------------------------------------------------------------------

def test_git_anonymity_strict():
    """Verify that git history and branches contain zero leaks of author identity."""
    # 1. No backup branch
    branches_proc = subprocess.run(
        ["git", "branch", "-a"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "backup-before-anonymize" not in branches_proc.stdout, (
        "Fatal: de-anonymization branch 'backup-before-anonymize' still exists!"
    )

    # 2. Git log audit
    log_proc = subprocess.run(
        ["git", "log", "--all", "--format=%an|%ae|%B"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    forbidden = ["Asalio123"]
    for term in forbidden:
        assert term.lower() not in log_proc.stdout.lower(), (
            f"Fatal: Identity leak '{term}' found in git commit log!"
        )


# -----------------------------------------------------------------------------
# 3. Clean Git Tracking & .gitignore Hygiene
# -----------------------------------------------------------------------------

def test_no_tracked_cache_or_smoke_data():
    """Verify that no files under data/processed* or non-main outputs are tracked."""
    proc = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    tracked_files = [line.strip() for line in proc.stdout.splitlines() if line.strip()]

    illegal_prefixes = [
        "data/processed",
        "data/processed_smoke",
        "data/raw",
        "data/interim",
    ]
    for f in tracked_files:
        for prefix in illegal_prefixes:
            assert not f.startswith(prefix), f"Tracked illegal data file: {f}"

        if f.startswith("outputs/") and not f.startswith("outputs/main/"):
            pytest.fail(f"Tracked illegal output file outside main: {f}")

    # Check .gitignore line
    gitignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "data/processed*/" in gitignore_text, (
        ".gitignore must contain 'data/processed*/' to prevent smoke leaks"
    )


# -----------------------------------------------------------------------------
# 4. find_run Immunity to Smoke
# -----------------------------------------------------------------------------

def test_find_run_smoke_immunity(tmp_path: Path):
    """Adversarially test that find_run ignores newer smoke runs when allow_smoke=False."""
    outputs_dir = tmp_path / "outputs"
    outputs_dir.mkdir()

    # Create older production run
    prod_dir = outputs_dir / "20261006_183551_default_prod"
    prod_dir.mkdir()
    (prod_dir / "metrics.json").write_text(json.dumps({"edges_star": 12900}))
    (prod_dir / "labels.parquet").touch()

    # Create newer smoke run
    smoke_dir = outputs_dir / "20261010_150000_default_smoke_test"
    smoke_dir.mkdir()
    (smoke_dir / "metrics.json").write_text(json.dumps({"edges_star": 646}))
    (smoke_dir / "labels.parquet").touch()

    # By default, must find PROD, NOT smoke!
    chosen_prod = find_run(outputs_dir, "edges_star", allow_smoke=False)
    assert chosen_prod == prod_dir, (
        f"find_run selected {chosen_prod} instead of production {prod_dir}!"
    )
    data = json.loads((chosen_prod / "metrics.json").read_text())
    assert data["edges_star"] == 12900

    # With allow_smoke=True, finds newer smoke
    chosen_smoke = find_run(outputs_dir, "edges_star", allow_smoke=True)
    assert chosen_smoke == smoke_dir, (
        f"find_run with allow_smoke=True did not pick smoke run!"
    )
    data_smoke = json.loads((chosen_smoke / "metrics.json").read_text())
    assert data_smoke["edges_star"] == 646


# -----------------------------------------------------------------------------
# 5. Makefile Completeness
# -----------------------------------------------------------------------------

def test_makefile_all_runners_exist():
    """Verify that Makefile defines targets for all script runners, including 01b and 10."""
    makefile_text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")

    required_targets = [
        "env", "data", "match-monotowns", "panel", "graphs", "cluster",
        "dynamics", "measures", "icvi", "icvi-null", "synth", "interpret",
        "drivers", "convergence", "laglead", "aux", "map", "report",
        "deck", "reproduce", "smoke", "test"
    ]
    for target in required_targets:
        pattern = rf"^{re.escape(target)}\s*:"
        assert re.search(pattern, makefile_text, re.MULTILINE), (
            f"Missing Makefile target: '{target}'"
        )


# -----------------------------------------------------------------------------
# 6. Report and Deck Numerical Alignment
# -----------------------------------------------------------------------------

def test_report_deck_noise_envelope_alignment():
    """Verify that report §5 and deck Slide 9 both report 0.34 (1 - 0.655) and NOT 0.35."""
    report_text = (REPO_ROOT / "report" / "methodology.md").read_text(encoding="utf-8")
    deck_text = (REPO_ROOT.parent / "deck" / "deck.html").read_text(encoding="utf-8")
    stability_data = json.loads(
        (REPO_ROOT / "outputs" / "main" / "stability.json").read_text(encoding="utf-8")
    )

    # 1. Verify stability.json ground truth
    ari_null = stability_data["summary"]["ari_perturb_med_mean"]
    expected_noise_diff = round(1.0 - ari_null, 2)
    assert expected_noise_diff == 0.34, f"Ground truth noise diff is {expected_noise_diff}, not 0.34"

    # 2. Check report methodology
    assert "0.34" in report_text, "report/methodology.md must contain '0.34'"
    assert "на 0.35" not in report_text, "report/methodology.md still contains outdated '0.35'!"

    # 3. Check deck
    assert "0,34" in deck_text, "deck.html must contain '0,34'"


def test_driver_model_metrics_truthfulness():
    """Verify that driver model metrics in outputs/main match slide declarations."""
    metrics = json.loads(
        (REPO_ROOT / "outputs" / "main" / "model_metrics.json").read_text(encoding="utf-8")
    )
    pooled_h1 = metrics["h1"]["pooled"]
    pr_lgbm = pooled_h1["lgbm"]["pr_auc"]
    pr_margin = pooled_h1["margin_rank"]["pr_auc"]
    pr_logreg5 = pooled_h1["logreg5"]["pr_auc"]

    # Invariants
    assert 0.028 < pr_lgbm < 0.029, f"LGBM PR-AUC {pr_lgbm} out of range"
    assert 0.026 < pr_margin < 0.027, f"margin_rank PR-AUC {pr_margin} out of range"
    assert 0.050 < pr_logreg5 < 0.051, f"logreg5 PR-AUC {pr_logreg5} out of range"

    # Slide 2 check
    deck_text = (REPO_ROOT.parent / "deck" / "deck.html").read_text(encoding="utf-8")
    # Must NOT say 'простое правило выигрывает у бустинга (0,051 против 0,029)' without distinguishing logreg5
    assert "простое правило выигрывает у бустинга (0,051 против 0,029)" not in deck_text, (
        "Slide 2 must not conflate logreg5 (0.051) with simple rule margin_rank (0.026)"
    )


# -----------------------------------------------------------------------------
# 7. Workstream 1: CRITERIA.md Completeness & Artifact Integrity
# -----------------------------------------------------------------------------

def test_criteria_md_four_tables_and_artifacts():
    """Verify CRITERIA.md contains all 4 comprehensive tables and all referenced artifacts exist."""
    criteria_text = (REPO_ROOT / "CRITERIA.md").read_text(encoding="utf-8")

    # 1. Check all 4 table headers exist
    assert "## 1. Задачи номинации" in criteria_text
    assert "## 2. Критерии оценивания" in criteria_text
    assert "## 3. Требования Положения" in criteria_text
    assert "## 4. Учтённые замечания жюри прошлого сезона" in criteria_text

    # 2. Check all 6 criteria weights
    for weight in ["15%", "30%", "10%"]:
        assert weight in criteria_text

    # 3. Check referenced outputs/main artifacts exist
    main_dir = REPO_ROOT / "outputs" / "main"
    key_artifacts = [
        "graph_summary.json",
        "table_topology.parquet",
        "stability.json",
        "type_registry.parquet",
        "type_id_map.parquet",
        "plateau_table.parquet",
        "metrics_04_cluster.json",
        "labels.parquet",
        "labels.csv",
        "icvi_null.parquet",
        "table_methods.parquet",
        "table_B_measures.parquet",
        "events_admitted.parquet",
        "passports_macro.parquet",
        "transition_cards.parquet",
        "model_metrics.json",
        "radar_watchlist.parquet",
        "convergence_summary.json",
        "lead_summary.json",
    ]
    for artifact in key_artifacts:
        assert (main_dir / artifact).exists(), f"Artifact referenced in CRITERIA.md missing: {artifact}"


# -----------------------------------------------------------------------------
# 8. Workstream 2: External Validation (KW-Test & Zubarevich Reframing)
# -----------------------------------------------------------------------------

def test_external_validation_kruskal_wallis_and_zubarevich():
    """Adversarially compute Kruskal-Wallis test on wages and verify Zubarevich four Russias stats."""
    from scipy import stats
    import pandas as pd

    labels_df = pd.read_parquet(REPO_ROOT / "outputs" / "main" / "labels.parquet")
    nodes_df = pd.read_parquet(REPO_ROOT / "data" / "processed" / "nodes_static.parquet")

    # 1. Kruskal-Wallis on Rosstat wages
    merged = pd.merge(labels_df[["territory_id", "leiden_consensus"]],
                      nodes_df[["territory_id", "log_wage"]], on="territory_id").dropna()
    assert len(merged) == 2016, "All 2016 territories must have valid wage entries"

    merged["wage"] = np.exp(merged["log_wage"].values, dtype=np.float64)
    groups = [g["wage"].values.astype(np.float64) for _, g in merged.groupby("leiden_consensus")]
    assert len(groups) == 3, "Must have exactly 3 macro groups"

    kw_res = stats.kruskal(*groups)
    assert 588.0 < kw_res.statistic < 592.0, f"KW statistic {kw_res.statistic} out of expected 589.5 range"
    assert float(kw_res.pvalue) < 1e-100, f"KW p-value {kw_res.pvalue} must be < 1e-100"

    k = len(groups)
    n = len(merged)
    eta2_H = (kw_res.statistic - k + 1) / (n - k)
    assert 0.285 < eta2_H < 0.295, f"Wage effect size eta2_H {eta2_H:.4f} not around 0.29"

    # 2. Zubarevich 'Four Russias' exact numbers
    fr_df = pd.read_parquet(REPO_ROOT / "outputs" / "main" / "validation_four_russias.parquet")

    # Russia-1: Р1
    r1_rows = fr_df[fr_df["russia"] == "Р1"]
    total_r1 = r1_rows["n"].sum()
    assert total_r1 == 273, f"Total Russia-1 MOs should be 273, got {total_r1}"

    r1_urban = r1_rows[r1_rows["type_id"].isin([1, 2])]["n"].sum()
    assert r1_urban == 244, f"Russia-1 in types 1+2 should be 244, got {r1_urban}"
    share_r1_urban = r1_urban / total_r1
    assert round(share_r1_urban, 3) == 0.894, f"Share should round to 0.894, got {share_r1_urban}"

    # Lifts
    lift_t1 = float(r1_rows[r1_rows["type_id"] == 1]["lift"].iloc[0])
    lift_t2 = float(r1_rows[r1_rows["type_id"] == 2]["lift"].iloc[0])
    assert 5.40 < lift_t1 < 5.50, f"Type 1 lift {lift_t1} out of range"
    assert 4.35 < lift_t2 < 4.45, f"Type 2 lift {lift_t2} out of range"

    # Russia-4: 100% in Type 0
    r4_rows = fr_df[fr_df["russia"] == "Р4"]
    assert r4_rows["n"].sum() == 45
    assert r4_rows[r4_rows["type_id"] == 0]["n"].iloc[0] == 45


# -----------------------------------------------------------------------------
# 9. Workstream 3: Type Lifecycle Invariants
# -----------------------------------------------------------------------------

def test_type_lifecycle_registry_invariants():
    """Verify that type_registry.parquet and type_id_map.parquet match reported lifecycle facts."""
    import pandas as pd

    reg = pd.read_parquet(REPO_ROOT / "outputs" / "main" / "type_registry.parquet")
    assert len(reg) == 9, f"type_registry must contain exactly 9 types (0..8), found {len(reg)}"

    # Type 4: Seasonal excursion
    t4 = reg[reg["type_id"] == 4].iloc[0]
    assert t4["birth_month"] == "2023-01"
    assert t4["death_month"] == "2023-12"
    assert t4["merged_into"] == 0
    counts_t4 = json.loads(t4["n_nodes_by_month"])
    assert counts_t4["2023-11"] == 537, f"Type 4 peak in Nov 2023 should be 537, got {counts_t4.get('2023-11')}"

    # Type 7: Birth from Type 4
    t7 = reg[reg["type_id"] == 7].iloc[0]
    assert t7["birth_month"] == "2023-12"
    assert t7["parent_id"] == 4
    assert pd.isna(t7["death_month"])

    # Type 8: 1-month artifact
    t8 = reg[reg["type_id"] == 8].iloc[0]
    assert t8["birth_month"] == "2024-10"
    assert t8["death_month"] == "2024-10"
    assert t8["lifetime"] == 1
    counts_t8 = json.loads(t8["n_nodes_by_month"])
    assert counts_t8["2024-10"] == 230

    # Type mapping
    tmap = pd.read_parquet(REPO_ROOT / "outputs" / "main" / "type_id_map.parquet")
    # Subtype 3 overwhelmingly maps to Macro 2
    sub3_to_macro2 = tmap[(tmap["registry_type"] == 3) & (tmap["macro_type"] == 2)]
    assert float(sub3_to_macro2["share_within_macro_type"].iloc[0]) > 0.95


