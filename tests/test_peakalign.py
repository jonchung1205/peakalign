"""
tests/test_peakalign.py
-----------------------
Unit and integration tests for peakalign.
Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from peakalign import align, ColumnMap, Tolerances, infer_column_map
from peakalign.matcher import (
    _delta_mz_ppm, _gaussian_sim, _compute_score, PeakMatcher
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def make_feature_table(
    n=50,
    mz_range=(100, 1000),
    rt_range=(60, 1800),
    n_samples=6,
    seed=42,
) -> pd.DataFrame:
    """Generate a synthetic feature table similar to XCMS output."""
    rng = np.random.default_rng(seed)
    mz = rng.uniform(*mz_range, n)
    rt = rng.uniform(*rt_range, n)
    mz_width = rng.uniform(0.001, 0.005, n)
    rt_width = rng.uniform(2, 20, n)

    sample_names = [f"Sample_{i+1}" for i in range(n_samples)]
    intensities = rng.lognormal(mean=10, sigma=1.5, size=(n, n_samples))

    df = pd.DataFrame({
        "mz": mz,
        "mzmin": mz - mz_width,
        "mzmax": mz + mz_width,
        "rt": rt,
        "rtmin": rt - rt_width,
        "rtmax": rt + rt_width,
        "isotopes": [None] * n,
        "adduct": [None] * n,
        "pcgroup": rng.integers(1, 20, n),
    })
    for i, s in enumerate(sample_names):
        df[s] = intensities[:, i]

    return df


def make_perturbed_table(df_a: pd.DataFrame, mz_noise=0.00001, rt_noise=2.0, seed=99):
    """
    Create a second table from df_a with small mz/rt perturbations and
    intensity scaling (simulates two different acquisition settings).
    """
    rng = np.random.default_rng(seed)
    df_b = df_a.copy()
    n = len(df_b)
    df_b["mz"] = df_b["mz"] + rng.normal(0, mz_noise, n)
    df_b["rt"] = df_b["rt"] + rng.normal(0, rt_noise, n)
    # Scale intensities slightly
    sample_cols = [c for c in df_b.columns if c.startswith("Sample_")]
    for col in sample_cols:
        df_b[col] = df_b[col] * rng.uniform(0.9, 1.1, n)
    return df_b


# ---------------------------------------------------------------------------
# Schema inference tests
# ---------------------------------------------------------------------------

class TestSchemaInference:

    def test_infer_standard_xcms_columns(self):
        df = make_feature_table(n=10)
        cmap = infer_column_map(df)
        assert cmap.mz == "mz"
        assert cmap.rt == "rt"
        assert cmap.mz_min == "mzmin"
        assert cmap.mz_max == "mzmax"
        assert cmap.rt_min == "rtmin"
        assert cmap.rt_max == "rtmax"
        assert len(cmap.intensity_cols) == 6
        assert all(c.startswith("Sample_") for c in cmap.intensity_cols)

    def test_infer_skyline_style_columns(self):
        df = pd.DataFrame({
            "Precursor Mz": [100.0, 200.0],
            "Best Retention Time": [300.0, 400.0],
            "Area_S1": [1000.0, 2000.0],
            "Area_S2": [1100.0, 2100.0],
        })
        cmap = infer_column_map(df, mz_col="Precursor Mz", rt_col="Best Retention Time")
        assert cmap.mz == "Precursor Mz"
        assert cmap.rt == "Best Retention Time"
        assert set(cmap.intensity_cols) == {"Area_S1", "Area_S2"}

    def test_explicit_intensity_cols(self):
        df = make_feature_table(n=5)
        cmap = infer_column_map(df, intensity_cols=["Sample_1", "Sample_2"])
        assert cmap.intensity_cols == ["Sample_1", "Sample_2"]

    def test_missing_mz_raises(self):
        df = pd.DataFrame({"retention_time_xyz": [1.0], "Sample_1": [100.0]})
        with pytest.raises(ValueError, match="m/z column"):
            infer_column_map(df)

    def test_validate_against_raises_on_missing_col(self):
        df = make_feature_table(n=5)
        cmap = infer_column_map(df)
        df2 = df.drop(columns=["mz"])
        with pytest.raises(ValueError, match="not found in DataFrame"):
            cmap.validate_against(df2)

    def test_summary_string(self):
        df = make_feature_table(n=5)
        cmap = infer_column_map(df)
        s = cmap.summary()
        assert "mz" in s
        assert "rt" in s


# ---------------------------------------------------------------------------
# Scoring helper tests
# ---------------------------------------------------------------------------

class TestScoringHelpers:

    def test_delta_mz_ppm_exact(self):
        assert _delta_mz_ppm(500.0, 500.0) == 0.0

    def test_delta_mz_ppm_10ppm(self):
        mz = 500.0
        offset = mz * 10 / 1e6
        result = _delta_mz_ppm(mz, mz + offset)
        assert abs(result - 10.0) < 0.01

    def test_gaussian_sim_at_zero(self):
        assert _gaussian_sim(0.0, 5.0) == pytest.approx(1.0)

    def test_gaussian_sim_at_sigma(self):
        val = _gaussian_sim(5.0, 5.0)
        assert 0.5 < val < 0.7  # approx 0.606

    def test_compute_score_perfect_match(self):
        score = _compute_score(
            delta_mz_ppm=0.0,
            delta_rt=0.0,
            intensity_corr=1.0,
            tol=Tolerances()
        )
        assert score == pytest.approx(1.0, abs=0.01)

    def test_compute_score_bad_match(self):
        score = _compute_score(
            delta_mz_ppm=50.0,
            delta_rt=120.0,
            intensity_corr=-0.5,
            tol=Tolerances()
        )
        assert score < 0.3

    def test_compute_score_nan_intensity(self):
        # Should not crash, should redistribute weight
        score = _compute_score(0.5, 1.0, float("nan"), Tolerances())
        assert 0 <= score <= 1


# ---------------------------------------------------------------------------
# End-to-end matching tests
# ---------------------------------------------------------------------------

class TestEndToEnd:

    def test_identical_tables_all_matched(self):
        df = make_feature_table(n=30)
        result = align(df, df.copy(), require_intensity_corr=False)
        assert len(result.matched) == 30
        assert len(result.unique_a) == 0
        assert len(result.unique_b) == 0

    def test_perturbed_table_high_match_rate(self):
        df_a = make_feature_table(n=50)
        df_b = make_perturbed_table(df_a)
        result = align(df_a, df_b, tol=Tolerances(mz_ppm=15, rt_seconds=20))
        match_rate = len(result.matched) / 50
        assert match_rate >= 0.80, f"Match rate too low: {match_rate:.2%}"

    def test_no_overlap_tables_all_unique(self):
        df_a = make_feature_table(n=10, mz_range=(100, 200), seed=1)
        df_b = make_feature_table(n=10, mz_range=(800, 900), seed=2)
        result = align(df_a, df_b)
        assert len(result.matched) == 0
        assert len(result.unique_a) == 10
        assert len(result.unique_b) == 10

    def test_your_example_rows_match(self):
        """Reproduce the exact rows from the user's question."""
        sample_cols = [
            "BS4_1","BS4_2","BS4_3","BS80_1","BS80_2","BS80_3",
            "LL_1","LL_2","LL_3","LP_1","LP_2","LP_3",
            "Mix_1","Mix_2","Mix_3","VN_1","VN_2","VN_3"
        ]
        row1 = {
            "mz": 41.00358783, "mzmin": 41.00344413, "mzmax": 41.00375439,
            "rt": 850.4147182, "rtmin": 846.6281438, "rtmax": 852.7976502,
            "BS4_1":358.6130627,"BS4_2":355.515236,"BS4_3":364.7028539,
            "BS80_1":344.9169014,"BS80_2":862.0590945,"BS80_3":663.2903313,
            "LL_1":40745.1005,"LL_2":37387.70425,"LL_3":38223.40551,
            "LP_1":48412.62064,"LP_2":48423.04832,"LP_3":46852.75853,
            "Mix_1":21904.72907,"Mix_2":22990.75469,"Mix_3":23569.82193,
            "VN_1":285.0306367,"VN_2":345.0016282,"VN_3":151.0521234,
            "isotopes":None,"adduct":None,"pcgroup":14,
        }
        row2 = {
            "mz": 41.00358783, "mzmin": 41.00352024, "mzmax": 41.00375439,
            "rt": 847.8280169, "rtmin": 845.9032736, "rtmax": 849.5713783,
            "BS4_1":360.1580035,"BS4_2":433.4721809,"BS4_3":298.5109898,
            "BS80_1":402.9214243,"BS80_2":1057.502595,"BS80_3":712.9981646,
            "LL_1":41307.01153,"LL_2":38067.74493,"LL_3":39557.49842,
            "LP_1":48867.96085,"LP_2":50064.81752,"LP_3":48421.49366,
            "Mix_1":21904.72907,"Mix_2":23986.96268,"Mix_3":24127.93647,
            "VN_1":344.6254794,"VN_2":354.3049011,"VN_3":178.9391008,
            "isotopes":None,"adduct":None,"pcgroup":6,
        }
        df_a = pd.DataFrame([row1])
        df_b = pd.DataFrame([row2])
        result = align(df_a, df_b, tol=Tolerances(mz_ppm=10, rt_seconds=30))
        assert len(result.matched) == 1, "Expected the two rows to match"
        assert result.matched["_confidence"].iloc[0] > 0.8

    def test_result_summary_runs(self):
        df_a = make_feature_table(n=20)
        df_b = make_perturbed_table(df_a)
        result = align(df_a, df_b)
        s = result.summary()
        assert "Matched" in s

    def test_tolerances_custom(self):
        df_a = make_feature_table(n=20)
        df_b = make_perturbed_table(df_a, rt_noise=50)  # large RT noise
        # Tight RT tol — should match fewer
        r_tight = align(df_a, df_b, tol=Tolerances(rt_seconds=5))
        # Loose RT tol — should match more
        r_loose = align(df_a, df_b, tol=Tolerances(rt_seconds=120))
        assert len(r_loose.matched) >= len(r_tight.matched)

    def test_to_excel(self, tmp_path):
        df_a = make_feature_table(n=10)
        df_b = make_perturbed_table(df_a)
        result = align(df_a, df_b)
        out = str(tmp_path / "out.xlsx")
        result.to_excel(out)
        sheets = pd.read_excel(out, sheet_name=None)
        assert set(sheets.keys()) == {"matched", "unique_A", "unique_B"}

    def test_require_intensity_corr_false(self):
        """With require_intensity_corr=False, should still return matches."""
        df_a = make_feature_table(n=20, n_samples=2)  # too few for corr
        df_b = make_perturbed_table(df_a)
        result = align(df_a, df_b, require_intensity_corr=False)
        assert len(result.matched) > 0

    def test_diagnostics_keys(self):
        df_a = make_feature_table(n=15)
        df_b = make_perturbed_table(df_a)
        result = align(df_a, df_b)
        for key in ["n_features_a", "n_features_b", "n_matched", "n_unique_a", "n_unique_b"]:
            assert key in result.diagnostics
