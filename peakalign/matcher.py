"""
matcher.py
----------
Core peak matching engine.

Matching pipeline:
  1. Compute pairwise candidate pairs within mz_tol + rt_tol windows
  2. Score each candidate pair (mz similarity + RT similarity + intensity correlation)
  3. Apply RT drift correction (optional polynomial on high-confidence anchors)
  4. Resolve ambiguous matches via bipartite (Hungarian) assignment
  5. Run statistical tests on accepted matches (permutation p-value, BH-FDR,
     match specificity, bootstrap CI on correlation)
  6. Classify features as matched / unique_A / unique_B
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.stats import pearsonr, spearmanr

from .schema import ColumnMap
from .statistics import StatTestParams, run_statistical_tests


# ---------------------------------------------------------------------------
# MatchResult
# ---------------------------------------------------------------------------

@dataclass
class MatchResult:
    """
    Container for the output of a peak matching run.

    Attributes
    ----------
    matched : pd.DataFrame
        Features present in both tables. Columns from both tables (suffixed
        with label_a / label_b) plus match metadata:

        Core match columns:
          _confidence      : composite match score [0-1]
          _delta_mz_ppm    : m/z deviation in ppm
          _delta_rt        : RT deviation in seconds (B - A)
          _intensity_corr  : Pearson/Spearman r across shared samples

        Statistical testing columns (if run_stats=True):
          _perm_pvalue     : permutation p-value for intensity correlation
          _perm_qvalue     : BH-FDR corrected q-value
          _fdr_significant : bool, q-value < fdr_alpha
          _specificity     : gap between best and second-best candidate score
          _corr_ci_low     : lower bound of 95% bootstrap CI on correlation
          _corr_ci_high    : upper bound of 95% bootstrap CI on correlation
          _ci_width        : _corr_ci_high - _corr_ci_low
          _stat_notes      : pipe-delimited flags for suspicious matches

    unique_a : pd.DataFrame
        Features only in table A.
    unique_b : pd.DataFrame
        Features only in table B.
    diagnostics : dict
        Summary statistics about the match run.
    """
    matched: pd.DataFrame
    unique_a: pd.DataFrame
    unique_b: pd.DataFrame
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        n_matched = len(self.matched)
        n_a = len(self.unique_a)
        n_b = len(self.unique_b)
        total_a = n_matched + n_a
        total_b = n_matched + n_b

        lines = [
            "=" * 55,
            "PeakAlign Match Summary",
            "=" * 55,
            f"  Table A features       : {total_a}",
            f"  Table B features       : {total_b}",
            f"  Matched pairs          : {n_matched}",
            f"  Unique to A            : {n_a}  ({100*n_a/max(total_a,1):.1f}%)",
            f"  Unique to B            : {n_b}  ({100*n_b/max(total_b,1):.1f}%)",
        ]

        if n_matched > 0:
            m = self.matched
            lines += [
                "",
                "  --- Match quality ---",
                f"  Median Δmz (ppm)       : {m['_delta_mz_ppm'].median():.3f}",
                f"  Median |Δrt| (s)       : {m['_delta_rt'].abs().median():.2f}",
                f"  Median intensity r     : {m['_intensity_corr'].median():.3f}",
                f"  Mean confidence        : {m['_confidence'].mean():.3f}",
            ]

            if "_perm_qvalue" in m.columns and m["_perm_qvalue"].notna().any():
                n_sig = int(m["_fdr_significant"].sum())
                med_spec = m["_specificity"].median()
                med_ci_w = m["_ci_width"].median()
                lines += [
                    "",
                    "  --- Statistical testing ---",
                    f"  FDR-significant matches: {n_sig} / {n_matched}"
                    f"  ({100*n_sig/max(n_matched,1):.1f}%)",
                    f"  Median specificity     : {med_spec:.3f}",
                    f"  Median CI width (r)    : {med_ci_w:.3f}",
                ]
                flagged = m[m["_stat_notes"].str.len() > 0]
                if len(flagged) > 0:
                    lines.append(
                        f"  Flagged matches        : {len(flagged)}"
                        f"  (see _stat_notes column)"
                    )

        lines.append("=" * 55)
        return "\n".join(lines)

    def significant_matches(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Return only FDR-significant matched pairs.

        Parameters
        ----------
        alpha : float
            q-value threshold. Default 0.05.
        """
        if "_perm_qvalue" not in self.matched.columns:
            raise ValueError(
                "Statistical testing was not run. "
                "Pass stat_params=StatTestParams() to align()."
            )
        return self.matched[self.matched["_perm_qvalue"] < alpha].copy()

    def suspicious_matches(self) -> pd.DataFrame:
        """Return matches flagged with stat_notes (wide CI, low specificity, etc.)."""
        if "_stat_notes" not in self.matched.columns:
            return pd.DataFrame()
        return self.matched[self.matched["_stat_notes"].str.len() > 0].copy()

    def to_excel(self, path: str) -> None:
        """Write matched, unique_a, unique_b to separate sheets."""
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            self.matched.to_excel(writer, sheet_name="matched", index=False)
            self.unique_a.to_excel(writer, sheet_name="unique_A", index=False)
            self.unique_b.to_excel(writer, sheet_name="unique_B", index=False)
    def triage(
        self,
        fdr_alpha: float = 0.05,
        ci_width_threshold: float = 0.4,
        specificity_threshold: float = 0.5,
    ) -> dict[str, pd.DataFrame]:
        """
        Stratify matched pairs into four evidence tiers for publication reporting.

        Triage is based on three independent statistical criteria evaluated
        in combination:

          1. FDR significance    : BH-corrected permutation q-value < fdr_alpha
          2. CI stability        : 95% bootstrap CI width on Pearson r
          3. Match specificity   : confidence gap to next-best candidate

        Tiers
        -----
        Tier 1 — High confidence (accepted)
            FDR significant, tight CI (width < threshold), specific match.
            These are the most statistically defensible matches. Suitable
            for direct use in downstream analysis without further review.

        Tier 2 — Accepted, flag for review
            FDR significant but either:
              (a) Wide CI with positive lower bound — correlation is real but
                  driven by a small number of samples; or
              (b) Bootstrap failed (ci_width = 2.0) with high specificity —
                  too few valid samples to estimate CI, but match is unambiguous.
            Recommend reporting these separately in supplementary material.

        Tier 3 — Borderline (manual review recommended)
            FDR significant but CI lower bound crosses zero — the correlation
            could plausibly be zero. Low specificity matches also fall here.
            These should be manually inspected before inclusion in analysis.
            Consider removing if downstream analysis is sensitive to false matches.

        Tier 4 — Not supported (rejected)
            Failed FDR threshold. Match may reflect m/z and RT proximity alone
            without biological support from intensity profiles. Recommend
            exclusion from downstream analysis.

        Parameters
        ----------
        fdr_alpha : float
            FDR q-value threshold. Default 0.05.
        ci_width_threshold : float
            Maximum CI width to consider a match stable. Default 0.4.
            A width of 0.4 means the 95% CI spans 0.4 units of Pearson r,
            e.g. [0.6, 1.0]. Wider than this suggests correlation instability.
        specificity_threshold : float
            Minimum specificity score to consider a match unambiguous.
            Default 0.5. Below this, a competing candidate scored nearly
            as well as the accepted match.

        Returns
        -------
        dict with keys: 'tier1', 'tier2', 'tier3', 'tier4', 'summary'
            Each tier is a pd.DataFrame subset of self.matched.
            'summary' is a pd.DataFrame with one row per tier describing
            counts, percentages, and median quality metrics — suitable for
            direct inclusion in a manuscript methods table.

        Raises
        ------
        ValueError
            If statistical testing columns are not present. Run align() with
            stat_params=StatTestParams() to generate them.

        Examples
        --------
        >>> triage = result.triage()
        >>> print(triage["summary"].to_string())
        >>> triage["tier1"].to_csv("high_confidence_matches.csv", index=False)
        >>> triage["tier2"].to_csv("flagged_matches.csv", index=False)
        """
        required = [
            "_perm_qvalue", "_fdr_significant", "_specificity",
            "_corr_ci_low", "_corr_ci_high", "_ci_width",
            "_intensity_corr", "_confidence",
        ]
        missing_cols = [c for c in required if c not in self.matched.columns]
        if missing_cols:
            raise ValueError(
                "Statistical testing columns not found. "
                "Run align() with stat_params=StatTestParams() first.\n"
                f"Missing: {missing_cols}"
            )

        m = self.matched
        n = len(m)

        if n == 0:
            empty = pd.DataFrame()
            return {k: empty for k in ["tier1", "tier2", "tier3", "tier4", "summary"]}

        sig      = m["_fdr_significant"] == True
        not_sig  = m["_fdr_significant"] == False

        # CI states
        boot_failed   = m["_ci_width"] == 2.0          # bootstrap produced [-1, 1]
        ci_tight      = m["_ci_width"] < ci_width_threshold
        ci_wide_pos   = (m["_ci_width"] >= ci_width_threshold) & (m["_ci_width"] < 2.0) & (m["_corr_ci_low"] >= 0)
        ci_crosses    = (m["_ci_width"] < 2.0) & (m["_corr_ci_low"] < 0)

        specific      = m["_specificity"] >= specificity_threshold
        not_specific  = m["_specificity"] < specificity_threshold

        # --- Tier 1: FDR sig + tight CI + specific ---
        t1_mask = sig & ci_tight & specific

        # --- Tier 2: FDR sig + (wide-but-positive CI) OR (bootstrap failed + specific) ---
        t2_mask = sig & ~t1_mask & (
            ci_wide_pos |
            (boot_failed & specific)
        )

        # --- Tier 3: FDR sig + (CI crosses zero) OR (low specificity) ---
        # Catch-all for anything significant but not cleanly in T1 or T2
        t3_mask = sig & ~t1_mask & ~t2_mask

        # --- Tier 4: not FDR significant ---
        t4_mask = not_sig

        tier1 = m[t1_mask].copy()
        tier2 = m[t2_mask].copy()
        tier3 = m[t3_mask].copy()
        tier4 = m[t4_mask].copy()

        # Verify exhaustive and mutually exclusive
        assigned = t1_mask.sum() + t2_mask.sum() + t3_mask.sum() + t4_mask.sum()
        assert assigned == n, (
            f"Triage is not exhaustive: {assigned} assigned but {n} total. "
            "Please report this as a bug."
        )

        def _tier_metrics(df: pd.DataFrame) -> dict:
            if len(df) == 0:
                return {
                    "n": 0, "pct": 0.0,
                    "median_r": np.nan, "median_qvalue": np.nan,
                    "median_specificity": np.nan, "median_ci_width": np.nan,
                    "median_delta_mz_ppm": np.nan, "median_delta_rt_s": np.nan,
                }
            return {
                "n":                   len(df),
                "pct":                 round(100 * len(df) / n, 1),
                "median_r":            round(float(df["_intensity_corr"].median()), 3),
                "median_qvalue":       round(float(df["_perm_qvalue"].median()), 4),
                "median_specificity":  round(float(df["_specificity"].median()), 3),
                "median_ci_width":     round(float(df["_ci_width"].median()), 3),
                "median_delta_mz_ppm": round(float(df["_delta_mz_ppm"].median()), 3),
                "median_delta_rt_s":   round(float(df["_delta_rt"].abs().median()), 2),
            }

        summary_rows = []
        tier_meta = [
            ("Tier 1", "High confidence — accepted",               tier1),
            ("Tier 2", "Accepted — flag for review",               tier2),
            ("Tier 3", "Borderline — manual review recommended",   tier3),
            ("Tier 4", "Not FDR-significant — rejected",           tier4),
        ]
        for tier_id, description, df in tier_meta:
            row = {"tier": tier_id, "description": description}
            row.update(_tier_metrics(df))
            summary_rows.append(row)

        summary = pd.DataFrame(summary_rows).set_index("tier")

        return {
            "tier1":   tier1,
            "tier2":   tier2,
            "tier3":   tier3,
            "tier4":   tier4,
            "summary": summary,
        }

    def triage_summary(
        self,
        fdr_alpha: float = 0.05,
        ci_width_threshold: float = 0.4,
        specificity_threshold: float = 0.5,
    ) -> str:
        """
        Print a publication-ready triage summary to stdout.

        Returns the same string for programmatic use.
        """
        t = self.triage(fdr_alpha, ci_width_threshold, specificity_threshold)
        s = t["summary"]
        n_total = len(self.matched)

        lines = [
            "=" * 65,
            "PeakAlign Triage Summary",
            "=" * 65,
            f"  Total matched pairs : {n_total}",
            f"  FDR alpha           : {fdr_alpha}",
            f"  CI width threshold  : {ci_width_threshold}",
            f"  Specificity threshold: {specificity_threshold}",
            "",
            f"  {'Tier':<8} {'n':>6} {'%':>6}  {'Med r':>7}  {'Med q':>8}  {'Med spec':>9}  {'Med CIw':>8}",
            "  " + "-" * 60,
        ]

        tier_labels = {
            "Tier 1": "High confidence (accepted)",
            "Tier 2": "Accepted, flag for review",
            "Tier 3": "Borderline, manual review",
            "Tier 4": "Not significant (rejected)",
        }
        for tier_id, label in tier_labels.items():
            row = s.loc[tier_id]
            lines.append(
                f"  {tier_id:<8} {int(row['n']):>6} {row['pct']:>5.1f}%"
                f"  {row['median_r']:>7.3f}"
                f"  {row['median_qvalue']:>8.4f}"
                f"  {row['median_specificity']:>9.3f}"
                f"  {row['median_ci_width']:>8.3f}"
            )

        lines += [
            "  " + "-" * 60,
            "",
            "  Tiers defined as:",
            "    Tier 1 : FDR sig + CI tight + specific match",
            "    Tier 2 : FDR sig + wide/failed CI or borderline specificity",
            "    Tier 3 : FDR sig + CI crosses zero or low specificity",
            "    Tier 4 : q >= fdr_alpha",
            "=" * 65,
        ]
        result_str = "\n".join(lines)
        print(result_str)
        return result_str

    def to_excel_tiered(self, path: str, **triage_kwargs) -> None:
        """
        Write a tiered Excel workbook with one sheet per tier plus unique features.

        Sheets:
          tier1_accepted       : High confidence matches
          tier2_flag_review    : Accepted, flagged for review
          tier3_borderline     : Borderline, manual review recommended
          tier4_rejected       : Not FDR-significant
          unique_A             : Features only in table A
          unique_B             : Features only in table B
          triage_summary       : One-row-per-tier summary table

        Parameters
        ----------
        path : str
            Output .xlsx file path.
        **triage_kwargs
            Passed to triage() — fdr_alpha, ci_width_threshold,
            specificity_threshold.
        """
        t = self.triage(**triage_kwargs)
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            t["tier1"].to_excel(writer,   sheet_name="tier1_accepted",    index=False)
            t["tier2"].to_excel(writer,   sheet_name="tier2_flag_review", index=False)
            t["tier3"].to_excel(writer,   sheet_name="tier3_borderline",  index=False)
            t["tier4"].to_excel(writer,   sheet_name="tier4_rejected",    index=False)
            self.unique_a.to_excel(writer, sheet_name="unique_A",          index=False)
            self.unique_b.to_excel(writer, sheet_name="unique_B",          index=False)
            t["summary"].to_excel(writer,  sheet_name="triage_summary")




# ---------------------------------------------------------------------------
# Tolerances
# ---------------------------------------------------------------------------

@dataclass
class Tolerances:
    """
    Matching tolerance parameters.

    Parameters
    ----------
    mz_ppm : float
        m/z tolerance in parts-per-million. Default 10.
    mz_da : float or None
        Absolute m/z tolerance in Daltons. When both mz_ppm and mz_da are
        set, the wider window is used.
    rt_seconds : float
        RT tolerance in seconds. Default 30.
    min_intensity_corr : float
        Minimum Pearson/Spearman r across shared samples to accept a match.
        Set to -1.0 to disable the intensity filter entirely.
    corr_method : str
        'pearson' or 'spearman'. Default 'pearson'.
    confidence_weights : dict
        Weights for the composite score.
        Keys: 'mz', 'rt', 'intensity'. Must sum to 1.
    """
    mz_ppm: float = 10.0
    mz_da: Optional[float] = None
    rt_seconds: float = 30.0
    min_intensity_corr: float = 0.7
    corr_method: str = "pearson"
    confidence_weights: dict = field(default_factory=lambda: {
        "mz": 0.35, "rt": 0.30, "intensity": 0.35
    })

    def __post_init__(self):
        w = self.confidence_weights
        if abs(sum(w.values()) - 1.0) > 1e-6:
            raise ValueError("confidence_weights must sum to 1.0")
        if self.corr_method not in ("pearson", "spearman"):
            raise ValueError("corr_method must be 'pearson' or 'spearman'")


# ---------------------------------------------------------------------------
# Internal scoring helpers
# ---------------------------------------------------------------------------

def _mz_tol_da(mz: float, tol: Tolerances) -> float:
    ppm_da = mz * tol.mz_ppm / 1e6
    if tol.mz_da is not None:
        return max(ppm_da, tol.mz_da)
    return ppm_da


def _delta_mz_ppm(mz_a: float, mz_b: float) -> float:
    return abs(mz_a - mz_b) / ((mz_a + mz_b) / 2) * 1e6


def _gaussian_sim(delta: float, sigma: float) -> float:
    return float(np.exp(-0.5 * (delta / sigma) ** 2))


def _cross_table_corr(
    row_a: pd.Series,
    row_b: pd.Series,
    cols_a: list[str],
    cols_b: list[str],
    method: str,
) -> float:
    vals_a = row_a[cols_a].values.astype(float)
    vals_b = row_b[cols_b].values.astype(float)
    mask = (vals_a > 0) & (vals_b > 0) & np.isfinite(vals_a) & np.isfinite(vals_b)
    if mask.sum() < 3:
        return float("nan")
    a, b = vals_a[mask], vals_b[mask]
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    if method == "spearman":
        r, _ = spearmanr(a, b)
    else:
        r, _ = pearsonr(a, b)
    return float(r)


def _compute_score(
    delta_mz_ppm: float,
    delta_rt: float,
    intensity_corr: float,
    tol: Tolerances,
) -> float:
    w = tol.confidence_weights
    mz_sim = _gaussian_sim(delta_mz_ppm, tol.mz_ppm * 0.5)
    rt_sim = _gaussian_sim(delta_rt, tol.rt_seconds * 0.5)
    if np.isnan(intensity_corr):
        w_mz = w["mz"] + w["intensity"] / 2
        w_rt = w["rt"] + w["intensity"] / 2
        score = w_mz * mz_sim + w_rt * rt_sim
    else:
        int_sim = (intensity_corr + 1) / 2
        score = w["mz"] * mz_sim + w["rt"] * rt_sim + w["intensity"] * int_sim
    return float(np.clip(score, 0.0, 1.0))


# ---------------------------------------------------------------------------
# RT drift correction
# ---------------------------------------------------------------------------

def _estimate_rt_drift(
    rt_a: np.ndarray,
    rt_b: np.ndarray,
    min_anchors: int = 20,
) -> Optional[object]:
    if len(rt_a) < min_anchors:
        warnings.warn(
            f"Only {len(rt_a)} anchor pairs for RT drift correction "
            f"(need {min_anchors}). Skipping.",
            stacklevel=3,
        )
        return None

    try:
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline

        deltas = rt_b - rt_a
        model = make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1.0))
        model.fit(rt_a.reshape(-1, 1), deltas)

        def drift_fn(rt):
            correction = model.predict(np.array(rt).reshape(-1, 1))
            return np.array(rt) + correction

        return drift_fn

    except ImportError:
        coeffs = np.polyfit(rt_a, rt_b - rt_a, deg=1)
        poly = np.poly1d(coeffs)

        def drift_fn(rt):
            return np.array(rt) + poly(np.array(rt))

        return drift_fn


# ---------------------------------------------------------------------------
# PeakMatcher
# ---------------------------------------------------------------------------

class PeakMatcher:
    """
    Matches peaks between two LCMS feature tables.

    Parameters
    ----------
    tol : Tolerances
        Matching tolerances and scoring weights.
    stat_params : StatTestParams or None
        Statistical testing configuration. Pass None to skip all
        statistical tests. Default: StatTestParams() (run with defaults).
    rt_drift_correction : bool
        If True, estimate and correct systematic RT drift before final
        matching. Default True.
    require_intensity_corr : bool
        If True, pairs below tol.min_intensity_corr are rejected.
        Default True.
    """

    def __init__(
        self,
        tol: Optional[Tolerances] = None,
        stat_params: Optional[StatTestParams] = None,
        rt_drift_correction: bool = True,
        require_intensity_corr: bool = True,
    ):
        self.tol = tol or Tolerances()
        self.stat_params = stat_params if stat_params is not None else StatTestParams()
        self.rt_drift_correction = rt_drift_correction
        self.require_intensity_corr = require_intensity_corr

    def match(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        map_a: ColumnMap,
        map_b: ColumnMap,
        label_a: str = "A",
        label_b: str = "B",
    ) -> MatchResult:
        map_a.validate_against(df_a)
        map_b.validate_against(df_b)

        df_a = df_a.reset_index(drop=True)
        df_b = df_b.reset_index(drop=True)

        shared_a, shared_b = self._shared_intensity_cols(map_a, map_b)

        # Step 1: candidate pairs
        candidates = self._candidate_pairs(df_a, df_b, map_a, map_b)
        if len(candidates) == 0:
            return self._empty_result(df_a, df_b, label_a, label_b)

        # Step 2: RT drift correction
        if self.rt_drift_correction and len(candidates) >= 10:
            df_a, candidates = self._apply_rt_drift_correction(
                df_a, df_b, map_a, map_b, candidates
            )

        # Step 3: score all candidates
        scored = self._score_candidates(
            df_a, df_b, map_a, map_b, candidates, shared_a, shared_b
        )

        # Step 4: intensity correlation filter
        if self.require_intensity_corr and len(shared_a) >= 3:
            corr_mask = (
                scored["intensity_corr"].isna() |
                (scored["intensity_corr"] >= self.tol.min_intensity_corr)
            )
            scored = scored[corr_mask].copy()

        if len(scored) == 0:
            return self._empty_result(df_a, df_b, label_a, label_b)

        # Step 5: bipartite matching
        matched_pairs = self._bipartite_match(scored)

        # Step 6: statistical testing
        matched_pairs = run_statistical_tests(
            matched_pairs=matched_pairs,
            df_a=df_a,
            df_b=df_b,
            shared_cols_a=shared_a,
            shared_cols_b=shared_b,
            all_candidates=scored,
            params=self.stat_params,
        )

        # Step 7: build output
        return self._build_result(
            df_a, df_b, matched_pairs, scored, label_a, label_b
        )

    def _candidate_pairs(self, df_a, df_b, map_a, map_b):
        mz_a = df_a[map_a.mz].values.astype(float)
        mz_b = df_b[map_b.mz].values.astype(float)
        rt_a = df_a[map_a.rt].values.astype(float)
        rt_b = df_b[map_b.rt].values.astype(float)

        candidates = []
        for i, (mz_i, rt_i) in enumerate(zip(mz_a, rt_a)):
            mz_tol = _mz_tol_da(mz_i, self.tol)
            mz_mask = np.abs(mz_b - mz_i) <= mz_tol
            rt_mask = np.abs(rt_b - rt_i) <= self.tol.rt_seconds
            for j in np.where(mz_mask & rt_mask)[0]:
                candidates.append((i, int(j)))
        return candidates

    def _apply_rt_drift_correction(self, df_a, df_b, map_a, map_b, candidates):
        rt_a_anchors = df_a[map_a.rt].values[[c[0] for c in candidates]]
        rt_b_anchors = df_b[map_b.rt].values[[c[1] for c in candidates]]

        drift_fn = _estimate_rt_drift(rt_a_anchors, rt_b_anchors)
        if drift_fn is None:
            return df_a, candidates

        df_a = df_a.copy()
        df_a[map_a.rt] = drift_fn(df_a[map_a.rt].values.astype(float))
        if map_a.rt_min:
            df_a[map_a.rt_min] = drift_fn(df_a[map_a.rt_min].values.astype(float))
        if map_a.rt_max:
            df_a[map_a.rt_max] = drift_fn(df_a[map_a.rt_max].values.astype(float))

        return df_a, self._candidate_pairs(df_a, df_b, map_a, map_b)

    def _score_candidates(self, df_a, df_b, map_a, map_b, candidates, shared_a, shared_b):
        rows = []
        for i, j in candidates:
            row_a = df_a.iloc[i]
            row_b = df_b.iloc[j]
            mz_i, mz_j = float(row_a[map_a.mz]), float(row_b[map_b.mz])
            rt_i, rt_j = float(row_a[map_a.rt]), float(row_b[map_b.rt])
            delta_mz_ppm = _delta_mz_ppm(mz_i, mz_j)
            delta_rt = rt_j - rt_i
            intensity_corr = (
                _cross_table_corr(row_a, row_b, shared_a, shared_b, self.tol.corr_method)
                if shared_a and shared_b else float("nan")
            )
            rows.append({
                "idx_a": i, "idx_b": j,
                "delta_mz_ppm": delta_mz_ppm, "delta_rt": delta_rt,
                "intensity_corr": intensity_corr,
                "confidence": _compute_score(delta_mz_ppm, delta_rt, intensity_corr, self.tol),
            })
        return pd.DataFrame(rows)

    def _bipartite_match(self, scored):
        idx_a_vals = scored["idx_a"].unique()
        idx_b_vals = scored["idx_b"].unique()
        cost_matrix = np.ones((len(idx_a_vals), len(idx_b_vals)))
        a_pos = {v: i for i, v in enumerate(idx_a_vals)}
        b_pos = {v: i for i, v in enumerate(idx_b_vals)}
        for _, row in scored.iterrows():
            cost_matrix[a_pos[row["idx_a"]], b_pos[row["idx_b"]]] = 1.0 - row["confidence"]
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        matched_rows = []
        for ri, ci in zip(row_ind, col_ind):
            ia, ib = idx_a_vals[ri], idx_b_vals[ci]
            subset = scored[(scored["idx_a"] == ia) & (scored["idx_b"] == ib)]
            if len(subset) > 0:
                matched_rows.append(subset.iloc[0].to_dict())
        return pd.DataFrame(matched_rows)

    def _build_result(self, df_a, df_b, matched_pairs, scored, label_a, label_b):
        if len(matched_pairs) == 0:
            return self._empty_result(df_a, df_b, label_a, label_b)

        matched_idx_a = set(matched_pairs["idx_a"].astype(int))
        matched_idx_b = set(matched_pairs["idx_b"].astype(int))
        a_renamed = df_a.add_suffix(f"_{label_a}")
        b_renamed = df_b.add_suffix(f"_{label_b}")
        stat_cols = [c for c in matched_pairs.columns if c.startswith("_")]

        rows = []
        for _, pair in matched_pairs.iterrows():
            ia, ib = int(pair["idx_a"]), int(pair["idx_b"])
            row = {}
            row.update(a_renamed.iloc[ia].to_dict())
            row.update(b_renamed.iloc[ib].to_dict())
            row["_confidence"]     = pair["confidence"]
            row["_delta_mz_ppm"]   = pair["delta_mz_ppm"]
            row["_delta_rt"]       = pair["delta_rt"]
            row["_intensity_corr"] = pair["intensity_corr"]
            for col in stat_cols:
                row[col] = pair[col]
            rows.append(row)

        matched_df = pd.DataFrame(rows)
        unique_a = df_a.iloc[[i for i in range(len(df_a)) if i not in matched_idx_a]].copy()
        unique_b = df_b.iloc[[i for i in range(len(df_b)) if i not in matched_idx_b]].copy()
        n_sig = int(matched_df["_fdr_significant"].sum()) \
            if "_fdr_significant" in matched_df.columns else None

        return MatchResult(
            matched=matched_df, unique_a=unique_a, unique_b=unique_b,
            diagnostics={
                "n_features_a": len(df_a), "n_features_b": len(df_b),
                "n_matched": len(matched_df), "n_unique_a": len(unique_a),
                "n_unique_b": len(unique_b),
                "n_candidates_evaluated": len(scored),
                "n_fdr_significant": n_sig,
            },
        )

    def _empty_result(self, df_a, df_b, label_a, label_b):
        return MatchResult(
            matched=pd.DataFrame(), unique_a=df_a.copy(), unique_b=df_b.copy(),
            diagnostics={
                "n_features_a": len(df_a), "n_features_b": len(df_b),
                "n_matched": 0, "n_unique_a": len(df_a), "n_unique_b": len(df_b),
                "n_candidates_evaluated": 0, "n_fdr_significant": 0,
            },
        )

    def _shared_intensity_cols(self, map_a, map_b):
        shared = [c for c in map_a.intensity_cols if c in map_b.intensity_cols]
        return shared, shared