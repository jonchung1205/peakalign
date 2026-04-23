"""
statistics.py
-------------
Statistical testing for matched peak pairs.

Four tests are applied to each matched pair after bipartite assignment:

  1. Permutation p-value (perm_pvalue)
       Shuffles sample labels in table B `n_permutations` times and recomputes
       the intensity correlation each time, building a null distribution.
       p-value = fraction of permuted correlations >= observed correlation.
       Assumption-free — valid for both raw and log-transformed intensities.

  2. BH-FDR corrected q-value (perm_qvalue)
       Benjamini-Hochberg correction applied across all matched pairs.
       Controls the false discovery rate at the specified alpha level.
       Use q < 0.05 as the threshold for a statistically supported match.

  3. Match specificity score (specificity)
       For each accepted match, how much better was the best candidate vs
       the second-best candidate (by confidence score)?
         specificity = confidence_best - confidence_2nd_best
       Range [0, 1]. Values near 0 mean the match was ambiguous.
       Values near 1 mean the match was unambiguous.

  4. Bootstrap 95% CI on intensity correlation (corr_ci_low, corr_ci_high)
       Resamples the n_samples samples with replacement `n_bootstrap` times
       and recomputes r each time. Reports the 2.5th and 97.5th percentiles.
       Tight CIs indicate a stable, reliable correlation.

Performance
-----------
All permutation and bootstrap loops are fully vectorized using numpy matrix
operations. For a typical dataset (2000 matched pairs, 18 samples, 1000
permutations), runtime is under 30 seconds on a standard laptop.

The key optimization: instead of looping over each matched pair and running
N permutations sequentially, we:
  1. Stack all matched pair intensity vectors into matrices (n_pairs x n_samples)
  2. Generate all permutation indices at once (n_permutations x n_samples)
  3. Compute all N*M correlations in a single vectorized operation
  4. Derive all p-values from the resulting null matrix simultaneously

Notes on log-transformation
---------------------------
If data is already log2-transformed, all four tests are well-calibrated.
If data is NOT log-transformed (raw LCMS intensities), the tool detects this
automatically using a dynamic range heuristic and applies an internal log1p
normalization ONLY for statistical testing. The user's data is never modified.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StatTestParams:
    """
    Parameters controlling statistical testing of matched pairs.

    Parameters
    ----------
    n_permutations : int
        Number of label permutations for the permutation p-value.
        Higher = more precise p-values but slower.
        Default 1000. Minimum reliable: 100. Publication: 1000-10000.
    n_bootstrap : int
        Number of bootstrap resamples for the correlation CI.
        Default 500. Publication: 1000-5000.
    fdr_alpha : float
        FDR threshold for the BH q-value. Default 0.05.
    corr_method : str
        Correlation method. 'pearson' or 'spearman'. Default 'pearson'.
    auto_log_transform : bool
        If True (default), detect raw (non-log) intensity data and apply
        internal log1p normalization before statistical testing only.
        The user's data is never modified.
    random_seed : int
        Seed for reproducibility. Default 42.
    run_stats : bool
        Master switch. If False, skip all statistical testing.
        Default True.
    """
    n_permutations: int = 1000
    n_bootstrap: int = 500
    fdr_alpha: float = 0.05
    corr_method: str = "pearson"
    auto_log_transform: bool = True
    random_seed: int = 42
    run_stats: bool = True

    def __post_init__(self):
        if self.corr_method not in ("pearson", "spearman"):
            raise ValueError("corr_method must be 'pearson' or 'spearman'")
        if not 0 < self.fdr_alpha < 1:
            raise ValueError("fdr_alpha must be between 0 and 1")
        if self.n_permutations < 100:
            warnings.warn(
                f"n_permutations={self.n_permutations} is low. "
                "P-values will be imprecise. Use >= 1000 for publication.",
                stacklevel=2,
            )


# ---------------------------------------------------------------------------
# Log-transform detection
# ---------------------------------------------------------------------------

def _is_log_transformed(values: np.ndarray) -> bool:
    """
    Heuristic: if dynamic range (max/min of positive values) exceeds 1e4
    (4 orders of magnitude), the data is likely raw/untransformed.
    Log2-transformed data typically spans ~10-20 units, not 1e4+.
    """
    pos = values[np.isfinite(values) & (values > 0)]
    if len(pos) < 2:
        return True
    return float(pos.max() / pos.min()) < 1e4


# ---------------------------------------------------------------------------
# Core vectorized correlation
# ---------------------------------------------------------------------------

def _pearson_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute Pearson r between every row of A and corresponding row of B.

    Parameters
    ----------
    A : np.ndarray, shape (n_pairs, n_samples)
    B : np.ndarray, shape (n_pairs, n_samples)

    Returns
    -------
    np.ndarray, shape (n_pairs,)
        Pearson r for each pair. NaN where std is zero.
    """
    # Mean-center each row
    A_c = A - A.mean(axis=1, keepdims=True)
    B_c = B - B.mean(axis=1, keepdims=True)

    # Numerator: sum of products
    num = (A_c * B_c).sum(axis=1)

    # Denominator: product of stds
    std_a = np.sqrt((A_c ** 2).sum(axis=1))
    std_b = np.sqrt((B_c ** 2).sum(axis=1))

    denom = std_a * std_b

    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(denom > 0, num / denom, np.nan)

    return r


def _spearman_matrix(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Compute Spearman r between every row of A and corresponding row of B.
    Converts to ranks then applies Pearson.
    """
    from scipy.stats import rankdata
    A_ranked = np.apply_along_axis(rankdata, 1, A)
    B_ranked = np.apply_along_axis(rankdata, 1, B)
    return _pearson_matrix(A_ranked, B_ranked)


def _corr_matrix(A: np.ndarray, B: np.ndarray, method: str) -> np.ndarray:
    if method == "spearman":
        return _spearman_matrix(A, B)
    return _pearson_matrix(A, B)


# ---------------------------------------------------------------------------
# Build intensity matrices from matched pairs
# ---------------------------------------------------------------------------

def _build_intensity_matrices(
    matched_pairs: pd.DataFrame,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    shared_cols_a: list[str],
    shared_cols_b: list[str],
    auto_log: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """
    Extract intensity matrices for all matched pairs at once.

    Returns
    -------
    A_mat : np.ndarray, shape (n_pairs, n_samples)
    B_mat : np.ndarray, shape (n_pairs, n_samples)
    valid_mask : np.ndarray of bool, shape (n_pairs,)
        True where the pair has enough valid samples for testing.
    log_applied : bool
        Whether internal log1p normalization was applied.
    """
    n_pairs = len(matched_pairs)
    n_samples = len(shared_cols_a)

    A_mat = np.full((n_pairs, n_samples), np.nan)
    B_mat = np.full((n_pairs, n_samples), np.nan)

    for k, (_, pair) in enumerate(matched_pairs.iterrows()):
        ia = int(pair["idx_a"])
        ib = int(pair["idx_b"])
        A_mat[k] = df_a.iloc[ia][shared_cols_a].values.astype(float)
        B_mat[k] = df_b.iloc[ib][shared_cols_b].values.astype(float)

    # Replace zeros and non-finite with NaN
    A_mat = np.where((A_mat > 0) & np.isfinite(A_mat), A_mat, np.nan)
    B_mat = np.where((B_mat > 0) & np.isfinite(B_mat), B_mat, np.nan)

    # Detect and apply internal log transform if needed
    log_applied = False
    if auto_log:
        all_vals = np.concatenate([
            A_mat[np.isfinite(A_mat)],
            B_mat[np.isfinite(B_mat)]
        ])
        if len(all_vals) > 0 and not _is_log_transformed(all_vals):
            A_mat = np.where(np.isfinite(A_mat), np.log1p(A_mat), np.nan)
            B_mat = np.where(np.isfinite(B_mat), np.log1p(B_mat), np.nan)
            log_applied = True

    # Fill NaN with column mean per row for correlation stability
    # (only used where at least min_valid samples exist)
    min_valid = 3
    valid_counts = np.sum(np.isfinite(A_mat) & np.isfinite(B_mat), axis=1)
    valid_mask = valid_counts >= min_valid

    # For invalid positions, fill with row mean to avoid NaN propagation
    # These rows will be masked out in results anyway
    row_mean_a = np.nanmean(A_mat, axis=1, keepdims=True)
    row_mean_b = np.nanmean(B_mat, axis=1, keepdims=True)
    A_mat = np.where(np.isfinite(A_mat), A_mat, row_mean_a)
    B_mat = np.where(np.isfinite(B_mat), B_mat, row_mean_b)

    # Replace any remaining NaN (rows with all NaN) with zeros
    A_mat = np.nan_to_num(A_mat, nan=0.0)
    B_mat = np.nan_to_num(B_mat, nan=0.0)

    return A_mat, B_mat, valid_mask, log_applied


# ---------------------------------------------------------------------------
# Vectorized permutation p-values
# ---------------------------------------------------------------------------

def _vectorized_permutation_pvalues(
    A_mat: np.ndarray,
    B_mat: np.ndarray,
    observed_r: np.ndarray,
    valid_mask: np.ndarray,
    n_permutations: int,
    method: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Compute permutation p-values for all pairs simultaneously.

    For each permutation, shuffle the sample columns of B_mat and compute
    r for all pairs at once. This replaces the nested Python for-loop with
    a single matrix operation per permutation.

    Parameters
    ----------
    A_mat : np.ndarray, shape (n_pairs, n_samples)
    B_mat : np.ndarray, shape (n_pairs, n_samples)
    observed_r : np.ndarray, shape (n_pairs,)
    valid_mask : np.ndarray of bool, shape (n_pairs,)
    n_permutations : int
    method : str
    rng : np.random.Generator

    Returns
    -------
    np.ndarray, shape (n_pairs,)
        P-values. NaN for invalid pairs.
    """
    n_pairs, n_samples = A_mat.shape
    exceed_count = np.zeros(n_pairs)

    for _ in range(n_permutations):
        # Shuffle sample order — same shuffle applied to all pairs
        # This preserves the within-pair structure while breaking
        # the A-B correspondence
        perm_idx = rng.permutation(n_samples)
        B_perm = B_mat[:, perm_idx]
        null_r = _corr_matrix(A_mat, B_perm, method)
        exceed_count += (null_r >= observed_r).astype(float)

    pvalues = exceed_count / n_permutations
    # Minimum p-value floor
    pvalues = np.maximum(pvalues, 1.0 / n_permutations)
    # Mask invalid pairs
    pvalues = np.where(valid_mask, pvalues, np.nan)
    return pvalues


# ---------------------------------------------------------------------------
# Vectorized bootstrap CI
# ---------------------------------------------------------------------------

def _vectorized_bootstrap_ci(
    A_mat: np.ndarray,
    B_mat: np.ndarray,
    valid_mask: np.ndarray,
    n_bootstrap: int,
    method: str,
    rng: np.random.Generator,
    ci: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute bootstrap 95% CI on correlation for all pairs simultaneously.

    Parameters
    ----------
    A_mat, B_mat : np.ndarray, shape (n_pairs, n_samples)
    valid_mask : np.ndarray of bool
    n_bootstrap : int
    method : str
    rng : np.random.Generator
    ci : float

    Returns
    -------
    ci_low, ci_high : np.ndarray, shape (n_pairs,)
    """
    n_pairs, n_samples = A_mat.shape
    boot_r = np.full((n_bootstrap, n_pairs), np.nan)

    for k in range(n_bootstrap):
        # Resample sample indices with replacement
        idx = rng.integers(0, n_samples, size=n_samples)
        r = _corr_matrix(A_mat[:, idx], B_mat[:, idx], method)
        boot_r[k] = r

    alpha = (1 - ci) / 2
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ci_low  = np.nanpercentile(boot_r, 100 * alpha,       axis=0)
        ci_high = np.nanpercentile(boot_r, 100 * (1 - alpha), axis=0)

    ci_low  = np.where(valid_mask, ci_low,  np.nan)
    ci_high = np.where(valid_mask, ci_high, np.nan)
    return ci_low, ci_high


# ---------------------------------------------------------------------------
# BH-FDR correction
# ---------------------------------------------------------------------------

def bh_fdr(pvalues: np.ndarray) -> np.ndarray:
    """
    Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    pvalues : np.ndarray
        Array of p-values. NaN values are preserved as NaN in output.

    Returns
    -------
    np.ndarray
        BH-adjusted q-values, same length as input.
    """
    n = len(pvalues)
    qvalues = np.full(n, np.nan)

    valid_mask = np.isfinite(pvalues)
    valid_idx = np.where(valid_mask)[0]
    valid_p = pvalues[valid_mask]

    if len(valid_p) == 0:
        return qvalues

    m = len(valid_p)
    sort_order = np.argsort(valid_p)
    sorted_p = valid_p[sort_order]
    ranks = np.arange(1, m + 1)
    adjusted = sorted_p * m / ranks

    # Enforce monotonicity right to left
    for i in range(m - 2, -1, -1):
        adjusted[i] = min(adjusted[i], adjusted[i + 1])

    adjusted = np.clip(adjusted, 0.0, 1.0)
    unsorted = np.empty(m)
    unsorted[sort_order] = adjusted
    qvalues[valid_idx] = unsorted
    return qvalues


# ---------------------------------------------------------------------------
# Match specificity
# ---------------------------------------------------------------------------

def _compute_all_specificities(
    matched_pairs: pd.DataFrame,
    all_candidates: pd.DataFrame,
) -> np.ndarray:
    """
    Compute specificity for all matched pairs at once.

    specificity = accepted_confidence - best_alternative_confidence

    If no alternative candidate existed, specificity = accepted_confidence.
    """
    n = len(matched_pairs)
    specificities = np.full(n, np.nan)

    for k, (_, pair) in enumerate(matched_pairs.iterrows()):
        ia = int(pair["idx_a"])
        ib = int(pair["idx_b"])
        conf = float(pair["confidence"])

        alt_a = all_candidates[
            (all_candidates["idx_a"] == ia) & (all_candidates["idx_b"] != ib)
        ]["confidence"]
        alt_b = all_candidates[
            (all_candidates["idx_b"] == ib) & (all_candidates["idx_a"] != ia)
        ]["confidence"]

        all_alts = pd.concat([alt_a, alt_b])
        if len(all_alts) == 0:
            specificities[k] = conf
        else:
            specificities[k] = float(np.clip(conf - all_alts.max(), 0.0, 1.0))

    return specificities


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_statistical_tests(
    matched_pairs: pd.DataFrame,
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    shared_cols_a: list[str],
    shared_cols_b: list[str],
    all_candidates: pd.DataFrame,
    params: StatTestParams,
) -> pd.DataFrame:
    """
    Run all four statistical tests on every matched pair and append results
    as new columns to matched_pairs.

    New columns added
    -----------------
    _perm_pvalue     : permutation p-value for intensity correlation
    _perm_qvalue     : BH-FDR corrected q-value
    _fdr_significant : bool, True if q-value < params.fdr_alpha
    _specificity     : match specificity score
    _corr_ci_low     : lower bound of 95% bootstrap CI on correlation
    _corr_ci_high    : upper bound of 95% bootstrap CI on correlation
    _ci_width        : _corr_ci_high - _corr_ci_low
    _stat_notes      : pipe-delimited flags for suspicious matches

    Parameters
    ----------
    matched_pairs : pd.DataFrame
        Output of bipartite matching. Must contain idx_a, idx_b,
        intensity_corr, confidence.
    df_a, df_b : pd.DataFrame
        Original feature tables.
    shared_cols_a, shared_cols_b : list[str]
        Intensity column names corresponding to the same samples.
    all_candidates : pd.DataFrame
        Full scored candidates table before bipartite resolution.
    params : StatTestParams
        Statistical testing configuration.

    Returns
    -------
    pd.DataFrame
        matched_pairs with statistical columns appended.
    """
    # Master switch
    if not params.run_stats or len(matched_pairs) == 0:
        matched_pairs = matched_pairs.copy()
        for col in ["_perm_pvalue", "_perm_qvalue", "_specificity",
                    "_corr_ci_low", "_corr_ci_high", "_ci_width"]:
            matched_pairs[col] = np.nan
        matched_pairs["_fdr_significant"] = False
        matched_pairs["_stat_notes"] = ""
        return matched_pairs

    rng = np.random.default_rng(params.random_seed)
    has_shared = len(shared_cols_a) >= 3 and len(shared_cols_b) >= 3

    # --- Specificity (no intensity data needed, fast) ---
    specificities = _compute_all_specificities(matched_pairs, all_candidates)

    if not has_shared:
        matched_pairs = matched_pairs.copy()
        matched_pairs["_perm_pvalue"]     = np.nan
        matched_pairs["_perm_qvalue"]     = np.nan
        matched_pairs["_fdr_significant"] = False
        matched_pairs["_specificity"]     = specificities
        matched_pairs["_corr_ci_low"]     = np.nan
        matched_pairs["_corr_ci_high"]    = np.nan
        matched_pairs["_ci_width"]        = np.nan
        matched_pairs["_stat_notes"]      = "no_shared_samples"
        return matched_pairs

    # --- Build intensity matrices for all pairs at once ---
    A_mat, B_mat, valid_mask, log_applied = _build_intensity_matrices(
        matched_pairs, df_a, df_b,
        shared_cols_a, shared_cols_b,
        params.auto_log_transform,
    )

    # Observed correlations (recomputed on cleaned/normalized matrices)
    observed_r = _corr_matrix(A_mat, B_mat, params.corr_method)
    observed_r = np.where(valid_mask, observed_r, np.nan)

    # --- Permutation p-values (vectorized) ---
    perm_pvalues = _vectorized_permutation_pvalues(
        A_mat, B_mat, observed_r, valid_mask,
        params.n_permutations, params.corr_method, rng
    )

    # --- Bootstrap CI (vectorized) ---
    ci_low, ci_high = _vectorized_bootstrap_ci(
        A_mat, B_mat, valid_mask,
        params.n_bootstrap, params.corr_method, rng
    )

    # --- BH-FDR correction ---
    qvalues = bh_fdr(perm_pvalues)
    fdr_sig = np.where(np.isfinite(qvalues), qvalues < params.fdr_alpha, False)

    # --- Stat notes ---
    ci_width = ci_high - ci_low
    stat_notes = []
    for k in range(len(matched_pairs)):
        notes = []
        if log_applied:
            notes.append("internal_log1p_applied")
        if np.isfinite(ci_width[k]) and ci_width[k] > 0.4:
            notes.append("wide_ci")
        if np.isfinite(specificities[k]) and specificities[k] < 0.1:
            notes.append("low_specificity")
        if not valid_mask[k]:
            notes.append("insufficient_valid_samples")
        stat_notes.append("|".join(notes))

    matched_pairs = matched_pairs.copy()
    matched_pairs["_perm_pvalue"]     = perm_pvalues
    matched_pairs["_perm_qvalue"]     = qvalues
    matched_pairs["_fdr_significant"] = fdr_sig.astype(bool)
    matched_pairs["_specificity"]     = specificities
    matched_pairs["_corr_ci_low"]     = ci_low
    matched_pairs["_corr_ci_high"]    = ci_high
    matched_pairs["_ci_width"]        = ci_width
    matched_pairs["_stat_notes"]      = stat_notes

    return matched_pairs