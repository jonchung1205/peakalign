"""
align.py
--------
High-level public API for peakalign.

Typical usage
-------------
>>> import pandas as pd
>>> from peakalign import align, ColumnMap, Tolerances
>>>
>>> df_a = pd.read_csv("hilneg_setting1.csv")
>>> df_b = pd.read_csv("hilneg_setting2.csv")
>>>
>>> # Fully automatic — let peakalign infer column roles
>>> result = align(df_a, df_b)
>>> print(result.summary())
>>>
>>> # With explicit column mapping (e.g. Skyline export)
>>> result = align(
...     df_a, df_b,
...     mz_col="Precursor Mz",
...     rt_col="Best Retention Time",
...     intensity_cols_a=[c for c in df_a.columns if c.startswith("Area_")],
...     intensity_cols_b=[c for c in df_b.columns if c.startswith("Area_")],
...     tol=Tolerances(mz_ppm=5, rt_seconds=15),
... )
>>>
>>> # Save outputs
>>> result.matched.to_csv("matched.csv", index=False)
>>> result.unique_a.to_csv("unique_to_A.csv", index=False)
>>> result.unique_b.to_csv("unique_to_B.csv", index=False)
>>> result.to_excel("peakalign_results.xlsx")
"""

from __future__ import annotations

from typing import Optional

import pandas as pd

from .schema import ColumnMap, infer_column_map
from .matcher import MatchResult, PeakMatcher, Tolerances
from .statistics import StatTestParams


__all__ = [
    "align",
    "ColumnMap",
    "Tolerances",
    "MatchResult",
    "PeakMatcher",
    "infer_column_map",
]


def align(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    # --- Column overrides (applied to both tables unless _a/_b suffix given) ---
    mz_col: Optional[str] = None,
    rt_col: Optional[str] = None,
    mz_min_col: Optional[str] = None,
    mz_max_col: Optional[str] = None,
    rt_min_col: Optional[str] = None,
    rt_max_col: Optional[str] = None,
    # --- Per-table intensity column lists ---
    intensity_cols_a: Optional[list[str]] = None,
    intensity_cols_b: Optional[list[str]] = None,
    # --- Pre-built column maps (override all of the above) ---
    map_a: Optional[ColumnMap] = None,
    map_b: Optional[ColumnMap] = None,
    # --- Tolerances ---
    tol: Optional[Tolerances] = None,
    # --- Behaviour flags ---
    rt_drift_correction: bool = True,
    require_intensity_corr: bool = True,
    stat_params: Optional["StatTestParams"] = None,
    # --- Output labels ---
    label_a: str = "A",
    label_b: str = "B",
) -> MatchResult:
    """
    Align two LCMS feature tables and identify matched / unique peaks.

    This is the main entry point for peakalign. Column roles are inferred
    automatically from column names if not specified.

    Parameters
    ----------
    df_a, df_b : pd.DataFrame
        Feature tables to align. Can be from any software (XCMS, MZmine,
        Skyline, MetaboAnalyst, MS-DIAL, Progenesis, etc.).

    mz_col : str, optional
        Name of the m/z column. Applied to both tables. Auto-detected if None.
    rt_col : str, optional
        Name of the RT column. Applied to both tables. Auto-detected if None.
    mz_min_col, mz_max_col : str, optional
        Names of m/z bound columns. Optional — used to improve match scoring.
    rt_min_col, rt_max_col : str, optional
        Names of RT bound columns. Optional.

    intensity_cols_a : list[str], optional
        Per-sample intensity column names in df_a. Auto-detected if None.
    intensity_cols_b : list[str], optional
        Per-sample intensity column names in df_b. Auto-detected if None.

    map_a, map_b : ColumnMap, optional
        Fully pre-built column maps. If supplied, all column override
        arguments above are ignored for that table.

    tol : Tolerances, optional
        Matching tolerances. Defaults: 10 ppm, 30 s RT, 0.7 intensity corr.

    rt_drift_correction : bool
        If True (default), estimate and correct systematic RT drift between
        tables before final matching.

    require_intensity_corr : bool
        If True (default), candidate pairs below tol.min_intensity_corr are
        rejected. If False, low correlation reduces score but does not
        disqualify the pair (useful when tables have very few shared samples).

    label_a, label_b : str
        Short labels appended to column names in the matched output to
        indicate provenance (default "A" and "B").

    Returns
    -------
    MatchResult
        .matched     — features present in both tables
        .unique_a    — features only in table A
        .unique_b    — features only in table B
        .diagnostics — summary statistics dict
        .summary()   — printable summary string
        .to_excel()  — write all three sheets to an Excel file

    Examples
    --------
    Auto-detect everything:

    >>> result = align(df_a, df_b)

    Specify columns explicitly (Skyline style):

    >>> result = align(
    ...     df_a, df_b,
    ...     mz_col="Precursor Mz",
    ...     rt_col="Best Retention Time",
    ...     intensity_cols_a=[c for c in df_a.columns if "Area" in c],
    ...     intensity_cols_b=[c for c in df_b.columns if "Area" in c],
    ... )

    Relaxed tolerances, no intensity filter:

    >>> result = align(
    ...     df_a, df_b,
    ...     tol=Tolerances(mz_ppm=20, rt_seconds=60),
    ...     require_intensity_corr=False,
    ... )
    """
    # Build column maps
    if map_a is None:
        map_a = infer_column_map(
            df_a,
            intensity_cols=intensity_cols_a,
            mz_col=mz_col,
            rt_col=rt_col,
            mz_min_col=mz_min_col,
            mz_max_col=mz_max_col,
            rt_min_col=rt_min_col,
            rt_max_col=rt_max_col,
        )
    if map_b is None:
        map_b = infer_column_map(
            df_b,
            intensity_cols=intensity_cols_b,
            mz_col=mz_col,
            rt_col=rt_col,
            mz_min_col=mz_min_col,
            mz_max_col=mz_max_col,
            rt_min_col=rt_min_col,
            rt_max_col=rt_max_col,
        )

    matcher = PeakMatcher(
        tol=tol or Tolerances(),
        stat_params=stat_params,
        rt_drift_correction=rt_drift_correction,
        require_intensity_corr=require_intensity_corr,
    )

    return matcher.match(df_a, df_b, map_a, map_b, label_a=label_a, label_b=label_b)