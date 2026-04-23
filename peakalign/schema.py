"""
schema.py
---------
Column role detection and mapping for feature tables from any LCMS software.
Handles XCMS, MZmine, Skyline, MetaboAnalyst, MS-DIAL, Progenesis, etc.
"""

from __future__ import annotations

import re
import warnings
from dataclasses import dataclass, field
from typing import Optional

import pandas as pd


# ---------------------------------------------------------------------------
# Known column name patterns per role (case-insensitive)
# ---------------------------------------------------------------------------

_MZ_PATTERNS = [
    r"^mz$", r"^m/z$", r"^mass$", r"^precursor.?mz$", r"^precursor.?mass$",
    r"^average.?mz$", r"^monoisotopic.?mass$", r"^exact.?mass$", r"^mw$",
]

_MZ_MIN_PATTERNS = [r"^mzmin$", r"^mz.?min$", r"^min.?mz$", r"^precursor.?mz.?min$"]
_MZ_MAX_PATTERNS = [r"^mzmax$", r"^mz.?max$", r"^max.?mz$", r"^precursor.?mz.?max$"]

_RT_PATTERNS = [
    r"^rt$", r"^retention.?time$", r"^rt.?mean$", r"^best.?retention.?time$",
    r"^average.?rt$", r"^apex.?rt$", r"^peak.?rt$", r"^med.?rt$",
]

_RT_MIN_PATTERNS = [r"^rtmin$", r"^rt.?min$", r"^min.?rt$", r"^start.?rt$"]
_RT_MAX_PATTERNS = [r"^rtmax$", r"^rt.?max$", r"^max.?rt$", r"^end.?rt$"]

_ANNOTATION_PATTERNS = [
    r"^isotopes$", r"^adduct$", r"^pcgroup$", r"^charge$", r"^formula$",
    r"^annotation$", r"^name$", r"^compound$", r"^id$", r"^feature.?id$",
    r"^flag$", r"^ms2$", r"^msms$", r"^score$", r"^fdr$",
]

# Prefixes commonly prepended to sample names in exports
_INTENSITY_PREFIXES = ["area_", "height_", "intensity_", "peak_area_", "peakarea_"]


def _matches_any(col: str, patterns: list[str]) -> bool:
    col_clean = col.strip().lower()
    return any(re.fullmatch(p, col_clean) for p in patterns)


# ---------------------------------------------------------------------------
# ColumnMap dataclass
# ---------------------------------------------------------------------------

@dataclass
class ColumnMap:
    """
    Describes the role of every column in a feature table.

    Parameters
    ----------
    mz : str
        Name of the m/z centroid column.
    rt : str
        Name of the retention time centroid column.
    intensity_cols : list[str]
        Names of per-sample intensity columns.
    mz_min : str or None
        Name of the mzmin column, if present.
    mz_max : str or None
        Name of the mzmax column, if present.
    rt_min : str or None
        Name of the rtmin column, if present.
    rt_max : str or None
        Name of the rtmax column, if present.
    annotation_cols : list[str]
        Columns that are metadata / annotations (carried through, not used
        for matching).
    """

    mz: str
    rt: str
    intensity_cols: list[str]
    mz_min: Optional[str] = None
    mz_max: Optional[str] = None
    rt_min: Optional[str] = None
    rt_max: Optional[str] = None
    annotation_cols: list[str] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def all_known_cols(self) -> list[str]:
        """Return every column whose role is explicitly known."""
        cols = [self.mz, self.rt] + self.intensity_cols + self.annotation_cols
        for c in [self.mz_min, self.mz_max, self.rt_min, self.rt_max]:
            if c is not None:
                cols.append(c)
        return cols

    def validate_against(self, df: pd.DataFrame) -> None:
        """Raise ValueError if any mapped column is missing from df."""
        missing = [c for c in self.all_known_cols() if c not in df.columns]
        if missing:
            raise ValueError(
                f"ColumnMap references columns not found in DataFrame: {missing}\n"
                f"Available columns: {list(df.columns)}"
            )

    def summary(self) -> str:
        lines = [
            f"  mz            : {self.mz}",
            f"  rt            : {self.rt}",
            f"  mz_min        : {self.mz_min}",
            f"  mz_max        : {self.mz_max}",
            f"  rt_min        : {self.rt_min}",
            f"  rt_max        : {self.rt_max}",
            f"  intensity_cols: {self.intensity_cols}",
            f"  annotation_cols:{self.annotation_cols}",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto-detection
# ---------------------------------------------------------------------------

def infer_column_map(
    df: pd.DataFrame,
    intensity_cols: Optional[list[str]] = None,
    mz_col: Optional[str] = None,
    rt_col: Optional[str] = None,
    mz_min_col: Optional[str] = None,
    mz_max_col: Optional[str] = None,
    rt_min_col: Optional[str] = None,
    rt_max_col: Optional[str] = None,
) -> ColumnMap:
    """
    Infer a ColumnMap from a DataFrame, with optional user overrides.

    Any explicitly supplied argument takes precedence over auto-detection.
    Auto-detection falls back to pattern matching on column names, then
    heuristic numeric-column analysis.

    Parameters
    ----------
    df : pd.DataFrame
        The feature table.
    intensity_cols : list[str], optional
        Explicitly specify sample intensity columns. If None, auto-detected.
    mz_col, rt_col, ... : str, optional
        Override auto-detection for individual roles.

    Returns
    -------
    ColumnMap
    """
    cols = list(df.columns)

    def _pick(override, patterns, label):
        if override is not None:
            if override not in cols:
                raise ValueError(f"Specified {label} column '{override}' not in DataFrame.")
            return override
        matches = [c for c in cols if _matches_any(c, patterns)]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            warnings.warn(
                f"Multiple candidate columns for {label}: {matches}. "
                f"Using '{matches[0]}'. Pass explicit override to suppress.",
                stacklevel=3,
            )
            return matches[0]
        return None

    # Required roles
    mz = _pick(mz_col, _MZ_PATTERNS, "mz")
    rt = _pick(rt_col, _RT_PATTERNS, "rt")

    if mz is None:
        raise ValueError(
            "Could not auto-detect the m/z column. "
            "Pass mz_col='<your column name>' explicitly."
        )
    if rt is None:
        raise ValueError(
            "Could not auto-detect the RT column. "
            "Pass rt_col='<your column name>' explicitly."
        )

    # Optional bound roles
    mz_min = _pick(mz_min_col, _MZ_MIN_PATTERNS, "mz_min")
    mz_max = _pick(mz_max_col, _MZ_MAX_PATTERNS, "mz_max")
    rt_min = _pick(rt_min_col, _RT_MIN_PATTERNS, "rt_min")
    rt_max = _pick(rt_max_col, _RT_MAX_PATTERNS, "rt_max")

    # Annotation columns
    annotation_cols = [c for c in cols if _matches_any(c, _ANNOTATION_PATTERNS)]

    # Intensity columns
    if intensity_cols is not None:
        missing = [c for c in intensity_cols if c not in cols]
        if missing:
            raise ValueError(f"Specified intensity_cols not found in DataFrame: {missing}")
        int_cols = list(intensity_cols)
    else:
        int_cols = _infer_intensity_cols(
            df,
            exclude={mz, rt, mz_min, mz_max, rt_min, rt_max} | set(annotation_cols),
        )

    if not int_cols:
        raise ValueError(
            "Could not auto-detect intensity (sample) columns. "
            "Pass intensity_cols=[...] explicitly."
        )

    return ColumnMap(
        mz=mz,
        rt=rt,
        intensity_cols=int_cols,
        mz_min=mz_min,
        mz_max=mz_max,
        rt_min=rt_min,
        rt_max=rt_max,
        annotation_cols=annotation_cols,
    )


def _infer_intensity_cols(df: pd.DataFrame, exclude: set) -> list[str]:
    """
    Heuristically find intensity columns:
    1. Numeric columns not in `exclude`
    2. That have meaningful variance (not constant/zero)
    3. Optionally with known intensity prefixes
    """
    candidates = []
    for col in df.columns:
        if col in exclude or col is None:
            continue
        if not pd.api.types.is_numeric_dtype(df[col]):
            continue
        # Skip columns that are nearly all NaN or zero — not real intensities
        valid = df[col].dropna()
        if len(valid) == 0:
            continue
        if valid.std() == 0 and valid.iloc[0] == 0:
            continue
        candidates.append(col)

    # Prefer columns with known intensity prefixes if any exist
    prefixed = [
        c for c in candidates
        if any(c.lower().startswith(p) for p in _INTENSITY_PREFIXES)
    ]
    return prefixed if prefixed else candidates
