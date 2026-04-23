# peakalign

**Post-hoc peak correspondence for LC-MS metabolomics feature tables.**

`peakalign` identifies which peaks are shared between two LC-MS feature tables and which are unique to each — without requiring raw data, software-specific formats, or manual inspection. It works on any CSV export from any peak-picking software.

---

## The problem

Untargeted LC-MS metabolomics produces a **feature table** — a matrix of detected peaks (rows) by samples (columns). When the same biological samples are processed under different acquisition settings, software versions, or instrument methods, you end up with two separate feature tables. The rows do not automatically correspond to each other, even when they represent the same underlying metabolite.

Existing alignment tools (XCMS, MZmine, OpenMS) solve this problem during processing from raw data files. They cannot align two **pre-built feature tables** — which is the situation you are in when comparing outputs from different software, different parameter sets, or different labs.

`peakalign` fills this gap.

---

## How it works

Matching uses three criteria evaluated simultaneously for every candidate pair:

| Criterion | Method |
|---|---|
| m/z similarity | Gaussian kernel over ppm deviation |
| RT similarity | Gaussian kernel over Δ seconds |
| Intensity correlation | Pearson or Spearman r across shared biological samples |

These are combined into a **composite confidence score [0–1]**. When multiple candidates compete for the same feature, `peakalign` uses the **Hungarian algorithm** (bipartite assignment) to find the globally optimal 1:1 pairing across the entire dataset — not greedy nearest-neighbor matching.

After matching, every accepted pair receives four **statistical tests**:

| Test | What it tells you |
|---|---|
| Permutation p-value | Is the intensity correlation significantly better than chance? |
| BH-FDR q-value | Is it significant after correcting for all matched pairs? |
| Match specificity | How much better was this match than the next-best candidate? |
| Bootstrap 95% CI on r | How stable is the correlation estimate across samples? |

Matches are then stratified into four **evidence tiers** for transparent reporting.

---

## Installation

```bash
# Basic install
pip install peakalign

# With RT drift correction (recommended)
pip install "peakalign[drift]"
```

**Requirements:** Python ≥ 3.9, numpy, pandas, scipy, openpyxl.
Optional: scikit-learn (for polynomial RT drift correction).

---

## Quickstart

```python
import pandas as pd
from peakalign import align, Tolerances, StatTestParams

df_a = pd.read_csv("setting1.csv", index_col=0)
df_b = pd.read_csv("setting2.csv", index_col=0)

result = align(
    df_a, df_b,
    tol=Tolerances(mz_ppm=10, rt_seconds=30),
    stat_params=StatTestParams(n_permutations=1000, n_bootstrap=500),
)

print(result.summary())
result.triage_summary()
result.to_excel_tiered("peakalign_results.xlsx")
```

---

## Input format

`peakalign` accepts any CSV or TSV feature table. Column roles are **auto-detected** from column names — no configuration required for standard exports.

| Role | Auto-detected names |
|---|---|
| m/z | `mz`, `m/z`, `mass`, `precursor mz`, `average mz`, ... |
| RT | `rt`, `retention time`, `best retention time`, `apex rt`, ... |
| m/z bounds | `mzmin`, `mzmax`, `mz min`, `mz max`, ... |
| RT bounds | `rtmin`, `rtmax`, `rt min`, `rt max`, ... |
| Intensities | All remaining numeric columns with variance |
| Annotations | `isotopes`, `adduct`, `pcgroup`, `charge`, `formula`, ... |

For non-standard exports (e.g. Skyline, custom column names), override explicitly:

```python
result = align(
    df_a, df_b,
    mz_col="Precursor Mz",
    rt_col="Best Retention Time",
    intensity_cols_a=[c for c in df_a.columns if c.startswith("Area_")],
    intensity_cols_b=[c for c in df_b.columns if c.startswith("Area_")],
)
```

Compatible with exports from: XCMS, MZmine, Skyline, MetaboAnalyst, MS-DIAL, Progenesis, and any software that produces tabular peak lists.

---

## Output

### `result.matched`

All features present in both tables. Columns from both tables are included (suffixed `_A` and `_B`) plus match metadata:

**Core match columns**

| Column | Description |
|---|---|
| `_confidence` | Composite match score [0–1] |
| `_delta_mz_ppm` | m/z deviation in ppm |
| `_delta_rt` | RT deviation in seconds (B − A) |
| `_intensity_corr` | Pearson r across shared samples |

**Statistical testing columns**

| Column | Description |
|---|---|
| `_perm_pvalue` | Permutation p-value for intensity correlation |
| `_perm_qvalue` | BH-FDR corrected q-value |
| `_fdr_significant` | True if q-value < fdr_alpha |
| `_specificity` | Gap between best and second-best candidate score |
| `_corr_ci_low` | Lower bound of 95% bootstrap CI on r |
| `_corr_ci_high` | Upper bound of 95% bootstrap CI on r |
| `_ci_width` | CI width — wider values indicate unstable correlation |
| `_stat_notes` | Pipe-delimited flags: `wide_ci`, `low_specificity`, `internal_log1p_applied` |

### `result.unique_a` / `result.unique_b`

Features detected in only one table. Original columns preserved, no suffix.

### `result.summary()`

```
=======================================================
PeakAlign Match Summary
=======================================================
  Table A features       : 3604
  Table B features       : 4811
  Matched pairs          : 2159
  Unique to A            : 1445  (40.1%)
  Unique to B            : 2652  (55.1%)

  --- Match quality ---
  Median Δmz (ppm)       : 0.491
  Median |Δrt| (s)       : 1.21
  Median intensity r     : 0.979
  Mean confidence        : 0.964

  --- Statistical testing ---
  FDR-significant matches: 2055 / 2159  (95.2%)
  Median specificity     : 0.985
  Median CI width (r)    : 0.335
=======================================================
```

---

## Evidence triage

After running statistical tests, `peakalign` stratifies every matched pair into one of four evidence tiers. This is designed for transparent reporting in publications.

```python
result.triage_summary()
```

```
=================================================================
PeakAlign Triage Summary
=================================================================
  Total matched pairs : 2159
  FDR alpha           : 0.05
  CI width threshold  : 0.4
  Specificity threshold: 0.5

  Tier          n      %    Med r     Med q   Med spec   Med CIw
  ------------------------------------------------------------
  Tier 1     1206  55.9%    0.990    0.0013      0.992     0.116
  Tier 2      766  35.5%    0.945    0.0013      0.970     0.595
  Tier 3       83   3.8%    0.923    0.0090      0.965     1.086
  Tier 4      104   4.8%    0.934    0.0889      0.974     0.912
  ------------------------------------------------------------

  Tiers defined as:
    Tier 1 : FDR sig + CI tight + specific match
    Tier 2 : FDR sig + wide/failed CI or borderline specificity
    Tier 3 : FDR sig + CI crosses zero or low specificity
    Tier 4 : q >= fdr_alpha
=================================================================
```

| Tier | Recommendation |
|---|---|
| Tier 1 | Use directly in downstream analysis |
| Tier 2 | Include; report separately in supplementary material |
| Tier 3 | Manual review before inclusion |
| Tier 4 | Exclude from downstream analysis |

Access individual tiers programmatically:

```python
t = result.triage()
t["tier1"]    # high confidence matches
t["tier2"]    # accepted, flagged for review
t["tier3"]    # borderline, manual review recommended
t["tier4"]    # not FDR-significant, exclude
t["summary"]  # one-row-per-tier metrics table → paste into manuscript
```

---

## Saving results

```python
# Simple: matched + unique tables in one Excel file
result.to_excel("results.xlsx")

# Tiered: one sheet per evidence tier (recommended for publication)
result.to_excel_tiered("results_tiered.xlsx")
```

The tiered workbook contains seven sheets: `tier1_accepted`, `tier2_flag_review`, `tier3_borderline`, `tier4_rejected`, `unique_A`, `unique_B`, and `triage_summary`.

---

## Statistical testing configuration

```python
from peakalign import StatTestParams

stat_params = StatTestParams(
    n_permutations=1000,      # permutations for p-value (use 1000+ for publication)
    n_bootstrap=500,          # bootstrap resamples for CI (use 500+ for publication)
    fdr_alpha=0.05,           # FDR threshold
    corr_method="pearson",    # or "spearman"
    auto_log_transform=True,  # detect and normalize raw intensities internally
    random_seed=42,           # for reproducibility
)
```

**On raw vs log-transformed data:** `peakalign` detects whether intensity data is raw or log-transformed using a dynamic range heuristic. If raw data is detected (dynamic range > 10,000×), `log1p` normalization is applied internally before permutation testing and bootstrap CI estimation only. The user's data is never modified. This behavior can be disabled with `auto_log_transform=False`.

**Performance:** The permutation and bootstrap loops are fully vectorized using numpy matrix operations. For a typical dataset (2,000 matched pairs, 18 samples, 1,000 permutations), runtime is under 30 seconds on a standard laptop.

---

## Tolerances

```python
from peakalign import Tolerances

tol = Tolerances(
    mz_ppm=10.0,                    # m/z tolerance in ppm
    mz_da=None,                     # absolute Da tolerance (uses wider if both set)
    rt_seconds=30.0,                # RT tolerance in seconds
    min_intensity_corr=0.7,         # minimum r to accept a match
    corr_method="pearson",          # or "spearman"
    confidence_weights={            # weights must sum to 1.0
        "mz": 0.35,
        "rt": 0.30,
        "intensity": 0.35,
    },
)
```

---

## RT drift correction

Systematic RT shifts between tables (e.g. from different gradient lengths or different acquisition days) are detected and corrected automatically before final matching. A polynomial model is fit to tentative anchor matches and used to correct the RT values of table A.

```python
result = align(df_a, df_b, rt_drift_correction=True)   # default
result = align(df_a, df_b, rt_drift_correction=False)  # disable
```

Requires `scikit-learn`. Install with `pip install "peakalign[drift]"`. Falls back to linear correction if scikit-learn is not available.

---

## Feature recovery summary

To generate the full feature accounting for manuscript reporting:

```python
n_a = len(df_a)
n_b = len(df_b)
n_matched = len(result.matched)
t = result.triage()
n_supported = len(t["tier1"]) + len(t["tier2"])

print(f"Setting A total features  : {n_a}")
print(f"Setting B total features  : {n_b}")
print(f"Matched pairs             : {n_matched}")
print(f"% of A matched            : {100*n_matched/n_a:.1f}%")
print(f"% of B matched            : {100*n_matched/n_b:.1f}%")
print(f"Unique to A               : {len(result.unique_a)}")
print(f"Unique to B               : {len(result.unique_b)}")
print(f"Statistically supported   : {n_supported} ({100*n_supported/n_matched:.1f}% of matched)")
```

---

## Running in Google Colab

Upload your feature tables and the `peakalign/` folder to `/content/`, then:

```python
import sys
sys.path.insert(0, "/content")

import pandas as pd
from peakalign import align, Tolerances, StatTestParams

df_a = pd.read_csv("/content/setting1.csv", index_col=0)
df_b = pd.read_csv("/content/setting2.csv", index_col=0)

result = align(df_a, df_b,
    stat_params=StatTestParams(n_permutations=1000, n_bootstrap=500))
result.triage_summary()
```

---

## Running tests

```bash
pip install -e ".[dev]"
pytest tests/ -v
```

---

## Roadmap

- [ ] HTML report with visual diagnostics
- [ ] CLI (`peakalign table_a.csv table_b.csv --out results/`)
- [ ] Adduct-aware matching (`[M+H]+` ↔ `[M+Na]+`)
- [ ] Multi-table alignment (N > 2 tables simultaneously)
- [ ] Streamlit web app

---

## License

MIT
