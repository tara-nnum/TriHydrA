# TriHydrA Layer 1: intrinsic time-series diagnostics

## 1. Purpose

Layer 1 examines one discharge series on its own. It asks whether the supplied
time axis and values contain conditions that may affect interpretation before
the record is compared with another series or placed in hydrological context.

Layer 1 does **not** automatically declare data to be wrong. Its findings are
screening evidence. A `Needs review` result means that the enabled checks found
enough evidence to justify inspection.

The source series is never imputed or overwritten. Individual calculations may
temporarily ignore missing values, but the raw record remains unchanged.

## 2. What Layer 1 contains

Layer 1 runs ten diagnostics. Nine contribute to the composite assessment.
Zero-flow regime is descriptive and therefore has no composite weight.

| Diagnostic | Main output | Composite role | Default weight |
|---|---|---:|---:|
| Missing values | Internal missing count and percentage | Scored | 2 |
| Long gaps | Missing intervals and longest gap | Scored | 2 |
| Negative discharge | Count, dates, values, maximum magnitude | Scored | 3 |
| Duplicate timestamps | Duplicate groups and conflicting values | Scored | 3 |
| Timestep consistency | Non-daily intervals and reversed ordering | Scored | 3 |
| Zero-flow regime | Zero-flow frequency, spells, and seasonality | Descriptor only | – |
| Non-zero plateau | Persistent repeated non-zero values | Scored | 1 |
| Spike/dip | Isolated one-observation impulse candidates | Scored | 1 |
| Step shift | Persistent structural regime boundaries | Scored | 3 |
| Epoch drift | Stable, rising, falling, or mixed long-term behaviour | Scored | 2 |

## 3. Threshold provenance

TriHydrA distinguishes three kinds of settings:

1. **Literature method** – the underlying statistical or hydrological method
   comes from published work. For example, epoch slopes use the Theil–Sen
   estimator.
2. **TriHydrA review policy** – a transparent, configurable decision rule used
   to turn evidence into a review tier. The 5% and 15% missingness boundaries
   are policy settings, not universal hydrological constants.
3. **Computational default** – a practical value needed to run a method, such
   as a minimum number of valid days or rounding precision. These settings may
   need adjustment for a different temporal resolution, unit, or organisation.

None of the numerical Tier 1/Tier 2/Tier 3 cutoffs should be described as a
universal definition of good or bad streamflow data.

## 4. Input and record boundaries

Layer 1 expects a dated daily `pandas.Series`. Checks work between the first and
last valid values. Leading and trailing `NaN` padding is counted for provenance
but is not treated as internal missingness.

Two distinct situations are intentionally kept separate:

- A row exists for a date but its discharge is `NaN`: **missing-values check**.
- A calendar date is absent from the index: **timestep-consistency check**.

Duplicate dates are also separated from timestep spacing so the same defect is
not counted twice.

## 5. Missing values

### What it detects

Explicit `NaN` observations between the first and last valid observations.

### Why it matters

Missing observations reduce the evidence available for annual signatures,
events, trends, and comparisons. The percentage describes record completeness;
it does not infer what the missing values would have been.

### Calculation

```text
internal missingness (%) = 100 × internal NaN rows / rows in valid record
```

### Configuration

| TOML setting | Default | Meaning | Provenance |
|---|---:|---|---|
| `layer1.missing_values.enabled` | `true` | Run this diagnostic | User switch |
| `tier_2_minimum_percent` | 5.0 | Tier 2 begins at 5% | TriHydrA policy |
| `tier_1_above_percent` | 15.0 | Tier 1 above 15% | TriHydrA policy |

### Tier rule

- Tier 3: less than 5% internal missingness.
- Tier 2: 5% through 15%, inclusive.
- Tier 1: more than 15%.

### Outputs

Internal, leading, trailing, and total `NaN` counts; internal missing
percentage; missing timestamps; interval evidence; first and last valid dates.

### Limitations

The check measures missingness, not its cause or hydrological importance. A
small strategically placed gap can matter more than a larger gap during an
unimportant period; the long-gap diagnostic adds persistence information.

## 6. Long gaps

### What it detects

Consecutive internal `NaN` rows. Under the daily input contract, consecutive
rows represent consecutive days.

### Calculation

All internal missing runs are retained as evidence. Runs longer than the
reporting threshold are marked for display. Composite tiering uses both the
longest run and its dependency on overall missingness.

### Configuration

| TOML setting | Default | Meaning | Provenance |
|---|---:|---|---|
| `layer1.long_gaps.enabled` | `true` | Run this diagnostic | User switch |
| `minimum_reported_gap_days` | 3 | Display runs longer than 3 rows | Computational default |
| `long_gap_definition_days` | 5 | A run over 5 days counts toward long-gap share | TriHydrA policy |
| `tier_2_minimum_days` | 6 | Direct Tier 2 duration | TriHydrA policy |
| `tier_1_minimum_days` | 31 | Direct Tier 1 duration | TriHydrA policy |
| `tier_2_missing_share` | 0.50 | Dependency rule for Tier 2 missingness | TriHydrA policy |
| `tier_1_missing_share` | 0.25 | Dependency rule for Tier 1 missingness | TriHydrA policy |

### Tier rule

- Tier 1 when the longest gap is at least 31 days, **or** missingness is Tier 1
  and at least 25% of missing days occur in runs longer than 5 days.
- Tier 2 when the longest gap is at least 6 days, **or** missingness is Tier 2
  and at least 50% of missing days occur in runs longer than 5 days.
- Tier 3 otherwise.

Tier 1 is evaluated first. If the missing-values check is disabled or cannot be
assessed, duration can still determine the long-gap tier, but the dependency
clauses cannot activate.

### Outputs

Every internal missing interval, displayed long-gap intervals, flagged dates,
longest gap, and configured reporting threshold.

### Limitations

The count is based on rows. An absent calendar date belongs to timestep
consistency rather than this diagnostic. This separation prevents hidden
reindexing and protects the raw input.

## 7. Duplicate timestamps

### What it detects

Dates occurring more than once, including whether repeated rows contain
conflicting values.

### Tier rule

- Tier 1: one or more duplicated dates.
- Tier 3: none.

There is no Tier 2 because a duplicated timestamp is a structural ambiguity.

### Outputs

Unique duplicated dates, all rows in duplicate groups, extra rows beyond the
first, conflicting groups, and the original values in each group.

### Limitations

TriHydrA reports duplicates but does not choose which value is correct and does
not delete or aggregate source rows.

## 8. Timestep consistency

### What it detects

- Sorted unique-date intervals that are not exactly one day.
- Backward transitions in the original source ordering.

Duplicate dates are excluded from spacing assessment because they have their
own diagnostic.

### Tier rule

- Tier 1: at least one irregular interval or an out-of-order source axis.
- Tier 3: a consistently daily, ordered axis.

### Outputs

Irregular interval count, previous/current dates, interval length in days,
out-of-order transitions, and flagged timestamps.

### Limitations

This rule assumes daily data. Sub-daily, monthly, or intentionally irregular
records must be converted to the documented daily contract before assessment.

## 9. Negative discharge

### What it detects

Values below a small negative numerical tolerance. Values between zero and
`-tolerance` are treated as numerical noise by this check.

### Configuration

| TOML setting | Default | Meaning | Provenance |
|---|---:|---|---|
| `layer1.negative_discharge.enabled` | `true` | Run this diagnostic | User switch |
| `tolerance` | 0.001 | Detection boundary in source units | Computational/project default |
| `low_flow_reference_quantile` | 0.05 | Percentile Q05, equivalent to conventional FDC Q95 | TriHydrA policy |
| `tier_1_reference_multiplier` | 1.0 | Multiplier applied to low-flow reference | TriHydrA policy |

### Tier rule

The record's valid values are clipped at zero before calculating percentile
Q05. Let `M` be the greatest absolute magnitude among detected negative values
and let `L` be Q05 × the configured multiplier.

- Tier 3: no detected negative values.
- Tier 2: negative values exist and `M < L`.
- Tier 1: `M ≥ L`.

### Outputs

Count, timestamps, individual values, maximum negative magnitude, tolerance,
station-specific low-flow reference, and Tier 1 threshold.

### Limitations

The tolerance is expressed in source units and must be appropriate for those
units. The check cannot distinguish a numerical model artefact from a datum,
rating-curve, unit, or sign-convention problem.

## 10. Zero-flow regime

### What it describes

The prevalence, duration, and monthly distribution of valid zero discharge.
Values are rounded before zero comparison.

### Why it is not scored

Zero flow can be normal in intermittent or ephemeral rivers. It is hydrological
context, not inherently a quality-control failure. Consequently, this check
never flags the series and contributes no points to the Layer 1 composite.

### Configuration

| TOML setting | Default | Meaning | Provenance |
|---|---:|---|---|
| `layer1.zero_flow_regime.enabled` | `true` | Calculate descriptor | User switch |
| `decimals` | 3 | Rounding precision for zero | Computational default |

### Outputs

Zero count and fraction among valid observations, spell count, longest spell,
spell dates and durations, monthly zero-flow ratios, and months containing
zero flow.

Completely missing calendar months remain unavailable rather than being
interpreted as zero flow. Missing monthly values are safely excluded from
numeric conversion and zero-flow calculations.

### Limitations

Rounding defines what counts as zero. A single setting cannot represent the
measurement resolution of every source dataset.

## 11. Non-zero plateau / low variability

### What it detects

Consecutive daily runs of the same rounded, valid, non-zero discharge value.

### Configuration

| TOML setting | Default | Meaning | Provenance |
|---|---:|---|---|
| `layer1.low_variability.enabled` | `true` | Run this diagnostic | User switch |
| `minimum_plateau_days` | 15 | Minimum displayed candidate duration | Computational/project default |
| `decimals` | 3 | Equality precision | Computational default |
| composite `tier_1_minimum_days` | 31 | Tier 1 duration | TriHydrA policy |

### Tier rule

- Tier 3: no retained plateau.
- Tier 2: a retained plateau exists but the longest is under 31 days.
- Tier 1: the longest retained plateau is at least 31 days.

### Outputs

Plateau count; dates; rounded value; observation and calendar duration; values
before and after; longest duration; and all flagged timestamps.

### Limitations

Rounding and reporting precision can create or remove apparent equality.
Stable regulated flow, rating-table discretisation, frozen sensors, and genuine
hydrological persistence can look similar; the result is a candidate for
interpretation, not proof of an instrument fault.

## 12. Isolated spike/dip candidates

### What it detects

One-observation impulses that rise or fall sharply and then recover. The
algorithm is deliberately designed not to label every flood peak as a spike.

### Calculation

For a candidate value `Q(t)` with neighbours `Q(t-1)` and `Q(t+1)`:

- incoming change = `Q(t) - Q(t-1)`;
- outgoing change = `Q(t+1) - Q(t)`;
- two-sided jump = the smaller absolute change;
- recovery = `1 - |Q(t+1)-Q(t-1)| / (|incoming|+|outgoing|)`;
- station-relative score = two-sided jump divided by the calendar month's
  median valid one-day absolute change.

Selection requires all of the following:

1. a turning point with opposite incoming and outgoing signs;
2. recovery at or above the configured minimum;
3. jump at or above that month's high absolute-change quantile;
4. score at or above a robust cutoff; and
5. complete daily context across five consecutive days.

The robust cutoff is the maximum of the minimum score, the configured score
quantile, and `median + multiplier × 1.4826 × MAD`. The median absolute
deviation (MAD) is a standard robust-scale idea; the exact combined rule and
its defaults are TriHydrA design choices.

A coherent five-day peak (`rising, rising, peak, falling, falling`) or trough is
rejected as a spike/dip candidate. Candidates without full five-day context are
also rejected.

### Configuration

| TOML setting | Default | Meaning | Provenance |
|---|---:|---|---|
| `minimum_recovery` | 0.80 | Required immediate return toward neighbours | TriHydrA policy |
| `minimum_score` | 8.0 | Minimum station-relative score | TriHydrA policy |
| `absolute_change_reference_quantile` | 0.99 | Monthly absolute-change threshold | TriHydrA policy |
| `score_reference_quantile` | 0.995 | Candidate-score threshold | TriHydrA policy |
| `robust_mad_multiplier` | 6.0 | Robust cutoff multiplier | TriHydrA policy |
| `minimum_outer_change_multiplier` | 1.0 | Five-day coherent-pattern guard | TriHydrA policy |
| composite `tier_1_minimum_unresolved_count` | 6 | Tier 1 begins at six candidates | TriHydrA policy |

### Tier rule

- Tier 3: zero unresolved candidates.
- Tier 2: one through five unresolved candidates.
- Tier 1: six or more unresolved candidates.

When Layer 2 is available, spike/peak cross-check evidence can reduce the
unresolved count when a candidate has plausible event context. This is the
only Layer 1 composite component whose final count may be informed by Layer 2.

### Outputs

Candidate timestamps and types; neighbouring values; changes; jump magnitude;
monthly threshold; recovery; raw score; robust cutoff; rejected coherent
patterns; and rejected incomplete contexts.

### Limitations

The method is a screening heuristic, not a published universal spike detector.
A real flash flood sampled daily may resemble a one-day impulse. Conversely,
an artefact spanning several observations may evade this isolated-point check.

## 13. Step shifts

### What it detects

Persistent changes between adjacent flow regimes. The method is adaptive to
record length and considers multiple robust block features rather than a single
percentage change.

### Calculation overview

1. Daily values are summarised into observed monthly statistics; months need a
   minimum number of valid days.
2. The record is split into adaptive multi-year blocks. Records of at least 12
   years use four-year blocks by default; shorter records use approximately one
   third of their duration, with a one-year minimum.
3. Adequately covered blocks are represented in log space by level, high-flow,
   seasonal-amplitude, and variability features.
4. Robust feature scales standardise differences. Adjacent blocks are merged
   bottom-up until retained boundaries exceed the structural score threshold.
5. Same-direction nearby candidates are consolidated, and level boundaries are
   refined using monthly evidence.
6. Each boundary is tiered by the absolute change in regime median against
   station-specific low-flow references—not percentage change alone.

Percentile notation and flow-duration-curve (FDC) notation run in opposite
directions here:

- percentile Q05 = conventional FDC Q95 (low-flow reference);
- percentile Q25 = conventional FDC Q75.

### Boundary tier rule

- Tier 3: absolute regime-median change ≤ percentile Q05.
- Tier 2: change is above Q05 but below Q25.
- Tier 1: change ≥ percentile Q25.

All retained boundaries are combined rather than allowing one boundary to
automatically dominate the record:

```text
step-shift score = mean(boundary tier points)
```

With default points 0/1/2:

- series Tier 3: score below 1.0;
- series Tier 2: score from 1.0 through 1.5;
- series Tier 1: score above 1.5.

### Main configuration

| Setting | Default | Purpose | Provenance |
|---|---:|---|---|
| `long_record_min_years` | 12 | Switch to long-record block rule | Computational/project default |
| `long_record_block_years` | 4 | Long-record block length | Computational/project default |
| `short_record_divisor` | 3.0 | Short-record block rule | Computational/project default |
| `minimum_block_years` | 1 | Smallest block | Computational default |
| `minimum_valid_days_per_month` | 10 | Monthly evidence coverage | Computational default |
| `minimum_block_coverage` | 0.55 | Required temporal coverage | Computational/project default |
| `minimum_calendar_months` | 8 | Seasonal representation | Computational/project default |
| `structural_threshold` | 3.0 | Standardised feature separation | TriHydrA policy |
| `tier_3_maximum_quantile` | 0.05 | Low-magnitude boundary cutoff | TriHydrA policy |
| `tier_1_minimum_quantile` | 0.25 | Large-magnitude boundary cutoff | TriHydrA policy |
| `refinement_block_fraction` | 0.5 | Boundary-date refinement window | Computational default |
| `consolidation_max_block_widths` | 2.0 | Nearby-boundary consolidation | Computational default |

### Outputs

All retained boundaries including Tier 3 evidence; public Tier 1/2 boundary
dates; before/after medians; absolute changes; station thresholds; structural
feature scores; regime summaries; adaptive-block metadata; boundary counts;
station-level score and tier; and resolved settings.

### Limitations

This is a custom robust structural-screening workflow, not a classical test
with a universal p-value. Results depend on record duration, coverage, and
station distribution. A persistent physical change, regulation, rating-curve
revision, datum change, or data-processing change can all produce a boundary.

## 14. Epoch drift

### What it detects

Multi-year evolution in typical flow level after removing the usual
month-of-year pattern. It describes assessed periods as stable, rising, or
falling and then summarises the record as stable, one-directional, or mixed.

### Calculation

1. Observed monthly medians are calculated when a month has enough valid days.
2. A station-specific positive offset permits log transformation.
3. Median log flow for each calendar month forms a monthly climatology.
4. The climatology is subtracted to form seasonally adjusted monthly anomalies.
5. A year receives an annual median anomaly when enough months are valid.
6. Consecutive valid years are divided into balanced epochs of at least five
   years by default.
7. Each epoch receives a robust Theil–Sen slope.
8. The absolute slope over the epoch span is divided by a robust annual noise
   scale. If the resulting change score is below the configured threshold, the
   epoch is stable; otherwise its slope direction determines rising/falling.
9. The share of assessed years belonging to stable epochs determines the tier.

The Theil–Sen slope is literature-derived. Seasonal adjustment, noise scaling,
epoch construction, and tier thresholds are TriHydrA implementation policy.

### Tier rule

- Tier 3 / Stable: at least 75% of assessed years are stable.
- Tier 2 / Drifting: at least 50% but less than 75% are stable.
- Tier 1 / Drifting: less than 50% are stable.

### Main configuration

| Setting | Default | Purpose | Provenance |
|---|---:|---|---|
| `minimum_valid_days_per_month` | 10 | Monthly evidence coverage | Computational default |
| `minimum_valid_months_per_year` | 8 | Annual evidence coverage | Computational/project default |
| `epoch_years` | 5 | Minimum base-epoch duration | TriHydrA policy |
| `minimum_valid_annual_levels` | 5 | Minimum continuous run | Computational default |
| `annual_noise_floor_log` | 0.03 | Prevent unstable division by tiny noise | Computational default |
| `meaningful_epoch_change_score` | 1.0 | Stable/drifting boundary | TriHydrA policy |
| `tier_3_minimum_stable_fraction` | 0.75 | Stable tier boundary | TriHydrA policy |
| `tier_2_minimum_stable_fraction` | 0.50 | Intermediate tier boundary | TriHydrA policy |
| `overview_epochs_per_segment` | 4 | Consolidated display grouping | Display/computational default |
| `maximum_overview_slopes` | 4 | Maximum display overview segments | Display default |

### Outputs

Valid annual levels; seasonally adjusted annual evidence; noise scale; every
base epoch and fitted value; robust slopes in percent per year; epoch change
scores and states; stable-year fraction; tier; dominant behaviour; and a
consolidated overview used for readable plotting.

### Limitations

The diagnostic needs a continuous run of enough valid annual levels. It detects
statistical evolution, not its cause. Climate variability, abstraction,
regulation, land-use change, rating changes, and model drift may look similar.
Calendar-year aggregation is used; hydrological-year definitions are not
inferred station by station.

## 15. Layer 1 composite assessment

### Component scoring

Each enabled and assessable component receives a tier and tier points:

| Tier | Default points | Interpretation |
|---|---:|---|
| Tier 3 | 0 | No concern under the configured rule |
| Tier 2 | 1 | Intermediate concern |
| Tier 1 | 2 | Strong concern |

```text
component contribution = component weight × tier points

score percentage = 100 × sum(contributions)
                   / sum(assessable weights × maximum tier points)
```

The percentage is therefore relative to the maximum possible score of the
checks that were both **enabled and assessable** in this run.

### Final classification

| Score percentage | Default classification |
|---:|---|
| below 7.5% | No review needed |
| 7.5% to below 20% | Minor concerns |
| 20% or more | Needs review |
| no assessable enabled checks | Not assessable |

These cutoffs are configurable TriHydrA review policy.

### What happens when checks are disabled?

Disabled checks are removed entirely from:

- the component table;
- the numerator;
- the maximum possible score; and
- evidence-coverage calculations.

They do **not** contribute zero points as though they had passed. This prevents
disabled checks from diluting the result.

If all nine composite checks run, the assessment scope is `Full`. If one or
more are disabled, it is `Focused`, and the conclusion explicitly says that it
applies only to the selected checks.

### What happens if only one check is enabled?

That check becomes the sole ranking factor. For example, with only step shift
enabled at its default weight of 3:

| Step-shift tier | Contribution | Maximum | Percentage | Classification |
|---|---:|---:|---:|---|
| Tier 3 | 0 | 6 | 0% | No concerns within selected checks |
| Tier 2 | 3 | 6 | 50% | Review recommended within selected checks |
| Tier 1 | 6 | 6 | 100% | Review recommended within selected checks |

The output is not presented as a complete station-quality verdict. It is a
`Focused Layer 1 assessment` and records which checks were enabled and disabled.

### What happens when an enabled check is not assessable?

An enabled but unassessable check is reported as `Not assessable` and excluded
from the score denominator. The output records:

- enabled check count;
- assessable check count;
- evidence coverage percentage;
- unavailable checks; and
- `assessment_incomplete = true`.

If none of the enabled checks can be assessed, the Layer 1 classification is
`Not assessable`.

### What happens when a weight is zero?

An enabled check with weight zero can still produce diagnostic evidence, but
it contributes nothing to the composite numerator or denominator. This differs
from disabling it: the check remains visible and assessable.

## 16. User-facing output

Layer 1 contributes the following types of information to TriHydrA outputs:

- the final class, raw score, normalized percentage, scope, and conclusion;
- enabled, disabled, assessable, and unavailable check lists;
- evidence coverage and incomplete-assessment status;
- compact metrics for every diagnostic;
- thresholds actually used, including station-specific thresholds;
- detailed evidence such as missing intervals, duplicates, plateau periods,
  spike/dip candidates, step boundaries, and epoch evidence; and
- an interactive overview showing only relevant findings, with missing periods
  shaded and structural/behavioural evidence overlaid on the raw record.

Clean checks may be omitted from the compact visual summary to reduce clutter;
their pass status and metrics remain available in structured results.

## 17. Interpretation cautions

- Tier 1 is the strongest concern; Tier 3 is the least concerning tier.
- A flag requests inspection and is not a declaration of erroneous data.
- Results are conditional on enabled checks and evidence availability.
- Daily sampling can hide sub-daily dynamics and can make real flash floods
  resemble isolated impulses.
- Source units affect tolerance- and magnitude-based rules.
- Regulation, measurement practice, rating changes, and real hydrological
  change can produce similar statistical patterns.
- Thresholds should be changed only with a documented rationale. TriHydrA
  stores the configuration used so the assessment remains reproducible.
- The default thresholds are screening defaults rather than globally calibrated
  limits. Large-sample testing has produced high trigger rates for some checks,
  particularly epoch drift, spikes/dips, low variability and step shifts.
  Users should examine the network-wide trigger summary before interpreting
  classifications or changing thresholds.

## 18. Scientific and technical references

The references below support the general quality-control and robust-statistical
ideas used by Layer 1. They do not imply that every TriHydrA numerical cutoff
was prescribed by the cited publication.

1. World Meteorological Organization. *Manual on Low-flow Estimation and
   Prediction* (WMO-No. 1029). The manual discusses examination of streamflow
   records for step changes, errors, outliers, and missing values.
   <https://old.wmo.int/extranet/pages/prog/hwrp/publications/low-flow_estimation_prediction/WMO%201029%20en.pdf>
2. World Meteorological Organization. Hydrology and Water Resources Programme:
   Data Management. WMO identifies error detection and quality control as core
   parts of hydrological data management.
   <https://community.wmo.int/site/knowledge-hub/programmes-and-initiatives/hydrology-and-water-resources/data-management>
3. Theil, H. (1950). *A rank-invariant method of linear and polynomial
   regression analysis*. The original rank-based regression work underlying
   the Theil–Sen slope.
   <https://ir.cwi.nl/pub/18446>
4. Sen, P. K. (1968). Estimates of the Regression Coefficient Based on
   Kendall's Tau. *Journal of the American Statistical Association*, 63(324),
   1379–1389. <https://doi.org/10.1080/01621459.1968.10480934>
5. Hampel, F. R. (1974). The Influence Curve and its Role in Robust
   Estimation. *Journal of the American Statistical Association*, 69(346),
   383–393. This is background for robust estimation; TriHydrA's exact
   median/MAD spike cutoff is its own implementation.
   <https://doi.org/10.1080/01621459.1974.10482962>

## 19. Relevant implementation files

- `trihydra/layer1/checks.py` – dispatches enabled checks.
- `trihydra/layer1/*.py` – individual diagnostic algorithms.
- `trihydra/composite.py` – tiering, weighting, normalization, and class.
- `trihydra/settings/defaults.py` – central defaults.
- `trihydra/settings/models.py` – configuration validation.
- `trihydra/layer1/diagnostics.py` – user-facing metrics and evidence.
- `trihydra/layer1/visualisation.py` – interactive Layer 1 overview.
- `trihydra.toml` – user-editable switches and thresholds.
