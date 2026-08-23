# TriHydrA Layer 2: hydrological signatures and series comparison

## 1. Purpose

Layer 2 describes how a discharge series behaves hydrologically. It summarizes
annual flow magnitude, flashiness, baseflow contribution, seasonality,
persistence, and observed high-flow events.

Layer 2 has two distinct uses:

1. **One series:** calculate descriptive hydrological signatures. These values
   do not by themselves mean that the station is good or bad.
2. **Two series:** compare the same set of behaviours and produce a configurable
   Layer 2 comparison assessment.

This distinction matters. A flashy or highly seasonal river is not inherently
problematic. Concern arises only when two series expected to be comparable show
materially different behaviour under the enabled comparison rules.

Layer 2 never imputes, overwrites, or otherwise repairs the source record.
Calculations temporarily ignore unavailable values or operate on genuinely
consecutive valid segments. The original series is checked after processing to
confirm that it has not changed.

## 2. What Layer 2 contains

| Group | Information produced | Single-series role | Comparison role |
|---|---|---|---|
| Whole-record flow | Mean, median, minimum, maximum, Q05, Q95 | Descriptive | Flow-distribution similarity |
| Annual response | Richards–Baker flashiness | Descriptive | Annual-shape similarity |
| Annual response | Lyne–Hollick baseflow index | Descriptive | Annual-shape similarity |
| Annual response | Walsh–Lawler seasonality index | Descriptive | Reported, but not a separate scored component |
| Annual persistence | Lag-1 autocorrelation | Descriptive | Reported, but not scored |
| Seasonal profile | Monthly median climatology; wettest/driest month | Descriptive | Shape and timing similarity |
| High-flow events | Start, peak, end, magnitude, timing, duration, slopes | Descriptive | Timing, duration, and event-shape similarity |
| Layer 1 cross-check | Spike/dip relationship to high-flow events | Corroborative evidence | Not a Layer 2 comparison component |

The current Layer 2 comparison has **eight** configurable components. Their
default weights are equal, but each component can be disabled or reweighted.

## 3. Method and threshold provenance

TriHydrA separates three kinds of decisions:

1. **Published method.** The underlying signature comes from published work,
   such as the Richards–Baker flashiness index or Lyne–Hollick filter.
2. **TriHydrA comparison policy.** A configurable rule translates a similarity
   or difference into Tier 1, Tier 2, or Tier 3. These cutoffs are project
   defaults, not universally accepted hydrological constants.
3. **Computational requirement.** Minimum sample sizes and event boundaries
   are practical settings needed to make the calculation reproducible.

The software records the settings used. Users changing them should document why
the new values are appropriate for their sampling interval and application.

## 4. Input handling and missing values

Layer 2 expects a dated daily `pandas.Series` in one consistent discharge unit.

Before calculation, TriHydrA:

- makes a deep copy;
- converts values to numeric;
- sorts the dates;
- retains the first row where duplicate dates exist; and
- leaves missing values as missing.

Adjacent-day calculations use a pair only when both values exist and their
timestamps differ by exactly one day. Therefore, a gap is not silently treated
as one unusually large daily change.

### Annual and monthly evidence requirements

| TOML setting | Default | Meaning | Provenance |
|---|---:|---|---|
| `layer2.annual.minimum_valid_days_per_year` | 30 | Minimum valid daily values before a year is retained | Computational default |
| `layer2.annual.minimum_valid_days_per_month` | 10 | Minimum valid daily values before a month is retained | Computational default |

A retained year can still have fewer than 12 usable months. Annual flashiness,
baseflow, and autocorrelation may remain available, but seasonality and
wettest/driest month require all 12 months.

### Calendar convention

The current implementation groups values by **calendar year**. It does not infer
station-specific water years or hydrological years. This is a known design
choice and should be stated when seasonal timing spans a calendar boundary.

## 5. Annual and whole-record flow magnitude

For every retained year, Layer 2 calculates:

- arithmetic mean flow;
- median flow;
- minimum flow;
- maximum flow;
- raw minimum and maximum before Layer 1 candidate screening;
- number of valid days; and
- number of usable months.

The whole-record summary also reports mean, median, minimum, maximum, Q05, and
Q95.

### Flagged extrema

All Layer 1 spike/dip candidate timestamps are excluded when Layer 2 calculates
the displayed annual and whole-record minimum and maximum. The original extrema
are preserved separately as `raw_minimum_flow` and `raw_maximum_flow`.

This is deliberately conservative: a flagged point should not define a
hydrological extreme before it has been resolved. The point remains in the raw
record and in Layer 1 evidence.

### Percentile and flow-duration notation

TriHydrA uses percentile notation in variable names:

| TriHydrA value | Percentile meaning | Equivalent FDC exceedance name |
|---|---|---|
| Q05 | 5th percentile; low-flow reference | FDC Q95, exceeded about 95% of the time |
| Q95 | 95th percentile; high-flow reference | FDC Q05, exceeded about 5% of the time |

The dual terminology is documented because percentile and exceedance naming run
in opposite directions. The code stores descriptive names such as
`q05_percentile_low_flow_fdc_q95` to reduce ambiguity.

## 6. Richards–Baker flashiness index

### What it describes

Flashiness measures the magnitude of consecutive day-to-day discharge changes
relative to the total flow during the assessment period.

### Calculation

For valid consecutive daily pairs:

```text
R-B index = sum(|Q[t] - Q[t-1]|) / sum(Q[t])
```

The denominator uses all valid values in the year. If total flow is zero or
negative, or no valid daily pair exists, the result is unavailable.

### Interpretation

- Higher values indicate more rapid day-to-day variation relative to total
  flow.
- Lower values indicate a smoother hydrograph.
- The index is dimensionless.
- It is a hydrological descriptor, not an intrinsic data-quality flag.

The formula follows Baker et al. (2004). TriHydrA additionally prevents missing
calendar gaps from being interpreted as daily changes.

### Outputs

Annual values, whole-record median, P10–P90 diagnostic range, and valid annual
sample count.

## 7. Lyne–Hollick baseflow index

### What it describes

The baseflow index (BFI) estimates the proportion of discharge assigned to the
slow-flow component by a recursive digital filter.

### Calculation

TriHydrA applies the Lyne–Hollick quickflow recursion to each continuous,
non-negative valid segment:

```text
qf[t] = max(alpha × qf[t-1]
            + (1 + alpha) / 2 × (Q[t] - Q[t-1]), 0)
baseflow = clip(Q - quickflow, 0, Q)
BFI = sum(baseflow) / sum(Q)
```

Passes alternate direction: forward, backward, then forward for the default
three-pass configuration. Segments shorter than the configured minimum and
segments containing negative discharge are excluded.

### Configuration

| TOML setting | Default | Meaning | Provenance |
|---|---:|---|---|
| `baseflow_alpha` | 0.925 | Digital-filter parameter | Established Lyne–Hollick convention |
| `baseflow_passes` | 3 | Alternating filter passes | Computational/method choice |
| `minimum_baseflow_segment_days` | 3 | Shortest continuous segment used | TriHydrA computational default |

### Interpretation

- Values closer to 1 indicate a larger filtered slow-flow fraction.
- Values closer to 0 indicate a larger filtered quickflow fraction.
- BFI is dimensionless.
- The filtered component is an analytical separation, not a direct physical
  measurement of groundwater discharge.

### Outputs

Annual BFI values, whole-record median, P10–P90 diagnostic range, and sample
count.

## 8. Walsh–Lawler seasonality index

### What it describes

The index quantifies the strength of within-year contrast among monthly mean
flows. Walsh and Lawler introduced the method for rainfall; TriHydrA applies the
same mathematical form to discharge as a streamflow seasonality descriptor.

### Calculation

For a year with all 12 usable months:

```text
SI = sum(|monthly_mean[m] - annual_monthly_mean|) / sum(monthly_mean[m])
annual_monthly_mean = sum(monthly_mean[m]) / 12
```

If fewer than 12 months are usable or the total monthly mean flow is not
positive, the index is unavailable.

### Interpretation

- Zero indicates equal monthly mean flow throughout the year.
- Larger values indicate stronger seasonal concentration.
- The value does not identify which season is wet; timing is reported
  separately.
- The index is dimensionless.

### Outputs

Annual SI values, whole-record median, P10–P90 diagnostic range, and sample
count. Annual SI is descriptive and is not currently a separate Layer 2
comparison component.

## 9. Lag-1 autocorrelation

### What it describes

Lag-1 autocorrelation measures persistence between one valid daily discharge
value and the immediately preceding calendar day.

### Calculation

TriHydrA calculates the Pearson correlation between `Q[t]` and `Q[t-1]` using
only genuine consecutive valid daily pairs. At least three pairs are required.

### Interpretation

- Values near 1 indicate strong day-to-day persistence.
- Lower values indicate less persistence and potentially more rapid change.
- It complements flashiness but is not its inverse and is not scored in the
  current comparison composite.

### Outputs

Annual values, median, P10–P90 range, and sample count.

## 10. Monthly seasonal profile

For each usable month in each year, TriHydrA calculates monthly mean, monthly
median, and valid-day count. It then creates a typical 12-month profile by
taking the median of the annual monthly medians for each calendar month.

The typical wettest and driest months are the maximum and minimum of this
profile. Individual annual monthly cycles remain available as evidence and are
shown lightly in the interactive figure.

The typical wettest/driest markers should be interpreted cautiously for a very
flat or predominantly dry profile: a mathematical maximum can exist even when
the absolute difference among months is very small.

## 11. High-flow event detection

### Purpose

Layer 2 extracts coherent observed high-flow periods so event timing and shape
can be summarized without inventing a synthetic hydrograph.

### Threshold rule

The default method uses two whole-record percentiles:

| TOML setting | Default | Role | Provenance |
|---|---:|---|---|
| `layer2.events.trigger_percentile` | 0.95 | An event must reach Q95 | TriHydrA event policy |
| `layer2.events.boundary_percentile` | 0.90 | Contiguous event shoulders remain at or above Q90 | TriHydrA event policy |

Configuration validation requires:

```text
0 < boundary percentile < trigger percentile < 1
```

### Extraction algorithm

1. Calculate Q95 and Q90 from all valid values in the selected series.
2. Mark observations at or above Q90.
3. Split marked observations whenever dates are not consecutive or flow falls
   below Q90.
4. Retain a segment only if its maximum reaches Q95.
5. Use the maximum value as the event peak.

This is a threshold-event catalogue, not a rainfall–runoff separation model.
No precipitation or hyetograph is used.

### Per-event metrics

| Metric | Calculation |
|---|---|
| Event start | First date in the Q90-or-higher segment |
| Peak date | Date of maximum segment discharge |
| Event end | Last date in the segment |
| Peak flow | Maximum segment discharge |
| Time to peak | Peak date minus event start, in days |
| Recession duration | Event end minus peak date, in days |
| Total duration | Inclusive calendar length: end minus start plus one day |
| Rising slope | `(peak flow - start flow) / time to peak` |
| Recession slope | `(end flow - peak flow) / recession days` |

If the peak occurs on the first day, rising slope is unavailable. If it occurs
on the last day, recession slope is unavailable.

### Known limitation

Q95/Q90 is deliberately simple and reproducible, but it is not guaranteed to
identify every hydrometeorological flood event. Event counts and shapes depend
on record distribution, sampling frequency, regulation, missing values, and
the chosen percentiles.

## 12. Layer 1 spike/dip and Layer 2 event cross-check

Layer 1 candidates are propagated into Layer 2. This dependency serves two
purposes:

1. prevent an unresolved candidate from defining displayed extrema; and
2. assess whether a spike lies within coherent high-flow-event context.

### Cross-check outcomes

| Situation | Status | Meaning |
|---|---|---|
| Dip candidate | `retained_for_review` | A high-flow event cannot directly explain a dip |
| Spike outside every event | `retained_for_review` | No coherent event context found |
| Spike is the event peak | `spike_peak_overlap_review` | Peak may be artefactual; manual inspection remains necessary |
| Spike in an event shorter than minimum duration | `retained_for_review` | Candidate may have created the apparent event |
| Spike inside a sufficiently long event but not at its peak | `plausible_event_context` | Event context reduces concern but does not prove validity |

### Configuration

| TOML setting | Default | Meaning | Provenance |
|---|---:|---|---|
| `spike_crosscheck_minimum_event_duration_days` | 3.0 | Minimum duration before surrounding event context can reduce concern | TriHydrA policy |

The value must be at least one day.

### Effect on representative events

If a Layer 1 spike candidate coincides with an event peak, that event remains in
the event catalogue but receives:

- `layer1_spike_peak_overlap = true`;
- `representative_eligible = false`; and
- an explicit exclusion reason.

Thus, flagged peaks are visible but cannot be selected as the representative
event.

## 13. Representative observed event

TriHydrA selects one **real detected event**, not an average or constructed
curve.

Among eligible events, it evaluates:

- peak flow;
- total duration;
- time to peak;
- rising slope; and
- recession slope.

For every metric, values are centered on the median and scaled by median
absolute deviation. If that scale is zero, standard deviation is used; if that
also fails, scale 1 is used. The event with the smallest mean squared scaled
distance from the multimetric median is selected.

The representative event is therefore the observed event most typical across
the available metrics. It is not necessarily the largest flood.

## 14. Descriptive diagnostic summary

For each annual signature and event property, the diagnostic table reports:

- median;
- 10th percentile;
- 90th percentile; and
- sample count.

The P10–P90 interval describes observed variability. It is not a confidence
interval and is not used as a comparison tier.

## 15. Two-series comparison support

### Standalone versus paired calculations

Each series receives standalone Layer 1 and native Layer 2 diagnostics over its
own delivered record.

For the default paired comparison, both Layer 2 comparison calculations use the
same common calendar and pairwise-valid support. This avoids attributing a
simulation-only tail or non-overlapping period to model–observation differences.
Layer 1 is not truncated to that common window.

For independent-timespan comparison, each selected timespan is assessed on its
own support. This mode is intended for comparisons such as a recent period
against a historical baseline.

Both series must use the same physical units before comparison.

## 16. The eight Layer 2 comparison components

| Component | Metric | Default tier rule | Default weight |
|---|---|---|---:|
| Flow behaviour | Inverse Jensen–Shannon distance between discharge histograms | similarity rule | 1 |
| Annual flashiness shape | Cosine similarity of annual flashiness series | similarity rule | 1 |
| Annual baseflow shape | Cosine similarity of annual BFI series | similarity rule | 1 |
| Seasonal profile shape | Cosine similarity of the 12 monthly medians | similarity rule | 1 |
| Seasonal timing | Maximum circular separation of wettest and driest months | month rule | 1 |
| Event time to peak | Absolute difference between median event times to peak | day rule | 1 |
| Event duration | Absolute difference between median event durations | day rule | 1 |
| Representative event shape | Cosine similarity after resampling both real events to 100 relative-time points | similarity rule | 1 |

Median bias is also reported for annual flashiness, annual BFI, and the seasonal
profile, but it is not separately tiered in the current composite.

### Symmetry

The implemented metrics are symmetric with respect to series order:

- Jensen–Shannon divergence is symmetric;
- cosine similarity is symmetric;
- timing and duration use absolute differences; and
- seasonal month distance is circular and absolute.

Changing which series is labelled reference or candidate can change the sign of
reported median bias, but should not change the comparison tier itself.

## 17. Similarity metrics

### Inverse Jensen–Shannon distance

TriHydrA builds common automatically selected histogram bins from the combined
valid values. A small numerical constant prevents zero-probability division.

```text
JSD = 0.5 × KL(P || M) + 0.5 × KL(Q || M)
M = 0.5 × (P + Q)
inverse JSD = 1 - sqrt(JSD)
```

Base-2 logarithms make the square-root distance bounded from 0 to 1. TriHydrA
then inverts it so all similarity metrics have the same direction:

- 1 means matching distributions;
- 0 means maximally dissimilar distributions under this formulation.

The name **Inverse JSD** refers to this TriHydrA transformation, not to a
separate published divergence.

### Cosine similarity

```text
cosine similarity = dot(A, B) / (norm(A) × norm(B))
```

Only positions valid in both arrays are used, and at least two paired values are
required. Annual series are aligned by calendar year where at least two common
years exist. Otherwise, each annual sequence is resampled to 100 relative-time
positions before comparison.

The representative-event comparison resamples time but does **not** normalize
discharge by the event peak. Cosine similarity is nevertheless invariant to a
single positive scaling factor, so it primarily evaluates curve direction and
shape rather than absolute magnitude. Peak discharge remains visible in the
descriptive outputs but is not a separately scored comparison component.

## 18. Component tier rules

Tier 3 is the closest agreement and Tier 1 the strongest disagreement.

### Similarity components

Used by inverse JSD and cosine similarity:

| Tier | Default rule |
|---|---|
| Tier 3 | Similarity >= 0.80 |
| Tier 2 | 0.50 <= similarity < 0.80 |
| Tier 1 | Similarity < 0.50 |

### Seasonal timing

Wettest and driest month differences use circular distance, so December and
January are one month apart. The larger of the wettest-month and driest-month
separations is scored.

| Tier | Default rule |
|---|---|
| Tier 3 | Difference <= 1 month |
| Tier 2 | Difference > 1 and <= 3 months |
| Tier 1 | Difference > 3 months |

### Median event time to peak

| Tier | Default rule |
|---|---|
| Tier 3 | Absolute difference <= 3 days |
| Tier 2 | Difference > 3 and < 5 days |
| Tier 1 | Difference >= 5 days |

### Median event duration

| Tier | Default rule |
|---|---|
| Tier 3 | Absolute difference <= 3 days |
| Tier 2 | Difference > 3 and <= 7 days |
| Tier 1 | Difference > 7 days |

All of these numerical boundaries are configurable TriHydrA policies.

## 19. Composite calculation

Default tier points are:

```text
Tier 3 = 0 points
Tier 2 = 1 point
Tier 1 = 2 points
```

For every enabled, assessable component:

```text
contribution = tier points × component weight
raw score = sum(contributions)
maximum assessable score = sum(2 × weight)
score percent = 100 × raw score / maximum assessable score
```

The default class is:

| Class | Default normalized score |
|---|---|
| Similar | <= 12.5% |
| Review | > 12.5% and <= 43.75% |
| Strong review | > 43.75% |

These boundaries were chosen to summarize the eight equal-weight components.
They are review-policy defaults rather than literature-defined categories.

## 20. Disabled and unavailable comparison components

### Disabled component

Setting a component to `false` under `[layer2.comparison.components]` removes it
from the comparison. It contributes neither points nor denominator weight.

### Weight zero

An enabled component with weight zero remains in evidence but contributes
nothing to the numerator or denominator.

### Unavailable component

A component is unavailable when required evidence cannot be calculated, for
example too few events or no usable seasonal profile. Unavailable components do
not increase the denominator.

### Minimum evidence

`minimum_assessable_components = 4` by default. The effective requirement is
the smaller of this value and the number of enabled components. Therefore:

- with all eight components enabled, at least four must be assessable;
- with three components enabled, all three must be assessable;
- with one component enabled, that one component can determine the result.

If the requirement is not met, the result is `Not assessable` rather than an
artificially reassuring score.

Configuration validation requires at least one component when comparison is
enabled and at least one enabled component with positive weight.

## 21. User-facing outputs

### Single-series summary

Layer 2 exposes:

- whole-record flow statistics and Q05/Q95 references;
- median annual flashiness, BFI, seasonality, and lag-1 autocorrelation;
- typical wettest and driest months;
- annual signature year count;
- event count;
- representative-event dates and metrics;
- Layer 1 candidates excluded from extrema;
- spike/peak overlap count;
- thresholds actually used; and
- detailed annual, monthly, event, and cross-check evidence.

### Comparison summary

The comparison exposes:

- raw and normalized score;
- maximum assessable score;
- class;
- number of assessable components;
- incomplete-assessment status;
- unavailable component names;
- each component metric, value, tier, weight, and contribution; and
- median biases and additional diagnostic comparisons.

### Interactive figures

The Layer 2 overview includes annual flow behaviour, annual response indices,
the real representative event, historical seasonality, diagnostic medians, and
the Layer 1 spike/dip–Layer 2 peak cross-check. Pairwise output overlays the two
series so differences can be inspected directly.

## 22. Interpretation cautions

- Layer 2 signatures are descriptors; unusual hydrology is not automatically
  erroneous data.
- Results depend on temporal resolution and units. Current event-day settings
  assume daily series.
- Calendar-year aggregation may split an event or season across years.
- Missing data can reduce usable years, months, pairs, and events without being
  imputed.
- BFI is filter-dependent and should not be interpreted as directly measured
  groundwater flow.
- Walsh–Lawler SI was developed for rainfall and is used here as an adapted
  discharge-seasonality index.
- Q95/Q90 events are threshold-defined high-flow periods, not verified floods.
- A representative event is typical under five metrics and need not be the
  largest or most damaging event.
- Cosine similarity can remain high when curves share direction but differ in
  absolute magnitude; supporting biases and plots should also be inspected.
- Comparison thresholds should be calibrated or justified for a new domain.
- `Review` and `Strong review` request inspection; they do not prove that either
  series is wrong.

## 23. Scientific and technical references

The references support the established methods. TriHydrA's event percentiles,
comparison cutoffs, weights, and final classes remain its own configurable
review policy.

1. Baker, D. B., Richards, R. P., Loftus, T. T., & Kramer, J. W. (2004).
   A New Flashiness Index: Characteristics and Applications to Midwestern
   Rivers and Streams. *JAWRA*, 40(2), 503–522.
   <https://doi.org/10.1111/j.1752-1688.2004.tb01046.x>
2. Lyne, V., & Hollick, M. (1979). *Stochastic Time-Variable Rainfall–Runoff
   Modelling*. Hydrology and Water Resources Symposium, Perth, pp. 89–92.
   The work introduced the recursive digital filter used as the basis of the
   Lyne–Hollick separation.
3. Nathan, R. J., & McMahon, T. A. (1990). Evaluation of automated techniques
   for base flow and recession analyses. *Water Resources Research*, 26(7),
   1465–1473. <https://doi.org/10.1029/WR026i007p01465>
4. Walsh, R. P. D., & Lawler, D. M. (1981). Rainfall seasonality: description,
   spatial patterns and change through time. *Weather*, 36(7), 201–208.
   <https://doi.org/10.1002/j.1477-8696.1981.tb05400.x>
5. Lin, J. (1991). Divergence measures based on the Shannon entropy.
   *IEEE Transactions on Information Theory*, 37(1), 145–151.
   <https://doi.org/10.1109/18.61115>

## 24. Relevant implementation files

- `trihydra/layer2/annual_signatures.py` – annual statistics, flashiness, BFI,
  seasonality, autocorrelation, and monthly profile.
- `trihydra/layer2/hydrograph_information.py` – high-flow event extraction and
  representative-event selection.
- `trihydra/layer2/peak_outlier_crosscheck.py` – Layer 1/Layer 2 corroboration
  and representative-event eligibility.
- `trihydra/layer2/diagnostics.py` – orchestration, summaries, thresholds, and
  evidence tables.
- `trihydra/layer2/visualisation.py` – interactive Layer 2 figures.
- `trihydra/composite.py` – pairwise metrics, tiering, weights, and composite.
- `trihydra/comparison/calculations.py` – native and comparison-support runs.
- `trihydra/settings/defaults.py` – central defaults.
- `trihydra/settings/models.py` – configuration validation.
- `trihydra.toml` – user-editable Layer 2 settings.
