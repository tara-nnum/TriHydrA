# Layer 3 — Network Context

## 1. Purpose

Layer 3 asks a different question from the first two layers:

> Does the target station's diagnosed behaviour agree with the behaviour of
> relevant gauges elsewhere in the supplied station network?

Layer 1 examines intrinsic data-quality evidence. Layer 2 describes
hydrological behaviour and compares two series when requested. Layer 3 does
not repeat either analysis. It reuses selected Layer 1 and Layer 2 results and
places them in spatial and catchment context.

Layer 3 is optional. It requires multiple input stations and a separate context
metadata table. Failure to assess Layer 3 does not prevent Layer 1 or Layer 2
from completing.

Layer 3 provides contextual evidence, not ground truth. Agreement with peers
does not prove that a record is correct, and disagreement does not prove that
it is wrong.

---

## 2. What Layer 3 can and cannot do

Layer 3 can:

- find nearby gauges using gauge coordinates;
- identify more distant catchments with compatible climate and catchment area;
- compare event dates, structural changes, long-term behaviour and selected
  Layer 2 signatures;
- preserve every target-to-peer metric used in a decision;
- report nearby-gauge, comparable-catchment and combined agreement;
- generate an interactive context dashboard when configured.

Layer 3 cannot:

- discover or download stations that were not loaded into the current run;
- infer missing coordinates, river names, catchment names or catchment areas;
- determine causality;
- replace rainfall, regulation, abstraction, reservoir or catchment-change
  information;
- certify that two catchments are hydrologically identical;
- make a Layer 1-style `Needs review` verdict.

Only stations whose time series are loaded in the current run can become
peers. A large context CSV alone is not enough: metadata does not provide the
discharge evidence needed for comparison.

---

## 3. Required inputs

### 3.1 Multiple station time series

At least the target and the configured minimum number of peers must be loaded.
The default Layer 3 `series_type` is `observation`, so the normal use is an
observed-gauge network.

The target series and every peer series independently pass through Layer 1 and
Layer 2. Layer 3 then adapts those completed results into a small evidence
record. It does not modify the original discharge series.

### 3.2 Context metadata CSV

The default file is `data/context.csv`. It must contain one unique row per
station and these exact columns:

| Column | Meaning | Validation |
|---|---|---|
| `station_id` | Identifier matching the loaded series | Required, non-empty and unique |
| `longitude` | Gauge longitude in decimal degrees | -180 to 180 |
| `latitude` | Gauge latitude in decimal degrees | -90 to 90 |
| `river_name` | River name | Required column; cleaned text |
| `catchment_name` | Catchment name | Required column; cleaned text |
| `catchment_area_km2` | Upstream catchment area | Must be greater than zero |
| `series_type` | Observation or simulation role | Must map to a supported alias |

Accepted `series_type` aliases are:

| Input value | Stored value |
|---|---|
| `obs`, `observed`, `observation` | `observation` |
| `sim`, `simulated`, `simulation`, `ml`, `model` | `simulation` |

Invalid metadata rows are rejected from Layer 3 and reported. Layer 1 and
Layer 2 may still run for those stations.

### 3.3 Climate resource

TriHydrA includes a categorical Köppen–Geiger raster for 1991–2020 and its
class legend. For each valid gauge coordinate, Layer 3 reads the class of the
raster cell containing that coordinate. It does not spatially interpolate the
categorical climate class.

The bundled legend contains the 30 Köppen–Geiger classes and the full class
description, for example `Aw — Tropical, savannah`. A user may supply an
alternative raster and legend through the configuration.

---

## 4. Metadata validation behaviour

The metadata reader:

1. reads the CSV;
2. confirms that all configured required columns exist;
3. trims station, river, catchment and series-type text;
4. normalises supported series-type aliases;
5. converts coordinates and catchment area to numeric values;
6. rejects invalid or duplicated rows;
7. retains the original CSV row number as `source_row`;
8. attaches the climate class at each accepted coordinate.

River and catchment matching is case-insensitive after surrounding whitespace
is removed. It remains an exact text comparison. Spelling variants or different
naming conventions are not reconciled automatically.

---

## 5. Two peer groups, two purposes

Layer 3 deliberately separates nearby gauges from comparable catchments.

### 5.1 Nearby gauges

Nearby gauges support comparisons tied to calendar dates, such as whether
high-flow peaks occurred at similar times. The default hard search radius is
50 km.

Eligible gauges are ranked in this order:

1. same river and same catchment;
2. same catchment;
3. same river;
4. any other gauge inside the radius.

Distance, catchment-area ratio and station ID break subsequent ties. River or
catchment membership never allows a gauge to bypass the radius.

Defaults:

| Setting | Default | Meaning |
|---|---:|---|
| `minimum_peers` | 1 | One nearby gauge is sufficient for assessment |
| `maximum_peers` | 5 | At most five nearby gauges are retained |
| `maximum_search_radius_km` | 50 | Hard great-circle distance limit |
| `prefer_same_catchment` | `true` | Intended preference switch; see limitations |
| `prefer_same_river` | `true` | Intended preference switch; see limitations |

Distances use the haversine great-circle formula and an Earth radius of
6371.0088 km. They are gauge-to-gauge distances, not river-network distances.

### 5.2 Comparable catchments

Comparable catchments provide broader hydrological context when a gauge is not
local but has compatible climate and catchment scale.

The default candidate must:

- not already have been selected as a nearby gauge;
- fall within 1000 km;
- have a catchment-area ratio no greater than 2.0;
- have the same Köppen–Geiger climate class.

The symmetric area ratio is:

```text
area ratio = max(peer area / target area, target area / peer area)
```

It is always at least 1. A ratio of 1 means equal areas; a ratio of 2 means one
catchment is twice the area of the other.

Candidates are ranked primarily by area ratio, then river/catchment
relationship, distance and station ID.

Defaults:

| Setting | Default | Meaning |
|---|---:|---|
| `minimum_peers` | 2 | At least two comparable catchments are required |
| `maximum_peers` | 5 | At most five are retained |
| `require_same_climate` | `true` | Require the same categorical climate class |
| `maximum_catchment_area_ratio` | 2.0 | Largest permitted symmetric area ratio |
| `maximum_search_radius_km` | 1000 | Safety limit for the search |

The 1000 km limit does not assert that distant gauges are locally
representative. Their contribution is separated from, and down-weighted
relative to, nearby-gauge evidence.

---

## 6. Evidence reused from Layers 1 and 2

Layer 3 stores only the evidence it needs:

| Evidence | Source |
|---|---|
| Record start and end | Valid values in the original series |
| High-flow peak dates | Layer 2 hydrograph-event catalogue |
| Meaningful step-shift dates | Layer 1 retained Tier 1 and Tier 2 boundaries |
| Dominant epoch behaviour | Layer 1 epoch diagnosis |
| Zero-flow ratio | Layer 1 zero-flow descriptor |
| Annual flashiness | Layer 2 annual signatures |
| Annual baseflow index | Layer 2 annual signatures |
| Monthly median seasonal profile | Layer 2 seasonality profile |
| Representative-event metrics | Layer 2 representative event |
| Representative-event discharge curve | Original values inside that event window |
| Layer 1 review class | Layer 1 composite assessment |

Tier 3 step-shift boundaries are excluded because Layer 1 classified their
magnitude as negligible. An executed check with zero detected peaks or shifts
is valid evidence, not missing evidence.

The representative event remains in the original discharge units. For event
shape comparison, both curves are resampled to 101 equally spaced relative
positions so events of different lengths can be compared point by point. Their
discharge magnitude is not normalised away.

---

## 7. Nearby-gauge timing and regime checks

### 7.1 High-flow peak timing

For each nearby peer:

1. restrict both peak-date catalogues to their common record period;
2. sort both lists of dates;
3. pair dates one-to-one when they fall within ±5 days by default;
4. calculate symmetric Dice agreement:

```text
agreement = 2 × matched pairs / (target peak count + peer peak count)
```

This formulation is symmetric: swapping target and peer gives the same score.
One event cannot be matched repeatedly.

The pair agrees when the score is at least 0.50 by default.

If neither station has peaks within the common period, the comparison is `not
applicable`; absence in both records is not automatically treated as perfect
agreement.

### 7.2 Step-shift timing

The same symmetric date-matching method is applied to meaningful Layer 1
step-shift boundaries. The default tolerance is ±50 days and the minimum
agreement is 0.50.

Only Tier 1 and Tier 2 boundaries enter this comparison. This check asks
whether persistent changes occurred at similar times; it does not compare the
direction or magnitude of the boundaries.

### 7.3 Epoch behaviour

The target and peer must have at least five overlapping years by default. The
check then compares the frozen dominant Layer 1 behaviour label, such as
`stable`, `rising` or `falling`.

```text
pair similarity = 1 when labels match, otherwise 0
```

The current check-level result uses the median of the pairwise 0/1 values. With
the default `peer_consensus_fraction = 0.50`, at least half of the assessed
pairwise evidence must support the target behaviour.

---

## 8. Hydrological-behaviour checks

These seven checks are run twice when evidence exists:

- against nearby gauges;
- against comparable catchments.

This makes local gauges useful for both event timing and hydrological
behaviour, rather than reserving behaviour checks only for distant catchments.

### 8.1 Median annual flashiness

The median annual Richards–Baker flashiness index is compared using a ratio:

```text
similarity = min(target, peer) / max(target, peer)
```

Both values are non-negative. Two zeros give similarity 1; only one zero gives
0. Default agreement threshold: 0.80.

### 8.2 Median annual baseflow index

The median annual Lyne–Hollick baseflow index is bounded between zero and one.
Similarity is:

```text
similarity = 1 - absolute(target - peer)
```

The result is clipped to 0–1. Default agreement threshold: 0.80.

### 8.3 Zero-flow behaviour

The valid-observation zero-flow ratio is compared with the same bounded
similarity:

```text
similarity = 1 - absolute(target ratio - peer ratio)
```

Default agreement threshold: 0.80.

### 8.4 Seasonality shape

The two 12-month median discharge profiles are aligned by calendar month.
At least six comparable finite months are required by default. Cosine
similarity is then calculated from the paired monthly values:

```text
cosine similarity = dot(target, peer) /
                    (norm(target) × norm(peer))
```

Two all-zero profiles agree only when both are effectively equal. Default
agreement threshold: 0.80.

Cosine similarity emphasises curve direction and shape. It is less sensitive
to a constant multiplicative magnitude difference than a direct error metric.

### 8.5 Representative event shape and magnitude

Each observed representative event is resampled to 101 positions across its
own duration. Two components are calculated:

```text
shape similarity = cosine similarity(resampled curves)

peak magnitude similarity = min(target peak, peer peak) /
                            max(target peak, peer peak)

combined event similarity = shape similarity × peak magnitude similarity
```

The multiplication prevents a similar outline from compensating for a very
different real-world peak magnitude. Default agreement threshold: 0.80.

The x-axis is relative event progression for the shape calculation, but the
actual event duration is compared separately in days.

### 8.6 Representative event time to peak

The absolute difference in time to peak is calculated in days. The pair agrees
when the difference is no more than 5 days by default.

For aggregation, the continuous similarity is:

```text
similarity = clip(1 - difference / (2 × tolerance), 0, 1)
```

Therefore the configured tolerance corresponds to similarity 0.50.

### 8.7 Representative event duration

The absolute duration difference is calculated in days. The pair agrees when
the difference is no more than 7 days by default. The same continuous formula
is used, so a 7-day difference corresponds to similarity 0.50.

---

## 9. From peer calculations to check results

Each target-to-peer row preserves:

- peer station ID;
- context group;
- target and peer values where applicable;
- matched and unmatched date counts where applicable;
- calculated similarity;
- threshold or day tolerance;
- `supported`, `not_supported`, `not_assessed` or `not_applicable` status;
- explanatory reason when assessment was impossible.

For an assessable check, the check-level continuous agreement is the median of
the assessed pairwise similarities. The check is supported when that median
meets its threshold.

The `supporting_peer_count` is reported for transparency, but the number of
available gauges is not itself a bonus in the final score. A station does not
receive a higher hydrological agreement merely because more gauges happened
to be available.

---

## 10. Context summaries

### 10.1 Nearby-gauge summary

Up to ten checks can contribute:

- high-flow peak timing;
- step-shift timing;
- epoch behaviour;
- median annual flashiness;
- median annual baseflow index;
- zero-flow behaviour;
- seasonality shape;
- representative event shape and magnitude;
- representative event time to peak;
- representative event duration.

Only checks with sufficient evidence enter the mean. Missing checks are not
treated as disagreement.

```text
nearby score (%) = 100 × mean(available check similarities)
```

Evidence coverage is also reported:

```text
coverage (%) = 100 × assessed checks / 10
```

### 10.2 Comparable-catchment summary

Seven behaviour checks can contribute. The three calendar-date/local-regime
checks do not use distant comparable catchments.

```text
comparable score (%) = 100 × mean(available behaviour similarities)
```

Coverage is the assessed fraction of those seven possible checks.

### 10.3 Combined contextual agreement

Shared behaviour checks combine nearby and comparable evidence using the
configured weights:

```text
combined check = nearby similarity × 0.70
               + comparable similarity × 0.30
```

The defaults deliberately give nearby gauges greater influence.

Special cases:

- when both groups exist, the 70/30 weights apply;
- when comparable evidence is absent but nearby evidence exists, the nearby
  evidence receives all available weight;
- when nearby evidence is absent but comparable evidence exists, the
  comparable contribution remains capped at its configured 30% influence;
- local-only timing/regime checks use their full own similarity;
- when neither side is assessable, that check is omitted.

The final combined score is the unweighted mean of all assessable combined
check values. The number of selected peers does not form a separate score.

### 10.4 Agreement categories

| Combined or group score | Classification |
|---:|---|
| 75% or more | Strong agreement |
| 40% to below 75% | Moderate agreement |
| Below 40% | Low agreement |
| No assessable checks | Not assessed |

These are TriHydrA interpretation bands, not universal hydrological laws.

Layer 3 uses the word **agreement**, not accuracy or correctness. A value such
as 80% means that the available configured comparisons produced strong
contextual agreement; it does not mean the station is “80% correct.”

---

## 11. Plotting policy

`layer3.plotting.mode` accepts:

| Mode | Behaviour |
|---|---|
| `none` | Do not create a Layer 3 HTML report |
| `all` | Show the full dashboard whenever Layer 3 is assessable |
| `recommended` | Gate the detailed dashboard using comparability and upstream review evidence |

In `recommended` mode:

1. if no combined agreement can be calculated, an insufficient-context page
   is returned;
2. if combined agreement is below 50% by default, a limited-comparability page
   is returned;
3. if Layer 1 did not classify the target as `Needs review`, a compact “no
   detailed context report required” page is returned;
4. otherwise the full interactive dashboard is produced.

Use `all` when visually inspecting Layer 3 itself, even for stations that did
not trigger Layer 1 review.

The full dashboard contains:

- segmented agreement bars;
- a recent common hydrograph with selectable traces;
- representative high-flow event comparison;
- selected peer information;
- evidence available through hover and supporting text outputs.

---

## 12. Output interpretation

The Layer 3 summary reports:

- assessment status;
- nearby-gauge and comparable-catchment counts;
- separate agreement categories and continuous internal values;
- combined agreement category;
- evidence coverage;
- contribution of each context group;
- whether the detailed context report is recommended.

Detailed evidence reports each target-versus-peer calculation on one line, for
example:

```text
Target vs PEER_A: similarity 0.910; rule >= 0.800 -> agrees.
Target vs PEER_B: similarity 0.630; rule >= 0.800 -> does not agree.
Result: 1/2 assessed peers agree.
```

Thresholds used by those comparisons are repeated in the report appendix and
stored with structured output evidence.

---

## 13. Configuration reference

### 13.1 Metadata

| Key | Default | Effect |
|---|---|---|
| `layer3.metadata.context_path` | `data/context.csv` | Context CSV location |
| `layer3.metadata.required_columns` | Seven columns listed above | Required schema |
| `layer3.metadata.climate_raster` | `None` | Use bundled raster unless overridden |
| `layer3.metadata.climate_legend` | `None` | Use bundled legend unless overridden |

### 13.2 Local peer selection

| Key | Default |
|---|---:|
| `layer3.local_peers.minimum_peers` | 1 |
| `layer3.local_peers.maximum_peers` | 5 |
| `layer3.local_peers.maximum_search_radius_km` | 50.0 |
| `layer3.local_peers.prefer_same_catchment` | `true` |
| `layer3.local_peers.prefer_same_river` | `true` |

### 13.3 Comparable-catchment selection

| Key | Default |
|---|---:|
| `layer3.analogue_peers.minimum_peers` | 2 |
| `layer3.analogue_peers.maximum_peers` | 5 |
| `layer3.analogue_peers.require_same_climate` | `true` |
| `layer3.analogue_peers.maximum_catchment_area_ratio` | 2.0 |
| `layer3.analogue_peers.maximum_search_radius_km` | 1000.0 |

### 13.4 Comparison and aggregation

| Key | Default | Used by |
|---|---:|---|
| `local_context_weight` | 0.70 | Shared-check combined score |
| `comparable_catchment_weight` | 0.30 | Shared-check combined score |
| `peak_tolerance_days` | 5 | Peak-date matching |
| `step_shift_tolerance_days` | 50 | Step-shift date matching |
| `minimum_peak_timing_similarity` | 0.50 | Peak timing decision |
| `minimum_step_shift_timing_similarity` | 0.50 | Step-shift timing decision |
| `minimum_epoch_overlap_years` | 5.0 | Epoch comparability |
| `analogue_similarity_minimum` | 0.80 | Scalar, seasonal and event-shape decisions |
| `minimum_profile_points` | 6 | Monthly shape calculation |
| `event_time_to_peak_tolerance_days` | 5.0 | Event timing decision |
| `event_duration_tolerance_days` | 7.0 | Event duration decision |
| `peer_consensus_fraction` | 0.50 | Epoch check-level decision |
| `similar_minimum_percent` | 75.0 | Strong-agreement band |
| `partial_minimum_percent` | 40.0 | Moderate-agreement band |
| `report_minimum_similarity_percent` | 50.0 | Recommended-plot gate |

The two context weights are normalised to sum to one. They must not both be
zero. Configuration validation also ensures ordered agreement bands, positive
search radii, sensible peer limits and bounded 0–1 similarity thresholds.

---

## 14. Turning Layer 3 off

Set:

```toml
[layers]
layer3 = false
```

Layer 1, Layer 2 and optional series comparison remain available. No Layer 3
penalty is added to their assessments because Layer 3 is contextual evidence,
not part of the Layer 1 or Layer 2 composite.

Turning off an upstream check can make the corresponding Layer 3 check
unavailable. The unavailable check is omitted from the Layer 3 mean and lowers
reported evidence coverage. It is not assigned zero agreement.

---

## 15. Known limitations

1. **Loaded-network limitation.** Only loaded time series can act as peers,
   even when the context CSV contains many more stations.
2. **Gauge distance is not river distance.** Haversine distance does not follow
   drainage connectivity or flow paths.
3. **Text relationships are exact.** River and catchment aliases are not
   reconciled.
4. **No precipitation context.** Similar climate and area do not guarantee
   similar storm forcing, geology, land use or regulation.
5. **Categorical climate lookup.** The raster cell at the gauge is used; it does
   not describe the full upstream catchment's climate distribution.
6. **Representative-event dependence.** Event comparisons inherit the Layer 2
   event-selection limitations and compare only the selected representative
   event.
7. **Event resampling.** Shape comparison standardises the number of points,
   which simplifies detailed timing within the event. Duration is therefore
   assessed separately in days.
8. **Step-shift context.** Timing is compared, not shift direction or
   magnitude.
9. **Threshold policy.** The distance limits, catchment-area ratio, tolerances,
   weights and agreement bands are configurable TriHydrA defaults. They have
   not been calibrated as universal thresholds for every hydroclimate.
10. **Preference switches.** The current implementation always uses same-river
    and same-catchment relationships in local ranking. Consequently,
    `prefer_same_river = false` and `prefer_same_catchment = false` do not yet
    disable those ranking preferences.

---

## 16. Scientific basis versus TriHydrA policy

The scientific basis for Layer 3 is that spatial proximity, flow-regime
similarity and physical/climatic catchment attributes can provide regional
hydrological context. Research on regionalisation also shows that proximity
and catchment attributes are imperfect proxies and that their performance is
application-dependent.

The following are methodological choices made by TriHydrA rather than values
claimed as universal in the literature:

- 50 km local radius;
- 1000 km comparable-catchment radius;
- 2.0 maximum area ratio;
- one and two minimum peers;
- ±5-day peak tolerance;
- ±50-day step-shift tolerance;
- five-year epoch overlap;
- 0.80 behaviour-similarity threshold;
- 70/30 local/comparable weights;
- 40% and 75% category boundaries;
- 50% detailed-report gate.

These settings should be sensitivity-tested for a new network or application.

---

## 17. References

- Beck, H. E., McVicar, T. R., Vergopolan, N., et al. (2023).
  *High-resolution (1 km) Köppen–Geiger maps for 1901–2099 based on
  constrained CMIP6 projections*. Scientific Data, 10, 724.
  https://doi.org/10.1038/s41597-023-02549-6
- Burn, D. H., & Boorman, D. B. (1993). *Estimation of hydrological
  parameters at ungauged catchments*. Journal of Hydrology, 143(3–4),
  429–454. https://doi.org/10.1016/0022-1694(93)90203-L
- Merz, R., & Blöschl, G. (2004). *Regionalisation of catchment model
  parameters*. Journal of Hydrology, 287(1–4), 95–123.
  https://doi.org/10.1016/j.jhydrol.2003.09.028
- Merz, R., & Blöschl, G. (2005). *Flood frequency regionalisation—spatial
  proximity vs. catchment attributes*. Journal of Hydrology, 302(1–4),
  283–306. https://doi.org/10.1016/j.jhydrol.2004.07.018

These references support the broad use of climatic, physiographic and spatial
context. They do not prescribe TriHydrA's exact operational thresholds.
