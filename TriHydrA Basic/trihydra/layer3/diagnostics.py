"""Human-readable Layer 3 context reporting."""

from __future__ import annotations

import pandas as pd

from trihydra.formatting import field, section, value


def _number(value_: object, decimals: int = 3) -> str:
    if value_ is None or pd.isna(value_):
        return "not available"
    return f"{float(value_):.{decimals}f}"


def _integer(value_: object) -> str:
    if value_ is None or pd.isna(value_):
        return "0"
    return str(int(value_))


def _decision(status: object) -> str:
    labels = {
        "supported": "agrees",
        "not_supported": "does not agree",
        "not_assessed": "not assessed",
        "not_applicable": "not applicable",
    }
    return labels.get(str(status), str(status).replace("_", " "))


def _context_label(value_: object) -> str:
    return {
        "nearby_gauge": "nearby gauge",
        "comparable_catchment": "comparable catchment",
    }.get(str(value_), str(value_).replace("_", " "))


def _peer_metric_line(row: pd.Series) -> str:
    """Describe one target-to-peer calculation on one readable line."""
    peer = str(row.get("peer_station_id", "unknown peer"))
    context = _context_label(row.get("context_group"))
    decision = _decision(row.get("status"))

    if pd.notna(row.get("symmetric_similarity")):
        matched = _integer(row.get("matched_date_pair_count"))
        target_count = _integer(row.get("target_date_count"))
        peer_count = _integer(row.get("peer_date_count"))
        return (
            f"    Target vs {peer} ({context}): agreement "
            f"{_number(row.get('symmetric_similarity'))}; matched {matched} date pair(s) "
            f"from {target_count} target and {peer_count} peer date(s); "
            f"rule >= {_number(row.get('minimum_match_fraction'))} within "
            f"+/-{_number(row.get('tolerance_days'), 0)} days -> {decision}."
        )

    if pd.notna(row.get("target_behaviour")) or pd.notna(row.get("peer_behaviour")):
        return (
            f"    Target vs {peer} ({context}): target {row.get('target_behaviour')}; "
            f"peer {row.get('peer_behaviour')}; overlap {_number(row.get('overlap_years'), 1)} years "
            f"(minimum {_number(row.get('minimum_overlap_years'), 1)}) -> {decision}."
        )

    similarity = row.get("similarity")
    threshold = row.get("minimum_similarity")
    parts = [f"    Target vs {peer} ({context})"]
    if pd.notna(row.get("target_value")) and pd.notna(row.get("peer_value")):
        parts.append(
            f"target {_number(row.get('target_value'))}; peer {_number(row.get('peer_value'))}"
        )
    if pd.notna(row.get("absolute_difference_days")):
        parts.append(f"difference {_number(row.get('absolute_difference_days'), 1)} days")
    if pd.notna(row.get("shape_similarity")):
        parts.append(f"shape {_number(row.get('shape_similarity'))}")
    if pd.notna(row.get("peak_magnitude_similarity")):
        parts.append(f"peak magnitude {_number(row.get('peak_magnitude_similarity'))}")
    if pd.notna(similarity):
        parts.append(f"similarity {_number(similarity)}")
    if pd.notna(threshold):
        parts.append(f"rule >= {_number(threshold)}")
    if pd.notna(row.get("tolerance_days")):
        parts.append(f"tolerance +/-{_number(row.get('tolerance_days'), 0)} days")
    if pd.notna(row.get("comparable_months")):
        parts.append(f"{_integer(row.get('comparable_months'))} comparable months")
    parts.append(decision)
    return "; ".join(parts) + "."


def _result_line(check_rows: pd.DataFrame) -> str:
    assessed = check_rows[check_rows["status"].isin(["supported", "not_supported"])]
    supported = int((assessed["status"] == "supported").sum())
    return f"    Result: {supported}/{len(assessed)} assessed peer(s) agree."


def render_layer3_summary(row: pd.Series, layer3: dict | None) -> list[str]:
    """Render every Layer 3 peer comparison and its final context result."""
    lines = section("LAYER 3  -  NETWORK CONTEXT")
    if row.get("layer3_status") != "assessed" or not layer3:
        return lines + ["  Not assessed - Layer 3 requires other gauges and station metadata."]
    lines += [
        field("Overall contextual agreement", value(row.get("layer3_context_agreement_class")), 2),
        field("Nearby-gauge agreement", value(row.get("layer3_nearby_gauge_agreement_class")), 2),
        field("Comparable-catchment agreement", value(row.get("layer3_comparable_catchment_agreement_class")), 2),
        field("Nearby gauges assessed", value(row.get("layer3_nearby_gauge_count")), 2),
        field("Comparable catchments assessed", value(row.get("layer3_comparable_catchment_count")), 2),
    ]
    evidence = layer3.get("evidence", pd.DataFrame())
    if not isinstance(evidence, pd.DataFrame) or evidence.empty:
        return lines
    details = evidence.loc[evidence["evidence_type"] == "peer_metric_comparison"].copy()
    if details.empty:
        return lines

    lines += [
        "",
        "  Pairwise context checks",
        "  Each line shows the value calculated before the peer decision was made.",
    ]
    for check_name, check_rows in details.groupby("check", sort=False):
        lines += ["", f"  {check_name}"]
        lines.extend(_peer_metric_line(peer) for _, peer in check_rows.iterrows())
        lines.append(_result_line(check_rows))
    return lines


def render_layer3_thresholds(layer3: dict | None) -> list[str]:
    """Render the Layer 3 rules used by the peer-level calculations."""
    if not layer3:
        return []
    evidence = layer3.get("evidence", pd.DataFrame())
    if not isinstance(evidence, pd.DataFrame) or evidence.empty:
        return []
    details = evidence.loc[evidence["evidence_type"] == "peer_metric_comparison"].copy()
    if details.empty:
        return []

    lines = section("LAYER 3 COMPARISON RULES")
    for check_name, rows in details.groupby("check", sort=False):
        assessed = rows.loc[rows["status"].isin(["supported", "not_supported"])]
        source = assessed.iloc[0] if not assessed.empty else rows.iloc[0]
        if pd.notna(source.get("minimum_match_fraction")):
            rule = (
                f"agreement >= {_number(source.get('minimum_match_fraction'))}; "
                f"date tolerance +/-{_number(source.get('tolerance_days'), 0)} days"
            )
        elif pd.notna(source.get("minimum_overlap_years")):
            rule = (
                f"same dominant behaviour with at least "
                f"{_number(source.get('minimum_overlap_years'), 1)} overlapping years"
            )
        elif pd.notna(source.get("minimum_similarity")):
            rule = f"similarity >= {_number(source.get('minimum_similarity'))}"
            if pd.notna(source.get("tolerance_days")):
                rule += f"; day tolerance +/-{_number(source.get('tolerance_days'), 0)}"
        else:
            rule = "no assessable threshold was recorded"
        lines.append(field(str(check_name), rule, 2))
    return lines


__all__ = ["render_layer3_summary", "render_layer3_thresholds"]
