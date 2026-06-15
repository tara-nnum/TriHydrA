import pandas as pd

from src.trihydra.layer1.checks import (
    check_missing_values,
    check_long_gaps,
    check_negative_discharge,
    check_duplicate_timestamps,
    check_timestep_consistency,
    check_low_variability_flow,
    check_zero_flow_regime,
    check_single_point_spike_dip,
    check_step_shift,
    check_gradual_drift,
)


# Load fake test data
file_path = r"D:\Workplace\ECMWF Code for Earth Challenge\Coding Phase\TriHydrA\data\trihydra_fake_daily_timeseries_with_anomalies.csv"

df = pd.read_csv(file_path)

df["date"] = pd.to_datetime(df["date"])

df = df.set_index("date")

print("\nInjected anomaly counts:\n")
print(df["injected_anomaly"].value_counts(dropna=False))

q = df["discharge_m3s"]

# Run Layer 1 basic checks
results = [
    check_missing_values(q),
    check_long_gaps(q),
    check_negative_discharge(q),
    check_duplicate_timestamps(q),
    check_timestep_consistency(q),
    check_low_variability_flow(q),
    check_zero_flow_regime(q),
    check_single_point_spike_dip(q),
    check_step_shift(q),
    check_gradual_drift(q),
]

# Print results nicely
print("\nLayer 1 basic check results:\n")

for result in results:
    print(f"{result['check']}")
    print(f"  flag      : {result['flag']}")
    print(f"  value     : {result['value']}")
    print(f"  threshold : {result['threshold']}")
    print(f"  timestamps: {result['flagged_timestamps']}")
    print(f"  message   : {result['message']}")
    print()