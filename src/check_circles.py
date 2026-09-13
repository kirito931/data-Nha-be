from pathlib import Path
import pandas as pd


index_path = Path(
    "outputs/metadata/radar_index_2025-08-01.csv"
)

df = pd.read_csv(index_path)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

df["delta_min"] = (
    df["timestamp"]
    .diff()
    .dt.total_seconds()
    / 60
)

print("========== TIMESTAMP INTERVALS ==========")

print(
    df[
        [
            "timestamp",
            "mode",
            "delta_min"
        ]
    ].head(50).to_string(index=False)
)


print("\n========== INTERVAL DISTRIBUTION ==========")

print(
    df["delta_min"]
    .round(2)
    .value_counts()
    .sort_index()
)


print("\n========== MODE SEQUENCE ==========")

print(
    df[
        ["timestamp", "mode"]
    ].head(40).to_string(index=False)
)