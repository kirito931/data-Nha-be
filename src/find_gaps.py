from pathlib import Path

import pandas as pd


INDEX_PATH = Path(
    "outputs/metadata/radar_index_2025-08.csv"
)


df = pd.read_csv(
    INDEX_PATH,
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp").reset_index(
    drop=True
)


df["delta_min"] = (
    df["timestamp"]
    .diff()
    .dt.total_seconds()
    / 60
)


# Những khoảng > 10 phút được coi là gap bất thường
gaps = df[df["delta_min"] > 10].copy()


print("=" * 70)
print("ABNORMAL TIME GAPS")
print("=" * 70)

if gaps.empty:

    print("No abnormal gaps found.")

else:

    for idx, row in gaps.iterrows():

        previous = df.loc[idx - 1]

        print()
        print(
            "Previous:",
            previous["timestamp"],
            previous["mode"],
            previous["filename"]
        )

        print(
            "Next    :",
            row["timestamp"],
            row["mode"],
            row["filename"]
        )

        print(
            "Gap     :",
            round(row["delta_min"], 2),
            "minutes"
        )