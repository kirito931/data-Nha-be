from pathlib import Path

import pandas as pd


GROUPS_PATH = Path(
    "outputs/metadata/temporal_groups_2025-08-01.csv"
)

INDEX_PATH = Path(
    "outputs/metadata/radar_index_2025-08-01.csv"
)

OUTPUT_PATH = Path(
    "outputs/metadata/target_mapping_2025-08-01.csv"
)


groups = pd.read_csv(
    GROUPS_PATH,
    parse_dates=[
        "t0",
        "t3",
        "t10",
        "t13"
    ]
)

index = pd.read_csv(
    INDEX_PATH,
    parse_dates=["timestamp"]
)

index = index.sort_values("timestamp").reset_index(
    drop=True
)


# Dùng timestamp của group bắt đầu làm reference.
# Tìm scan gần nhất cho +0h, +1h, +2h, +3h.

rows = []

MAX_DIFF_SECONDS = 60


for _, group in groups.iterrows():

    base_time = group["t0"]

    row = {
        "group_id": group["group_id"],
        "input_time": base_time,
    }

    for horizon in [0, 1, 2, 3]:

        target_time = (
            base_time
            + pd.Timedelta(hours=horizon)
        )

        differences = (
            index["timestamp"] - target_time
        ).abs()

        nearest_idx = differences.idxmin()

        nearest = index.loc[nearest_idx]

        diff_seconds = (
            differences.loc[nearest_idx]
            .total_seconds()
        )

        suffix = f"{horizon}h"

        if diff_seconds <= MAX_DIFF_SECONDS:

            row[f"target_{suffix}"] = (
                nearest["timestamp"]
            )

            row[f"target_{suffix}_file"] = (
                nearest["filename"]
            )

            row[f"target_{suffix}_mode"] = (
                nearest["mode"]
            )

            row[f"target_{suffix}_diff_sec"] = (
                diff_seconds
            )

        else:

            row[f"target_{suffix}"] = None

            row[f"target_{suffix}_file"] = None

            row[f"target_{suffix}_mode"] = None

            row[f"target_{suffix}_diff_sec"] = (
                diff_seconds
            )

    rows.append(row)


result = pd.DataFrame(rows)

result.to_csv(
    OUTPUT_PATH,
    index=False
)


print(
    "Saved:",
    OUTPUT_PATH
)

print(
    "\nFirst 10 mappings:"
)

print(
    result.head(10)
    .to_string(index=False)
)