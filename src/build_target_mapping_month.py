from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

GROUPS_PATH = Path(
    "outputs/metadata/temporal_groups_2025-08.csv"
)

INDEX_PATH = Path(
    "outputs/metadata/radar_index_2025-08.csv"
)

OUTPUT_PATH = Path(
    "outputs/metadata/target_mapping_2025-08.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

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

index = (
    index
    .sort_values("timestamp")
    .reset_index(drop=True)
)


print("=" * 70)
print("BUILDING FUTURE TARGET MAPPING")
print("=" * 70)

print(
    "Temporal groups:",
    len(groups)
)

print(
    "Radar scans:",
    len(index)
)


# ============================================================
# FIND NEAREST RADAR SCAN
# ============================================================

MAX_DIFF_SECONDS = 60


def find_nearest_scan(target_time):

    differences = (
        index["timestamp"] - target_time
    ).abs()

    nearest_idx = differences.idxmin()

    nearest = index.loc[nearest_idx]

    diff_seconds = (
        differences.loc[nearest_idx]
        .total_seconds()
    )

    if diff_seconds <= MAX_DIFF_SECONDS:

        return {
            "timestamp": nearest["timestamp"],
            "filename": nearest["filename"],
            "mode": nearest["mode"],
            "diff_seconds": diff_seconds,
        }

    return {
        "timestamp": None,
        "filename": None,
        "mode": None,
        "diff_seconds": diff_seconds,
    }


# ============================================================
# BUILD MAPPING
# ============================================================

rows = []


for _, group in groups.iterrows():

    input_time = group["t0"]

    row = {
        "group_id": group["group_id"],
        "input_time": input_time,

        "input_file_t0": group["file_t0"],
        "input_file_t3": group["file_t3"],
        "input_file_t10": group["file_t10"],
        "input_file_t13": group["file_t13"],
    }


    # --------------------------------------------------------
    # 0h, 1h, 2h, 3h
    # --------------------------------------------------------

    for horizon in [0, 1, 2, 3]:

        target_time = (
            input_time
            + pd.Timedelta(hours=horizon)
        )

        result = find_nearest_scan(
            target_time
        )

        suffix = f"{horizon}h"

        row[
            f"target_{suffix}_expected"
        ] = target_time

        row[
            f"target_{suffix}_actual"
        ] = result["timestamp"]

        row[
            f"target_{suffix}_file"
        ] = result["filename"]

        row[
            f"target_{suffix}_mode"
        ] = result["mode"]

        row[
            f"target_{suffix}_diff_sec"
        ] = result["diff_seconds"]


    rows.append(row)


result = pd.DataFrame(rows)


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("TARGET MAPPING COMPLETE")
print("=" * 70)

print(
    "Total groups:",
    len(result)
)


for horizon in [0, 1, 2, 3]:

    suffix = f"{horizon}h"

    col = f"target_{suffix}_file"

    available = result[col].notna().sum()

    missing = result[col].isna().sum()

    print()
    print(
        f"+{horizon}h:"
    )

    print(
        "  Available:",
        available
    )

    print(
        "  Missing  :",
        missing
    )


print()
print("First 5 mappings:")

print(
    result.head(5)
    .to_string(index=False)
)


print()
print("Last 5 mappings:")

print(
    result.tail(5)
    .to_string(index=False)
)


print()
print("Saved:")
print(OUTPUT_PATH)