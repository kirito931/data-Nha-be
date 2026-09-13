from pathlib import Path
import pandas as pd


INPUT = Path(
    "outputs/metadata/radar_index_2025-08.csv"
)

OUTPUT = Path(
    "outputs/metadata/temporal_groups_2025-08.csv"
)


df = pd.read_csv(
    INPUT,
    parse_dates=["timestamp"]
)

df = (
    df.sort_values("timestamp")
      .reset_index(drop=True)
)


groups = []

i = 0
group_id = 0

while i + 3 < len(df):

    block = df.iloc[i:i + 4]

    modes = block["mode"].tolist()

    deltas = (
        block["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .iloc[1:]
        .tolist()
    )

    valid_modes = (
        modes
        == [
            "Long Range",
            "Short Range",
            "Long Range",
            "Short Range"
        ]
    )

    valid_time = (
        2.7 <= deltas[0] <= 3.2
        and
        6.7 <= deltas[1] <= 7.3
        and
        2.7 <= deltas[2] <= 3.2
    )

    if valid_modes and valid_time:

        groups.append({
            "group_id": group_id,

            "t0": block.iloc[0]["timestamp"],
            "file_t0": block.iloc[0]["filename"],

            "t3": block.iloc[1]["timestamp"],
            "file_t3": block.iloc[1]["filename"],

            "t10": block.iloc[2]["timestamp"],
            "file_t10": block.iloc[2]["filename"],

            "t13": block.iloc[3]["timestamp"],
            "file_t13": block.iloc[3]["filename"],
        })

        group_id += 1

        # Một group đã dùng 4 scan
        i += 4

    else:
        # Không hợp lệ → dịch 1 scan để tìm
        # sequence hợp lệ tiếp theo
        i += 1


groups_df = pd.DataFrame(groups)

groups_df.to_csv(
    OUTPUT,
    index=False
)


print("=" * 70)
print("TEMPORAL GROUPING COMPLETE")
print("=" * 70)

print(
    "Number of valid groups:",
    len(groups_df)
)

print()

print("First 10 groups:")

print(
    groups_df.head(10)
    .to_string(index=False)
)

print()

print("Last 10 groups:")

print(
    groups_df.tail(10)
    .to_string(index=False)
)

print()

print("Saved:")
print(OUTPUT)