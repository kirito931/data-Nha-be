from pathlib import Path
import pandas as pd


INPUT = Path(
    "outputs/metadata/radar_index_2025-08-01.csv"
)

OUTPUT_DIR = Path("outputs/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = (
    OUTPUT_DIR /
    "temporal_groups_2025-08-01.csv"
)


df = pd.read_csv(
    INPUT,
    parse_dates=["timestamp"]
)

df = df.sort_values("timestamp").reset_index(drop=True)


groups = []

# Mỗi group kỳ vọng có:
# Long(t)
# Short(t+~3min)
# Long(t+~10min)
# Short(t+~13min)

i = 0
group_id = 0

while i + 3 < len(df):

    block = df.iloc[i:i+4].copy()

    modes = block["mode"].tolist()

    deltas = (
        block["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .iloc[1:]
        .tolist()
    )

    is_valid = (
        modes
        == [
            "Long Range",
            "Short Range",
            "Long Range",
            "Short Range"
        ]
        and
        2.7 <= deltas[0] <= 3.2
        and
        6.7 <= deltas[1] <= 7.3
        and
        2.7 <= deltas[2] <= 3.2
    )

    if is_valid:

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
        i += 4

    else:
        print(
            "Invalid block at row:",
            i
        )

        i += 1


groups_df = pd.DataFrame(groups)

groups_df.to_csv(
    OUTPUT,
    index=False
)

print("====================================")
print("TEMPORAL GROUP SUMMARY")
print("====================================")

print(
    "Number of groups:",
    len(groups_df)
)

print("\nFirst groups:")

print(
    groups_df.head(10)
    .to_string(index=False)
)

print("\nSaved to:")
print(OUTPUT)