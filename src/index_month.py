from pathlib import Path
from datetime import datetime

import pandas as pd
import pyart


# ============================================================
# 1. ĐƯỜNG DẪN
# ============================================================

DATA_ROOT = Path(r"2025 Pro-Raw T08")

OUTPUT_DIR = Path("outputs/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PATH = OUTPUT_DIR / "radar_index_2025-08.csv"


# ============================================================
# 2. CÁC HÀM HỖ TRỢ
# ============================================================

def classify_scan_mode(radar):
    """
    Xác định loại scan dựa trên số sweep và khoảng cách giữa gates.

    Long Range:
        - 4 sweeps
        - ~600 m/gate

    Short Range:
        - 8 sweeps
        - ~240 m/gate

    Nếu không khớp thì trả về Unknown.
    """

    ranges = radar.range["data"]

    if len(ranges) < 2:
        return "Unknown", None, None

    gate_spacing = float(ranges[1] - ranges[0])
    max_range_km = float(ranges[-1] / 1000.0)

    if (
        radar.nsweeps == 4
        and abs(gate_spacing - 600.0) < 1.0
    ):
        return "Long Range", gate_spacing, max_range_km

    if (
        radar.nsweeps == 8
        and abs(gate_spacing - 240.0) < 1.0
    ):
        return "Short Range", gate_spacing, max_range_km

    return "Unknown", gate_spacing, max_range_km


def parse_timestamp(filename):
    """
    Đọc timestamp từ tên file:

    NHB250801000007.RAWLTHU
       ^^^^^^^^^^^^
       YYMMDDHHMMSS
    """

    timestamp_text = filename[3:15]

    return datetime.strptime(
        timestamp_text,
        "%y%m%d%H%M%S"
    )


# ============================================================
# 3. TÌM TOÀN BỘ FILE RAW TRONG 31 NGÀY
# ============================================================

all_files = []

print("=" * 70)
print("SCANNING DATASET")
print("=" * 70)

for day_dir in sorted(DATA_ROOT.iterdir()):

    if not day_dir.is_dir():
        continue

    day_files = sorted(
        day_dir.glob("NHB*.RAW*")
    )

    print(
        f"Day {day_dir.name:>2}: "
        f"{len(day_files)} files"
    )

    all_files.extend(day_files)


print()
print("Total files found:", len(all_files))


# ============================================================
# 4. ĐỌC METADATA TỪNG FILE
# ============================================================

rows = []

errors = []

print()
print("=" * 70)
print("INDEXING RADAR FILES")
print("=" * 70)


for i, file_path in enumerate(
    all_files,
    start=1
):

    print(
        f"[{i}/{len(all_files)}] "
        f"{file_path.parent.name}/"
        f"{file_path.name}"
    )

    try:

        # ----------------------------------------------------
        # Đọc radar metadata bằng Py-ART
        # ----------------------------------------------------

        radar = pyart.io.read_sigmet(
            str(file_path)
        )

        # ----------------------------------------------------
        # Timestamp
        # ----------------------------------------------------

        timestamp = parse_timestamp(
            file_path.name
        )

        # ----------------------------------------------------
        # Scan configuration
        # ----------------------------------------------------

        mode, gate_spacing, max_range_km = (
            classify_scan_mode(radar)
        )

        # ----------------------------------------------------
        # Reflectivity shape
        # ----------------------------------------------------

        reflectivity = radar.fields[
            "reflectivity"
        ]["data"]

        # ----------------------------------------------------
        # Row metadata
        # ----------------------------------------------------

        rows.append({

            "timestamp": timestamp,

            "day": int(file_path.parent.name),

            "filename": file_path.name,

            # Đường dẫn tương đối so với DATA_ROOT
            "relative_path": str(
                file_path.relative_to(DATA_ROOT)
            ),

            "mode": mode,

            "scan_type": radar.scan_type,

            "sweeps": radar.nsweeps,

            "rays": radar.nrays,

            "gates": radar.ngates,

            "gate_spacing_m": gate_spacing,

            "max_range_km": max_range_km,

            "reflectivity_shape": str(
                reflectivity.shape
            ),

        })

    except Exception as e:

        print(
            f"    ERROR: {type(e).__name__}: {e}"
        )

        errors.append({

            "filename": file_path.name,

            "relative_path": str(
                file_path.relative_to(DATA_ROOT)
            ),

            "error_type": type(e).__name__,

            "error_message": str(e),

        })


# ============================================================
# 5. TẠO DATAFRAME
# ============================================================

df = pd.DataFrame(rows)

if not df.empty:

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)


# ============================================================
# 6. LƯU INDEX TOÀN THÁNG
# ============================================================

df.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# 7. LƯU ERROR LOG NẾU CÓ
# ============================================================

if errors:

    errors_df = pd.DataFrame(errors)

    error_path = (
        OUTPUT_DIR
        / "radar_index_errors_2025-08.csv"
    )

    errors_df.to_csv(
        error_path,
        index=False
    )

else:

    error_path = None


# ============================================================
# 8. BÁO CÁO TỔNG QUAN
# ============================================================

print()
print("=" * 70)
print("INDEXING COMPLETE")
print("=" * 70)

print(
    "Successfully indexed:",
    len(df)
)

print(
    "Files with errors:",
    len(errors)
)

print()
print("---------- MODE COUNT ----------")

if not df.empty:
    print(
        df["mode"]
        .value_counts()
        .to_string()
    )

print()
print("---------- DAY COUNT ----------")

if not df.empty:
    print(
        df["day"]
        .value_counts()
        .sort_index()
        .to_string()
    )

print()
print("---------- CONFIGURATIONS ----------")

if not df.empty:

    config_summary = (
        df.groupby(
            [
                "mode",
                "sweeps",
                "gates",
                "gate_spacing_m",
                "max_range_km"
            ]
        )
        .size()
        .reset_index(
            name="file_count"
        )
    )

    print(
        config_summary.to_string(
            index=False
        )
    )

print()
print("---------- TIME RANGE ----------")

if not df.empty:

    print(
        "First:",
        df["timestamp"].min()
    )

    print(
        "Last :",
        df["timestamp"].max()
    )

print()
print("Saved index:")
print(OUTPUT_PATH)

if error_path is not None:

    print()
    print("Error log:")
    print(error_path)