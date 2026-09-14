from pathlib import Path
from datetime import datetime

import pandas as pd
import pyart


folder = Path(r"D:\Documents\Đồ án\2025 Pro-Raw T08\01")

rows = []

files = sorted(folder.glob("NHB*.RAW*"))

print(f"Found {len(files)} files.")


for i, file_path in enumerate(files, start=1):

    print(f"[{i}/{len(files)}] {file_path.name}")

    try:
        radar = pyart.io.read_sigmet(str(file_path))

        timestamp = datetime.strptime(
            file_path.name[3:15],
            "%y%m%d%H%M%S"
        )

        ranges = radar.range["data"]

        gate_spacing = (
            ranges[1] - ranges[0]
        )

        max_range_km = ranges[-1] / 1000

        if radar.nsweeps == 4 and abs(gate_spacing - 600) < 1:
            mode = "Long Range"

        elif radar.nsweeps == 8 and abs(gate_spacing - 240) < 1:
            mode = "Short Range"

        else:
            mode = "Unknown"

        rows.append({
            "timestamp": timestamp,
            "filename": file_path.name,
            "mode": mode,
            "sweeps": radar.nsweeps,
            "rays": radar.nrays,
            "gates": radar.ngates,
            "gate_spacing_m": gate_spacing,
            "max_range_km": max_range_km,
        })

    except Exception as e:

        print("ERROR:", file_path.name, e)

        rows.append({
            "timestamp": None,
            "filename": file_path.name,
            "mode": "ERROR",
            "sweeps": None,
            "rays": None,
            "gates": None,
            "gate_spacing_m": None,
            "max_range_km": None,
        })


df = pd.DataFrame(rows)

df = df.sort_values("timestamp")

output_dir = Path("outputs/metadata")
output_dir.mkdir(parents=True, exist_ok=True)

output_path = (
    output_dir / "radar_index_2025-08-01.csv"
)

df.to_csv(output_path, index=False)

print("\n========== MODE COUNT ==========")
print(df["mode"].value_counts())

print("\n========== FIRST 30 FILES ==========")
print(df.head(30).to_string(index=False))

print("\nSaved:", output_path)