import numpy as np
import pyart

file_path = r"D:\Documents\Đồ án\2025 Pro-Raw T08\01\NHB250801000007.RAWLTHU"

print("Reading radar data...")

radar = pyart.io.read_sigmet(file_path)

reflectivity = radar.fields["reflectivity"]["data"]


print("============ WHOLE VOLUME =============")

total_cells = reflectivity.size

masked = np.ma.getmaskarray(reflectivity)

masked_cells = masked.sum()
valid_cells = total_cells - masked_cells

print("Total cells :", total_cells)
print("Valid cells :", valid_cells)
print("Masked cells:", masked_cells)

print(
    "Masked ratio:",
    round(masked_cells / total_cells * 100, 2),
    "%"
)

print("\n========== EACH SWEEP ==========")

for sweep in range(radar.nsweeps):
    start = radar.sweep_start_ray_index["data"][sweep]
    end = radar.sweep_end_ray_index["data"][sweep]

    sweep_data = reflectivity[start:end + 1]

    sweep_mask = np.ma.getmaskarray(sweep_data)

    total = sweep_data.size
    missing = sweep_mask.sum()
    valid = total - missing

    print(f"\nSweep {sweep}")

    print(
        "Fixed angle:",
        radar.fixed_angle["data"][sweep]
    )

    print("Shape       :", sweep_data.shape)
    print("Total cells :", total)
    print("Valid cells :", valid)
    print("Masked cells:", missing)

    print(
        "Masked ratio:",
        round(missing / total * 100, 2),
        "%"
    )

    valid_values = sweep_data.compressed()

    if len(valid_values) > 0:

        print(
            "Min dBZ     :",
            valid_values.min()
        )

        print(
            "Median dBZ  :",
            np.median(valid_values)
        )

        print(
            "Mean dBZ    :",
            valid_values.mean()
        )

        print(
            "Max dBZ     :",
            valid_values.max()
        )