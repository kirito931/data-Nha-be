import numpy as np
import pyart


file_path = r"D:\Documents\Đồ án\2025 Pro-Raw T08\01\NHB250801000007.RAWLTHU"

print("Reading radar data...")

radar = pyart.io.read_sigmet(file_path)

reflectivity = radar.fields["reflectivity"]["data"]
ranges_km = radar.range["data"] / 1000


# Các khoảng cách muốn kiểm tra
range_bins = [
    (0, 50),
    (50, 100),
    (100, 150),
    (150, 200),
    (200, 250),
    (250, 300),
]


for sweep in range(radar.nsweeps):

    print("\n" + "=" * 60)
    print(
        f"SWEEP {sweep} "
        f"({radar.fixed_angle['data'][sweep]:.2f} degrees)"
    )
    print("=" * 60)

    start = radar.sweep_start_ray_index["data"][sweep]
    end = radar.sweep_end_ray_index["data"][sweep]

    sweep_data = reflectivity[start:end + 1]

    for min_km, max_km in range_bins:

        gate_mask = (
            (ranges_km >= min_km)
            & (ranges_km < max_km)
        )

        region = sweep_data[:, gate_mask]

        mask = np.ma.getmaskarray(region)

        total = region.size
        masked = mask.sum()
        valid = total - masked

        valid_values = region.compressed()

        print(
            f"\nRange {min_km:3d}-{max_km:3d} km"
        )

        print(
            f"  Valid ratio : "
            f"{valid / total * 100:.4f}%"
        )

        print(
            f"  Masked ratio: "
            f"{masked / total * 100:.4f}%"
        )

        if len(valid_values) > 0:

            print(
                f"  Median dBZ  : "
                f"{np.median(valid_values):.4f}"
            )

            print(
                f"  95% dBZ     : "
                f"{np.percentile(valid_values, 95):.4f}"
            )

            print(
                f"  Max dBZ     : "
                f"{valid_values.max():.4f}"
            )

            print(f"  Valid cells  : {valid}/{total}")