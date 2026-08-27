from pathlib import Path

import numpy as np
import pyart


files = [
    Path(r"D:\Documents\Đồ án\2025 Pro-Raw T08\01\NHB250801000007.RAWLTHU"),
    Path(r"D:\Documents\Đồ án\2025 Pro-Raw T08\01\NHB250801000303.RAWLTHY"),
    Path(r"D:\Documents\Đồ án\2025 Pro-Raw T08\01\NHB250801001007.RAWLTJ1"),
    Path(r"D:\Documents\Đồ án\2025 Pro-Raw T08\01\NHB250801001304.RAWLTJ5")
]


for file_path in files:
    print("\nReading radar data...")
    radar = pyart.io.read_sigmet(file_path)

    print("Filename        :", file_path.name)
    print("Scan type       :", radar.scan_type)
    print("Sweeps          :", radar.nsweeps)
    print("Rays            :", radar.nrays)
    print("Gates           :", radar.ngates)
    print("Fixed angles    :", radar.fixed_angle["data"])

    print("Gate spacing    :", np.median(np.diff(radar.range["data"])), "m")
    print("Maximum range   :", radar.range["data"][-1] / 1000, "km")

    reflectivity = radar.fields["reflectivity"]["data"]
    valid = reflectivity.compressed()
    mask = np.ma.getmaskarray(reflectivity)

    print("Shape           :", reflectivity.shape)
    print("Valid cells     :", valid.size)
    print("Masked ratio    :", mask.mean() * 100, "%")

    if valid.size:
        print("Min dBZ         :", valid.min())
        print("Median dBZ      :", np.median(valid))
        print("Mean dBZ        :", valid.mean())
        print("95th percentile :", np.percentile(valid, 95))
        print("99th percentile :", np.percentile(valid, 99))
        print("Max dBZ         :", valid.max())