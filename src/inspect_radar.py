import pyart

file_path = r"D:\Documents\Đồ án\2025 Pro-Raw T08\01\NHB250801000007.RAWLTHU"

print("Reading radar data...")
radar = pyart.io.read_sigmet(file_path)

print("\n=========== RADAR OBJECT ============")
print(radar)

print("\n=========== FIELD ============")
print(radar.fields.keys())                          # Cho biết radar có các đại lượng gì

print("\n=========== SCAN TYPE ============")
print(radar.scan_type)                              

print("\n=========== NUMBER OF SWEEPS ============")
print(radar.nsweeps)                                # Số sweep trong file (sweep là 1 lần quét quanh 360° tại một elevation)

print("\n=========== NUMBER OF RAYS ============")
print(radar.nrays)                                  # Mỗi nan hoa là 1 radar ray

print("\n=========== NUMBER OF GATES ============")
print(radar.ngates)                                 # Trên mỗi ray, radar đo ở nhiều khoảng cách khác nhau, Mỗi đoạn đo như vậy gọi là một gate

print("\n=========== FIXED ANGLES ============")
print(radar.fixed_angle["data"])                    # góc elevation của từng sweep

print("\n=========== LOCATION ============")
print("Latitude :", radar.latitude["data"])
print("Longitude :", radar.longitude["data"])
print("Altitude :", radar.altitude["data"])

print("\n========== SWEEP BOUNDARIES ==========")

for i in range(radar.nsweeps):
    start = radar.sweep_start_ray_index["data"][i]
    end = radar.sweep_end_ray_index["data"][i]

    print(
        f"Sweep {i}: "
        f"ray {start} -> {end}, "
        f"total = {end - start + 1}, "
        f"angle = {radar.fixed_angle['data'][i]:.2f}°"
    )



import pandas as pd
import numpy as np

reflectivity = radar.fields["reflectivity"]["data"]
azimuth = radar.azimuth["data"]
elevation = radar.elevation["data"]
ranges = radar.range["data"]

rows = []

for ray in range(5):
    for gate in range(10):
        value = reflectivity[ray, gate]

        rows.append({
            "ray": ray,
            "azimuth_deg": azimuth[ray],
            "elevation_deg": elevation[ray],
            "gate": gate,
            "range_m": ranges[gate],
            "reflectivity_dbz":
                np.nan if np.ma.is_masked(value) else float(value)
        })

df = pd.DataFrame(rows)

print(df.head(30))