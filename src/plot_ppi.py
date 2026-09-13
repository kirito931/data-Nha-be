from pathlib import Path

import matplotlib.pyplot as plt
import pyart


# ============================================================
# 1. ĐƯỜNG DẪN HAI FILE
# ============================================================

long_file = Path(
    r"D:\Documents\Đồ án\2025 Pro-Raw T08\01\NHB250801000007.RAWLTHU"
)

short_file = Path(
    r"D:\Documents\Đồ án\2025 Pro-Raw T08\01\NHB250801000303.RAWLTHY"
)


# ============================================================
# 2. ĐỌC RADAR
# ============================================================

print("Reading Long Range...")
long_radar = pyart.io.read_sigmet(str(long_file))

print("Reading Short Range...")
short_radar = pyart.io.read_sigmet(str(short_file))


# ============================================================
# 3. IN THÔNG TIN ĐỂ KIỂM TRA
# ============================================================

print("\n========== LONG RANGE ==========")
print("Sweeps :", long_radar.nsweeps)
print("Rays   :", long_radar.nrays)
print("Gates  :", long_radar.ngates)
print(
    "Fixed angles:",
    long_radar.fixed_angle["data"]
)
print(
    "Max range:",
    long_radar.range["data"][-1] / 1000,
    "km"
)


print("\n========== SHORT RANGE ==========")
print("Sweeps :", short_radar.nsweeps)
print("Rays   :", short_radar.nrays)
print("Gates  :", short_radar.ngates)
print(
    "Fixed angles:",
    short_radar.fixed_angle["data"]
)
print(
    "Max range:",
    short_radar.range["data"][-1] / 1000,
    "km"
)


# ============================================================
# 4. TẠO DISPLAY
# ============================================================

long_display = pyart.graph.RadarDisplay(long_radar)
short_display = pyart.graph.RadarDisplay(short_radar)


# ============================================================
# 5. VẼ HAI PPI
# ============================================================

fig, axes = plt.subplots(
    1,
    2,
    figsize=(16, 7)
)


# ------------------------------------------------------------
# Long Range
# ------------------------------------------------------------

long_display.plot(
    "reflectivity",
    sweep=0,
    ax=axes[0],
    vmin=-10,
    vmax=70,
    cmap="NWSRef",
    colorbar_flag=True,
    colorbar_label="Reflectivity (dBZ)"
)

long_display.set_limits(
    xlim=(-50, 50),
    ylim=(-50, 50),
    ax=axes[0]
)

axes[0].set_title(
    "Long Range\n"
    "2025-08-01 00:00:07\n"
    "Sweep 0 ≈ 0.5°"
)


# ------------------------------------------------------------
# Short Range
# ------------------------------------------------------------

short_display.plot(
    "reflectivity",
    sweep=0,
    ax=axes[1],
    vmin=-10,
    vmax=70,
    cmap="NWSRef",
    colorbar_flag=True,
    colorbar_label="Reflectivity (dBZ)"
)

short_display.set_limits(
    xlim=(-50, 50),
    ylim=(-50, 50),
    ax=axes[1]
)

axes[1].set_title(
    "Short Range\n"
    "2025-08-01 00:03:03\n"
    "Sweep 0 ≈ 0.5°"
)


# ============================================================
# 6. HOÀN THIỆN
# ============================================================

plt.suptitle(
    "Nha Be Radar - Long Range vs Short Range",
    fontsize=16
)

plt.tight_layout()


# ============================================================
# 7. LƯU ẢNH
# ============================================================

output_dir = Path("outputs/ppi")
output_dir.mkdir(
    parents=True,
    exist_ok=True
)

output_path = (
    output_dir
    / "compare_long_short_sweep0_ver(-50, 50).png"
)

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

print("\nSaved:", output_path)

plt.show()