from pathlib import Path

import matplotlib.pyplot as plt
import pyart

# =========================
# 1. Đường dẫn dữ liệu
# =========================

file_path = Path(
    r"D:\Documents\Đồ án\2025 Pro-Raw T08\01\NHB250801000007.RAWLTHU"
)

output_dir = Path("outputs/ppi")
output_dir.mkdir(parents=True, exist_ok=True)

# =========================
# 2. Đọc radar
# =========================

print("Reading radar...")

radar = pyart.io.read_sigmet(str(file_path))

# =========================
# 3. Tạo radar display
# =========================

display = pyart.graph.RadarDisplay(radar)

# =========================
# 4. Vẽ Reflectivity
# =========================

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111)

display.plot(
    "reflectivity",
    sweep=0,
    ax=ax,
    vmin=-10,
    vmax=70,
    cmap="NWSRef",
    colorbar_label="Reflectivity (dBZ)"
)

display.set_limits(
    xlim=(-300, 300),
    ylim=(-300, 300),
    ax=ax
)

ax.set_title(
    "Nha Be Radar - Reflectivity\n"
    "2025-08-01 00:00:07 - Sweep 0"
)

plt.tight_layout()

# =========================
# 5. Lưu ảnh
# =========================

output_path = output_dir / "NHB_20250801_000007_sweep0.png"

plt.savefig(
    output_path,
    dpi=150,
    bbox_inches="tight"
)

print("Saved:", output_path)

plt.show()