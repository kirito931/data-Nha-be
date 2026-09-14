"""
Trích xuất toàn bộ ảnh PPI (224x224, RGB, headless) cho ngày 01/08/2025
"""
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')  # Chế độ headless, không bật cửa sổ GUI để tiết kiệm RAM
import matplotlib.pyplot as plt
import pyart
from PIL import Image


def convert_and_save(raw_path: Path, output_path: Path, range_limit_km: float = 150.0):
    if output_path.exists():
        return True
    try:
        radar = pyart.io.read_sigmet(str(raw_path))
        display = pyart.graph.RadarDisplay(radar)

        fig = plt.figure(figsize=(6, 6), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        display.plot(
            'reflectivity',
            sweep=0,
            ax=ax,
            vmin=-10,
            vmax=70,
            cmap='NWSRef',
            colorbar_flag=False,
            title_flag=False
        )
        display.set_limits(
            xlim=(-range_limit_km, range_limit_km),
            ylim=(-range_limit_km, range_limit_km),
            ax=ax
        )

        temp_img_path = output_path.with_suffix('.tmp.png')
        plt.savefig(temp_img_path, format='png')
        plt.close(fig)

        with Image.open(temp_img_path) as img:
            img_rgb = img.convert('RGB')
            img_resized = img_rgb.resize((224, 224), Image.Resampling.BILINEAR)
            img_resized.save(output_path)

        if temp_img_path.exists():
            temp_img_path.unlink()
        return True
    except Exception as e:
        print(f"Lỗi {raw_path.name}: {e}")
        return False


def run_export():
    raw_dir = Path(r"D:\Documents\Đồ án\2025 Pro-Raw T08\01")
    output_dir = Path("outputs/ppi/2025-08-01")
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_files = sorted(list(raw_dir.glob("*.RAW*")))
    total = len(raw_files)
    print(f"Bắt đầu xuất {total} ảnh PPI cho ngày 01/08/2025...")

    start_time = time.time()
    for idx, fpath in enumerate(raw_files, start=1):
        out_name = fpath.stem + ".png"
        out_path = output_dir / out_name
        convert_and_save(fpath, out_path, range_limit_km=150.0)

        if idx % 30 == 0 or idx == total:
            elapsed = time.time() - start_time
            print(f"[{idx:3d}/{total:3d}] Hoàn thành ({elapsed:.1f}s)")

    print(f"-> Hoàn tất xuất ảnh tại: {output_dir}")


if __name__ == '__main__':
    run_export()