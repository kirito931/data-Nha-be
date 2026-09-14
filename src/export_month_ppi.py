"""
Trích xuất tự động toàn bộ ảnh PPI (224x224, RGB, headless) cho cả 31 ngày tháng 08/2025.
Có cơ chế kiểm tra checkpoint: file nào đã xuất rồi sẽ tự động bỏ qua.
"""
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')  # Chạy headless, không mở cửa sổ đồ họa
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
        print(f"\n[LỖI] {raw_path.name}: {e}")
        return False


def export_entire_month():
    root_raw_dir = Path(r"D:\Documents\Đồ án\2025 Pro-Raw T08")
    # Lưu toàn bộ ảnh tập trung vào thư mục outputs/ppi/all_scans
    output_dir = Path("outputs/ppi/all_scans")
    output_dir.mkdir(parents=True, exist_ok=True)

    day_dirs = sorted([d for d in root_raw_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    total_files = sum(len(list(d.glob("*.RAW*"))) for d in day_dirs)

    print(f"Tìm thấy {len(day_dirs)} thư mục ngày ({total_files} file RAW).")
    print(f"Thư mục lưu ảnh đích: {output_dir}")

    global_start = time.time()
    processed_total = 0

    for day_dir in day_dirs:
        day_raw_files = sorted(list(day_dir.glob("*.RAW*")))
        day_start = time.time()
        converted_in_day = 0

        for fpath in day_raw_files:
            out_path = output_dir / (fpath.stem + ".png")
            if not out_path.exists():
                success = convert_and_save(fpath, out_path, range_limit_km=150.0)
                if success:
                    converted_in_day += 1
            processed_total += 1

        day_elapsed = time.time() - day_start
        total_elapsed = time.time() - global_start
        pct = (processed_total / total_files) * 100
        print(f"[Ngày {day_dir.name}/31] Xuất mới {converted_in_day:3d} ảnh ({day_elapsed:.1f}s) | "
              f"Tiến độ: {processed_total}/{total_files} ({pct:5.1f}%) | "
              f"Thời gian đã chạy: {total_elapsed/60:.1f} phút")

    print(f"\n-> HOÀN TẤT XUẤT ẢNH TOÀN THÁNG! Lưu tại: {output_dir}")


if __name__ == '__main__':
    export_entire_month()