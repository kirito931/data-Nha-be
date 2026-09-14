"""
================================================================================
BƯỚC 4 TRONG PIPELINE: XUẤT TẬP ẢNH RADAR PPI (PLAN POSITION INDICATOR) 224x224
================================================================================
Mục đích:
- Chuyển đổi dữ liệu ma trận phản hồi (dBZ) từ tọa độ cực (góc phương vị và khoảng cách)
  sang hình ảnh phẳng 2 chiều không gian (PPI Image) bằng công cụ đồ họa Py-ART.
- CÁC THIẾT LẬP VẬT LÝ VÀ ĐỒ HỌA THEO CHUẨN BÀI BÁO:
    * sweep=0: Chọn lát cắt quét có góc ngẩng thấp nhất (elevation angle thấp nhất, ~0.5 độ).
               Đây là tầng mây gần mặt đất nhất, phản ánh chính xác lượng mưa rơi xuống đô thị.
    * range_limit_km=150.0: Giới hạn bán kính quan sát là 150 km tính từ tâm đài radar Nhà Bè.
               Bao phủ toàn bộ TP.HCM và các tỉnh lân cận thuộc đồng bằng sông Cửu Long.
    * cmap='NWSRef': Bảng màu chuẩn của Cục Khí tượng Quốc gia Hoa Kỳ (NWS Reflectivity),
               với màu xanh lá/vàng/đỏ/tím tương ứng cường độ mưa từ nhẹ đến đặc biệt nguy hiểm.
    * vmin=-10, vmax=70: Thang giá trị phản xạ dBZ tiêu chuẩn.
    * Chế độ Headless (`matplotlib.use('Agg')`): Vẽ ảnh ngầm trong bộ nhớ RAM/CPU mà không
      mở cửa sổ giao diện đồ họa, giúp tăng tốc độ xuất hàng ngàn ảnh.
    * Kích thước đầu ra: 224x224 pixels, định dạng 3 kênh màu RGB (.png).
      Đây là kích thước đầu vào chuẩn hóa của kiến trúc mạng ConvNeXt.

- Xuất kết quả ra: outputs/ppi/all_scans/*.png
================================================================================
"""
from pathlib import Path
import time
import matplotlib
matplotlib.use('Agg')  # Chạy ngầm không mở cửa sổ đồ họa (tiết kiệm bộ nhớ và tăng tốc)
import matplotlib.pyplot as plt
import pyart
from PIL import Image


def convert_and_save(raw_path: Path, output_path: Path, range_limit_km: float = 150.0) -> bool:
    """
    Hàm đọc 1 file RAW SIGMET và chuyển đổi thành ảnh PPI 224x224 RGB.
    
    Tham số:
        raw_path: Đường dẫn file .RAW* đầu vào
        output_path: Đường dẫn file .png đầu ra
        range_limit_km: Bán kính quan sát xung quanh đài radar (mặc định 150 km)
    """
    # Nếu ảnh đã được xuất trước đó, bỏ qua để tiết kiệm thời gian (cơ chế cache)
    if output_path.exists():
        return True

    try:
        # 1. Đọc dữ liệu radar bằng Py-ART
        radar = pyart.io.read_sigmet(str(raw_path))
        display = pyart.graph.RadarDisplay(radar)

        # 2. Tạo khung vẽ hình vuông tỉ lệ 1:1, không có viền và không có trục tọa độ
        fig = plt.figure(figsize=(6, 6), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')  # Tắt trục tọa độ và thước đo

        # 3. Vẽ trường phản hồi reflectivity ở lát cắt thấp nhất (sweep 0)
        display.plot(
            'reflectivity',
            sweep=0,               # Lát cắt góc ngẩng thấp nhất sát mặt đất
            ax=ax,
            vmin=-10,              # Giá trị dBZ nhỏ nhất hiển thị
            vmax=70,               # Giá trị dBZ lớn nhất hiển thị
            cmap='NWSRef',         # Bảng màu chuẩn radar khí tượng NWS
            colorbar_flag=False,   # Không vẽ thanh chú thích màu (colorbar) vào ảnh
            title_flag=False       # Không vẽ tiêu đề chữ vào ảnh
        )

        # 4. Giới hạn khung nhìn trong phạm vi bán kính +/- 150 km tính từ tâm đài
        display.set_limits(
            xlim=(-range_limit_km, range_limit_km),
            ylim=(-range_limit_km, range_limit_km),
            ax=ax
        )

        # 5. Lưu tạm ra file ảnh đệm
        temp_img_path = output_path.with_suffix('.tmp.png')
        plt.savefig(temp_img_path, format='png')
        plt.close(fig)

        # 6. Dùng thư viện PIL đọc lại, ép về 3 kênh màu RGB và thu nhỏ về chuẩn 224x224 pixels
        with Image.open(temp_img_path) as img:
            img_rgb = img.convert('RGB')
            img_resized = img_rgb.resize((224, 224), Image.Resampling.BILINEAR)
            img_resized.save(output_path)

        # Xóa file tạm sau khi hoàn tất
        if temp_img_path.exists():
            temp_img_path.unlink()

        return True

    except Exception as e:
        print(f"\n[LỖI XUẤT ẢNH] {raw_path.name}: {e}")
        return False


def export_entire_month():
    """
    Hàm tự động quét toàn bộ 31 thư mục ngày trong tháng 08/2025 và xuất ảnh hàng loạt.
    """
    root_raw_dir = Path(r"2025 Pro-Raw T08")
    output_dir = Path("outputs/ppi/all_scans")
    output_dir.mkdir(parents=True, exist_ok=True)

    day_dirs = sorted([d for d in root_raw_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    total_files = sum(len(list(d.glob("*.RAW*"))) for d in day_dirs)

    print("=" * 70)
    print("BẮT ĐẦU XUẤT ẢNH RADAR PPI 224x224 TOÀN BỘ THÁNG 08/2025")
    print(f"Tổng số ngày: {len(day_dirs)} ngày | Tổng số file: {total_files} files")
    print(f"Thư mục lưu ảnh đích: {output_dir}")
    print("=" * 70)

    global_start = time.time()
    processed_total = 0

    for day_dir in day_dirs:
        day_raw_files = sorted(list(day_dir.glob("*.RAW*")))
        day_start = time.time()
        converted_in_day = 0

        for fpath in day_raw_files:
            # Tên file ảnh đích: đổi đuôi .RAW* thành .png
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
              f"Đã chạy: {total_elapsed/60:.1f} phút")

    print("\n" + "=" * 70)
    print(f"-> HOÀN TẤT XUẤT ẢNH TOÀN BỘ THÁNG! Ảnh được lưu tại: {output_dir}")
    print("=" * 70)


if __name__ == '__main__':
    export_entire_month()