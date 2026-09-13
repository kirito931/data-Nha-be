"""
Trích xuất ảnh PPI không viền (headless) và resize về 224x224 chuẩn ConvNeXt
"""
from pathlib import Path
import matplotlib.pyplot as plt
import pyart
from PIL import Image


def convert_raw_to_ppi(raw_path: Path, output_path: Path, range_limit_km: float = 150.0):
    """
    Đọc 1 file radar RAW -> xuất ảnh PPI sweep 0 -> resize chuẩn (224, 224) RGB
    """
    try:
        # Đảm bảo thư mục lưu ảnh tồn tại
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 1. Đọc file radar thô
        radar = pyart.io.read_sigmet(str(raw_path))
        display = pyart.graph.RadarDisplay(radar)
        
        # 2. Tạo figure không viền, không trục (headless)
        fig = plt.figure(figsize=(6, 6), dpi=100)
        ax = fig.add_axes([0, 0, 1, 1])  # Phủ kín 100% khung hình
        ax.axis('off')                   # Tắt toàn bộ trục tọa độ x, y
        
        # 3. Vẽ góc quét thấp nhất (sweep 0 ≈ 0.5 độ)
        display.plot(
            'reflectivity',
            sweep=0,
            ax=ax,
            vmin=-10,
            vmax=70,
            cmap='NWSRef',
            colorbar_flag=False,  # Bỏ thanh thang đo màu bên cạnh
            title_flag=False      # Bỏ tiêu đề chữ trên ảnh
        )
        
        # Giới hạn không gian đúng [-150, 150] km như bạn đã chọn
        display.set_limits(
            xlim=(-range_limit_km, range_limit_km),
            ylim=(-range_limit_km, range_limit_km),
            ax=ax
        )
        
        # 4. Lưu ra file ảnh tạm
        temp_img_path = output_path.with_suffix('.tmp.png')
        plt.savefig(temp_img_path, format='png')
        plt.close(fig)  # Đóng figure để giải phóng RAM
        
        # 5. Dùng thư viện PIL để resize về đúng 224x224 theo yêu cầu mô hình
        with Image.open(temp_img_path) as img:
            img_rgb = img.convert('RGB')
            img_resized = img_rgb.resize((224, 224), Image.Resampling.BILINEAR)
            img_resized.save(output_path)
            
        # Xóa file tạm
        if temp_img_path.exists():
            temp_img_path.unlink()
            
        return True

    except Exception as e:
        print(f"Lỗi khi xử lý file {raw_path.name}: {e}")
        return False


# ============================================================
# KHỐI CHẠY THỬ NGHIỆM CHO 1 FILE DUY NHẤT
# ============================================================
if __name__ == '__main__':
    # File mẫu ngày 01/08/2025 (Long Range)
    sample_file = Path(r"D:\Documents\Đồ án\2025 Pro-Raw T08\01\NHB250801000007.RAWLTHU")
    
    # Nơi lưu ảnh kiểm tra
    test_output = Path("outputs/ppi/test_sample_224.png")
    
    print("Đang chạy thử nghiệm tạo 1 ảnh PPI...")
    success = convert_raw_to_ppi(sample_file, test_output, range_limit_km=150.0)
    
    if success:
        print(f"-> Thành công! Đã lưu ảnh kiểm tra tại: {test_output}")
        # Kiểm tra lại kích thước ảnh đã lưu
        with Image.open(test_output) as img:
            print(f"-> Kích thước ảnh thực tế: {img.size} (Width x Height)")
            print(f"-> Chế độ màu: {img.mode}")
    else:
        print("-> Thất bại. Hãy kiểm tra lại đường dẫn file hoặc thông báo lỗi ở trên.")