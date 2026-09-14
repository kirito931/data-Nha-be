"""
================================================================================
BƯỚC 1 TRONG PIPELINE: QUÉT VÀ LẬP CHỈ MỤC DỮ LIỆU RADAR TOÀN THÁNG (INDEXING)
================================================================================
Mục đích:
- Quét toàn bộ thư mục chứa các file nhị phân SIGMET (.RAW*) trong 31 ngày của tháng 08/2025.
- Đọc thông tin tiêu đề (Header/Metadata) của từng file bằng thư viện Py-ART.
- Trích xuất thời gian quan trắc (timestamp), số lát cắt quét (sweeps), số cổng đo (gates).
- Phân loại chế độ quét của trạm Nhà Bè:
    * Long Range (Tầm xa): 4 sweeps, độ phân giải ~600m/gate, bán kính phủ ~300km.
    * Short Range (Tầm gần): 8 sweeps, độ phân giải ~240m/gate, bán kính phủ ~120km.
- Xuất kết quả ra file bảng: outputs/metadata/radar_index_2025-08.csv
================================================================================
"""
from pathlib import Path
from datetime import datetime
import pandas as pd
import pyart


# ============================================================
# 1. THIẾT LẬP ĐƯỜNG DẪN THƯ MỤC VÀ TỆP TIN
# ============================================================

# Thư mục gốc chứa 31 thư mục con (tương ứng ngày 01 đến 31 của tháng 08/2025)
DATA_ROOT = Path(r"2025 Pro-Raw T08")

# Thư mục lưu kết quả metadata
OUTPUT_DIR = Path("outputs/metadata")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Tệp CSV đầu ra lưu toàn bộ chỉ mục các file radar trong tháng
OUTPUT_PATH = OUTPUT_DIR / "radar_index_2025-08.csv"


# ============================================================
# 2. CÁC HÀM TIỆN ÍCH BÓC TÁCH THÔNG TIN
# ============================================================

def classify_scan_mode(radar):
    """
    Hàm phân loại chế độ quét của trạm radar Nhà Bè dựa trên cấu trúc vật lý:
    
    Tham số:
        radar: Đối tượng Radar đọc từ Py-ART (pyart.core.Radar)
        
    Trả về:
        tuple: (Tên chế độ, khoảng cách giữa 2 cổng đo tính bằng mét, tầm xa tối đa tính bằng km)
    
    Quy tắc nghiệp vụ trạm Nhà Bè:
        - Long Range: Có 4 lát cắt quét (nsweeps = 4), bước cự ly gate_spacing ~ 600m.
        - Short Range: Có 8 lát cắt quét (nsweeps = 8), bước cự ly gate_spacing ~ 240m.
    """
    # Mảng khoảng cách từ tâm đài radar đến từng cổng đo (đơn vị: mét)
    ranges = radar.range["data"]

    # Nếu file lỗi không đủ 2 cổng đo thì không xác định được
    if len(ranges) < 2:
        return "Unknown", None, None

    # Bước cự ly (độ phân giải dọc theo tia radar) = khoảng cách cổng 1 trừ cổng 0
    gate_spacing = float(ranges[1] - ranges[0])
    # Tầm quét xa nhất của radar (cổng cuối cùng đổi sang km)
    max_range_km = float(ranges[-1] / 1000.0)

    # Kiểm tra điều kiện Long Range (4 sweeps, bước cự ly 600 mét)
    if radar.nsweeps == 4 and abs(gate_spacing - 600.0) < 1.0:
        return "Long Range", gate_spacing, max_range_km

    # Kiểm tra điều kiện Short Range (8 sweeps, bước cự ly 240 mét)
    if radar.nsweeps == 8 and abs(gate_spacing - 240.0) < 1.0:
        return "Short Range", gate_spacing, max_range_km

    # Trường hợp các góc quét đặc biệt khác (hiếm gặp)
    return "Unknown", gate_spacing, max_range_km


def parse_timestamp(filename: str) -> datetime:
    """
    Hàm giải mã thời gian phát xung quét từ tên tệp SIGMET gốc.
    
    Quy ước đặt tên file của hệ thống radar Nhà Bè:
        Ví dụ: NHB250801000007.RAWLTHU
               --- ^^^^^^^^^^^^
               NHB: Viết tắt đài radar Nhà Bè
               25 : Năm 2025
               08 : Tháng 08
               01 : Ngày 01
               00 : 00 Giờ
               00 : 00 Phút
               07 : 07 Giây
               .RAWLTHU: Đuôi nhị phân định dạng IRIS SIGMET
    
    Cắt chuỗi từ ký tự thứ 3 đến ký tự thứ 15: "250801000007"
    Định dạng phân tích strptime: "%y%m%d%H%M%S"
    """
    timestamp_text = filename[3:15]
    return datetime.strptime(timestamp_text, "%y%m%d%H%M%S")


# ============================================================
# 3. QUÉT VÀ THU THẬP DANH SÁCH FILE TRONG 31 NGÀY
# ============================================================

all_files = []

print("=" * 70)
print("BẮT ĐẦU QUÉT THƯ MỤC DỮ LIỆU TOÀN THÁNG 08/2025")
print("=" * 70)

# Lặp qua từng thư mục ngày (từ 01 đến 31)
for day_dir in sorted(DATA_ROOT.iterdir()):
    if not day_dir.is_dir():
        continue

    # Lấy danh sách các file bắt đầu bằng NHB và có đuôi .RAW*
    day_files = sorted(day_dir.glob("NHB*.RAW*"))
    print(f"-> Ngày {day_dir.name:>2}: Tìm thấy {len(day_files)} file RAW")
    all_files.extend(day_files)

print("-" * 70)
print(f"TỔNG CỘNG TÌM THẤY: {len(all_files)} file RAW cần lập chỉ mục.")
print("-" * 70)


# ============================================================
# 4. ĐỌC METADATA CHI TIẾT TỪNG FILE BẰNG PY-ART
# ============================================================

rows = []
errors = []

print("\n" + "=" * 70)
print("TIẾN HÀNH TRÍCH XUẤT THÔNG TIN METADATA BẰNG PY-ART")
print("=" * 70)

for i, file_path in enumerate(all_files, start=1):
    # In tiến trình mỗi 500 file để theo dõi
    if i % 500 == 0 or i == len(all_files):
        print(f"-> Tiến độ: [{i}/{len(all_files)}] ({(i/len(all_files))*100:.1f}%) | Đang đọc: {file_path.name}")

    try:
        # 1. Đọc file nhị phân SIGMET bằng Py-ART
        radar = pyart.io.read_sigmet(str(file_path))

        # 2. Giải mã thời gian quét
        timestamp = parse_timestamp(file_path.name)

        # 3. Phân loại Long Range hay Short Range
        mode, gate_spacing, max_range_km = classify_scan_mode(radar)

        # 4. Lấy ma trận dữ liệu phản hồi độ phản xạ (reflectivity) để kiểm tra kích thước
        reflectivity = radar.fields["reflectivity"]["data"]

        # 5. Lưu thông tin thành một bản ghi (row)
        rows.append({
            "timestamp": timestamp,
            "day": int(file_path.parent.name),
            "filename": file_path.name,
            "relative_path": str(file_path.relative_to(DATA_ROOT)),
            "mode": mode,
            "scan_type": radar.scan_type,
            "sweeps": radar.nsweeps,
            "rays": radar.nrays,
            "gates": radar.ngates,
            "gate_spacing_m": gate_spacing,
            "max_range_km": max_range_km,
            "reflectivity_shape": str(reflectivity.shape)
        })

    except Exception as e:
        # Nếu file bị lỗi (hỏng dữ liệu do truyền tải), ghi lại vào danh sách lỗi
        print(f"   [LỖI FILE] {file_path.name}: {type(e).__name__} - {e}")
        errors.append({
            "filename": file_path.name,
            "relative_path": str(file_path.relative_to(DATA_ROOT)),
            "error_type": type(e).__name__,
            "error_message": str(e)
        })


# ============================================================
# 5. TỔ CHỨC DỮ LIỆU THÀNH DATAFRAME VÀ SẮP XẾP THỜI GIAN
# ============================================================

df = pd.DataFrame(rows)

# Sắp xếp lại toàn bộ bản ghi theo thứ tự thời gian tăng dần từ 01/08 đến 31/08
if not df.empty:
    df = df.sort_values("timestamp").reset_index(drop=True)


# ============================================================
# 6. LƯU KẾT QUẢ RA FILE CSV
# ============================================================

df.to_csv(OUTPUT_PATH, index=False)
print(f"\n-> ĐÃ LƯU BẢNG CHỈ MỤC TOÀN THÁNG TẠI: {OUTPUT_PATH}")

# Lưu danh sách file lỗi (nếu có) để đối chiếu
if errors:
    errors_df = pd.DataFrame(errors)
    error_path = OUTPUT_DIR / "radar_index_errors_2025-08.csv"
    errors_df.to_csv(error_path, index=False)
    print(f"-> Đã ghi nhận {len(errors)} file lỗi tại: {error_path}")


# ============================================================
# 7. BÁO CÁO THỐNG KÊ TỔNG QUAN
# ============================================================

print("\n" + "=" * 70)
print("BÁO CÁO TỔNG QUAN DỮ LIỆU ĐÃ LẬP CHỈ MỤC")
print("=" * 70)
print(f"Tổng số file đọc thành công : {len(df)}")
print(f"Tổng số file gặp lỗi đọc   : {len(errors)}")

if not df.empty:
    print("\n--- PHÂN BỐ CHẾ ĐỘ QUÉT (MODE) ---")
    print(df["mode"].value_counts().to_string())

    print("\n--- THỜI ĐIỂM QUAN TRẮC ĐẦU TIÊN VÀ CUỐI CÙNG ---")
    print(f"File đầu tiên : {df['timestamp'].min()}")
    print(f"File cuối cùng: {df['timestamp'].max()}")

print("=" * 70)