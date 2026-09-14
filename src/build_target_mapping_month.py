"""
================================================================================
BƯỚC 5 TRONG PIPELINE: ÁNH XẠ NHÃN MỤC TIÊU TƯƠNG LAI (FUTURE TARGET MAPPING)
================================================================================
Mục đích:
- Bản chất của bài toán Dự báo cực ngắn (Nowcasting) là: Dùng dữ liệu quan sát trong
  quá khứ (chuỗi 4 ảnh tại thời điểm t0) để dự đoán tình trạng thời tiết tại một thời điểm
  trong TƯƠNG LAI (t0 + delta).
- Bài báo gốc nghiên cứu 4 mốc thời gian tương lai (Horizons):
    * 0h (Now): Dự báo ngay tại thời điểm hiện tại t0 (đối chiếu baseline tức thời).
    * 1h      : Dự báo thời tiết sau 1 giờ  (t0 + 60 phút = 3.600 giây).
    * 2h      : Dự báo thời tiết sau 2 giờ  (t0 + 120 phút = 7.200 giây).
    * 3h      : Dự báo thời tiết sau 3 giờ  (t0 + 180 phút = 10.800 giây).

- THUẬT TOÁN TÌM KIẾM:
    Với mỗi nhóm chuỗi thời gian có mốc bắt đầu t0:
    1. Tính thời gian mục tiêu kỳ vọng: target_time = t0 + horizon (0h, 1h, 2h, 3h).
    2. Quét bảng chỉ mục radar toàn tháng để tìm lần quét thực tế có timestamp gần nhất.
    3. Kiểm tra sai số thời gian: |timestamp_thực_tế - timestamp_kỳ_vọng| <= 60 giây.
       (Ngưỡng 60 giây bảo đảm tính chính xác của dữ liệu khí tượng).
    4. Nếu tìm thấy file trong ngưỡng sai số, ghi nhận file đó làm đại diện tương lai.
       Nếu không tìm thấy (do trạm radar bảo trì hoặc ngừng quét), đánh dấu là None.

- Xuất kết quả ra: outputs/metadata/target_mapping_2025-08.csv
================================================================================
"""
from pathlib import Path
import pandas as pd


# ============================================================
# 1. ĐƯỜNG DẪN CÁC TỆP TIN
# ============================================================

# Bảng nhóm chuỗi thời gian (tạo từ Bước 2)
GROUPS_PATH = Path("outputs/metadata/temporal_groups_2025-08.csv")

# Bảng chỉ mục toàn bộ các lần quét trong tháng (tạo từ Bước 1)
INDEX_PATH = Path("outputs/metadata/radar_index_2025-08.csv")

# Bảng ánh xạ mục tiêu đầu ra
OUTPUT_PATH = Path("outputs/metadata/target_mapping_2025-08.csv")


# ============================================================
# 2. NẠP DỮ LIỆU
# ============================================================

print("=" * 70)
print("BẮT ĐẦU ÁNH XẠ MỤC TIÊU DỰ BÁO TƯƠNG LAI (0h, 1h, 2h, 3h)")
print("=" * 70)

groups = pd.read_csv(
    GROUPS_PATH,
    parse_dates=["t0", "t3", "t10", "t13"]
)

index = pd.read_csv(
    INDEX_PATH,
    parse_dates=["timestamp"]
)

# Sắp xếp index theo thời gian để tối ưu hóa việc tìm kiếm
index = index.sort_values("timestamp").reset_index(drop=True)

print(f"Tổng số nhóm đầu vào (Temporal groups) : {len(groups)} nhóm")
print(f"Tổng số lần quét radar (Radar scans)    : {len(index)} lần quét")


# ============================================================
# 3. HÀM TÌM LẦN QUÉT GẦN NHẤT VỚI THỜI ĐIỂM KỲ VỌNG
# ============================================================

# Ngưỡng sai số thời gian tối đa cho phép: 60 giây
MAX_DIFF_SECONDS = 60.0


def find_nearest_scan(target_time):
    """
    Tìm lần quét radar trong bảng chỉ mục có thời gian gần nhất với target_time.
    
    Trả về:
        dict: Thông tin file nếu độ lệch <= 60 giây, ngược lại trả về None.
    """
    # Tính khoảng cách thời gian tuyệt đối giữa tất cả các lần quét với target_time
    differences = (index["timestamp"] - target_time).abs()

    # Lấy vị trí chỉ số (index) có độ lệch nhỏ nhất
    nearest_idx = differences.idxmin()
    nearest = index.loc[nearest_idx]
    diff_seconds = differences.loc[nearest_idx].total_seconds()

    # Nếu sai số nằm trong giới hạn cho phép (<= 60 giây)
    if diff_seconds <= MAX_DIFF_SECONDS:
        return {
            "timestamp": nearest["timestamp"],
            "filename": nearest["filename"],
            "mode": nearest["mode"],
            "diff_seconds": diff_seconds,
        }

    # Nếu bị khuyết dữ liệu quá 60 giây
    return {
        "timestamp": None,
        "filename": None,
        "mode": None,
        "diff_seconds": diff_seconds,
    }


# ============================================================
# 4. TIẾN HÀNH ÁNH XẠ TỪNG NHÓM VÀ TỪNG MỐC HORIZON
# ============================================================

rows = []

for _, group in groups.iterrows():
    # Thời điểm bắt đầu của nhóm quan sát hiện tại (t0)
    input_time = group["t0"]

    row = {
        "group_id": group["group_id"],
        "input_time": input_time,

        # Lưu lại tên 4 file đầu vào của chuỗi
        "input_file_t0": group["file_t0"],
        "input_file_t3": group["file_t3"],
        "input_file_t10": group["file_t10"],
        "input_file_t13": group["file_t13"],
    }

    # Ánh xạ cho 4 mốc thời gian tương lai: 0h, 1h, 2h, 3h
    for horizon in [0, 1, 2, 3]:
        # Thời điểm tương lai cần dự báo
        target_time = input_time + pd.Timedelta(hours=horizon)

        # Tìm lần quét radar thực tế tương ứng trong tương lai
        result = find_nearest_scan(target_time)

        suffix = f"{horizon}h"
        row[f"target_{suffix}_expected"] = target_time
        row[f"target_{suffix}_actual"] = result["timestamp"]
        row[f"target_{suffix}_file"] = result["filename"]
        row[f"target_{suffix}_mode"] = result["mode"]
        row[f"target_{suffix}_diff_sec"] = result["diff_seconds"]

    rows.append(row)


# ============================================================
# 5. LƯU BẢNG ÁNH XẠ MỤC TIÊU VÀ BÁO CÁO THỐNG KÊ
# ============================================================

result = pd.DataFrame(rows)
result.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 70)
print("KẾT QUẢ ÁNH XẠ MỤC TIÊU TƯƠNG LAI HOÀN TẤT")
print("=" * 70)
print(f"Tổng số nhóm xử lý: {len(result)}")

# Thống kê tỷ lệ có sẵn dữ liệu nhãn tương lai cho từng mốc
for horizon in [0, 1, 2, 3]:
    col = f"target_{horizon}h_file"
    available = result[col].notna().sum()
    missing = result[col].isna().sum()
    pct = (available / len(result)) * 100
    print(f"  * Mốc +{horizon}h: Có dữ liệu: {available:4d}/{len(result)} ({pct:5.1f}%) | Bị khuyết: {missing} nhóm")

print(f"\n-> Bảng ánh xạ đã lưu tại: {OUTPUT_PATH}")
print("=" * 70)