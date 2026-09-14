"""
================================================================================
BƯỚC 6 TRONG PIPELINE: GHÉP NỐI METADATA TẠO BỘ DỮ LIỆU TỔNG (DATASET METADATA)
================================================================================
Mục đích:
- Kết hợp 3 bảng dữ liệu trung gian đã chuẩn bị ở các bước trước:
    1. target_mapping_2025-08.csv (Bước 5): Thông tin mốc thời gian và file mục tiêu tương lai.
    2. temporal_groups_2025-08.csv (Bước 2): Tên 4 file quét đầu vào (file_t0, file_t3, file_t10, file_t13).
    3. scan_labels_2025-08.csv     (Bước 3): Bảng tra cứu nhãn thời tiết (weather_class) và giá trị dBZ.
- Tạo ra BẢNG DỮ LIỆU TỔNG HOÀN CHỈNH (Master Dataset):
    Mỗi hàng đại diện cho 1 mẫu huấn luyện hoàn chỉnh, gồm:
      * Đầu vào (X): 4 file ảnh radar liên tiếp (t0, t3, t10, t13).
      * Nhãn mục tiêu (Y): Nhãn 5 lớp thời tiết và giá trị dBZ tương ứng cho cả 4 mốc (0h, 1h, 2h, 3h).
- Xuất kết quả ra: outputs/metadata/dataset_2025-08.csv
================================================================================
"""
from pathlib import Path
import pandas as pd


def build_month_dataset(mapping_csv: Path, temporal_csv: Path, labels_csv: Path, output_csv: Path):
    """
    Ghép nối 3 bảng dữ liệu thành một file Dataset Metadata duy nhất.
    """
    print("=" * 70)
    print("BẮT ĐẦU GHÉP NỐI CÁC BẢNG METADATA THÀNH DATASET TỔNG")
    print("=" * 70)

    # 1. Đọc các bảng dữ liệu trung gian
    df_map = pd.read_csv(mapping_csv)
    df_temp = pd.read_csv(temporal_csv)
    df_lab = pd.read_csv(labels_csv)

    print(f"-> target_mapping : {len(df_map)} hàng")
    print(f"-> temporal_groups: {len(df_temp)} hàng")
    print(f"-> scan_labels    : {len(df_lab)} hàng")

    # 2. Ghép 4 file đầu vào (file_t0, file_t3, file_t10, file_t13) vào bảng mapping dựa theo group_id
    input_cols = ["group_id", "file_t0", "file_t3", "file_t10", "file_t13"]
    df_merged = pd.merge(df_map, df_temp[input_cols], on="group_id", how="left")

    # 3. Tạo từ điển tra cứu nhanh (Dictionary Lookup) để tăng tốc độ gán nhãn
    # Key: Tên file radar -> Value: Lớp thời tiết (Clear, Light rain,...)
    label_dict = dict(zip(df_lab["filename"], df_lab["weather_class"]))
    # Key: Tên file radar -> Value: Giá trị dBZ có trọng số
    dbz_dict = dict(zip(df_lab["filename"], df_lab["weighted_dbz"]))

    # 4. Ánh xạ nhãn và giá trị dBZ cho cả 4 mốc dự báo tương lai
    for horizon in ["0h", "1h", "2h", "3h"]:
        f_col = f"target_{horizon}_file"
        df_merged[f"label_{horizon}"] = df_merged[f_col].map(label_dict)
        df_merged[f"dbz_{horizon}"] = df_merged[f_col].map(dbz_dict)

    # 5. Sắp xếp lại thứ tự các cột cốt lõi lên đầu để dễ theo dõi và kiểm tra
    core_cols = [
        "group_id", "input_time",
        "file_t0", "file_t3", "file_t10", "file_t13",
        "target_0h_file", "label_0h", "dbz_0h",
        "target_1h_file", "label_1h", "dbz_1h",
        "target_2h_file", "label_2h", "dbz_2h",
        "target_3h_file", "label_3h", "dbz_3h"
    ]
    remaining_cols = [c for c in df_merged.columns if c not in core_cols]
    df_final = df_merged[core_cols + remaining_cols]

    # 6. Lưu file CSV kết quả
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_csv, index=False)

    print("\n" + "=" * 70)
    print("HOÀN TẤT TẠO BẢNG DATASET METADATA TOÀN THÁNG")
    print("=" * 70)
    print(f"-> File đã lưu tại: {output_csv}")
    print(f"-> Tổng số mẫu nhóm (groups): {len(df_final)}")

    print("\n--- THỐNG KÊ SỐ LƯỢNG MẪU CÓ NHÃN THEO TỪNG MỐC DỰ BÁO ---")
    for h in ["0h", "1h", "2h", "3h"]:
        valid_cnt = df_final[f"label_{h}"].notna().sum()
        pct = (valid_cnt / len(df_final)) * 100
        print(f"  * Mốc {h:3s}: {valid_cnt:4d}/{len(df_final)} mẫu sẵn sàng ({pct:5.1f}%)")
    print("=" * 70)


if __name__ == "__main__":
    mapping_file = Path("outputs/metadata/target_mapping_2025-08.csv")
    temporal_file = Path("outputs/metadata/temporal_groups_2025-08.csv")
    labels_file = Path("outputs/metadata/scan_labels_2025-08.csv")
    out_file = Path("outputs/metadata/dataset_2025-08.csv")

    build_month_dataset(mapping_file, temporal_file, labels_file, out_file)