"""
Ghép nối bảng nhóm thời gian (target_mapping) với bảng nhãn thời tiết (scan_labels)
và bảng nhóm chuỗi ảnh đầu vào (temporal_groups).
"""
from pathlib import Path
import pandas as pd


def build_labeled_dataset(mapping_csv: Path, labels_csv: Path, output_csv: Path):
    if not mapping_csv.exists() or not labels_csv.exists():
        print("Lỗi: Không tìm thấy file đầu vào. Vui lòng kiểm tra lại đường dẫn!")
        return

    # 1. Đọc dữ liệu mapping và nhãn
    df_map = pd.read_csv(mapping_csv)
    df_lab = pd.read_csv(labels_csv)

    # 2. Kiểm tra và tự động ghép nối với temporal_groups để lấy tên 4 file đầu vào
    temporal_csv = mapping_csv.parent / mapping_csv.name.replace("target_mapping", "temporal_groups")
    if temporal_csv.exists():
        print(f"-> Đã tìm thấy và tự động nạp 4 file đầu vào từ: {temporal_csv.name}")
        df_temp = pd.read_csv(temporal_csv)
        input_cols = ["group_id"] + [c for c in ["file_t0", "file_t3", "file_t10", "file_t13"] if c in df_temp.columns]
        df_map = pd.merge(df_map, df_temp[input_cols], on="group_id", how="left")
    else:
        print(f"-> Không tìm thấy {temporal_csv.name}, tiếp tục xử lý với các cột hiện có.")

    # 3. Tạo từ điển tra cứu nhanh: filename -> weather_class và weighted_dbz
    label_dict = dict(zip(df_lab["filename"], df_lab["weather_class"]))
    dbz_dict = dict(zip(df_lab["filename"], df_lab["weighted_dbz"]))

    # 4. Ánh xạ nhãn và giá trị dBZ cho từng forecast horizon
    for horizon in ["0h", "1h", "2h", "3h"]:
        target_file_col = f"target_{horizon}_file"
        df_map[f"label_{horizon}"] = df_map[target_file_col].map(label_dict)
        df_map[f"dbz_{horizon}"] = df_map[target_file_col].map(dbz_dict)

    # 5. Sắp xếp lại thứ tự các cột cho trực quan
    core_cols = ["group_id", "input_time"]
    # Thêm các cột file đầu vào nếu có
    for col in ["file_t0", "file_t3", "file_t10", "file_t13"]:
        if col in df_map.columns:
            core_cols.append(col)

    # Thêm các cột file mục tiêu và nhãn
    for horizon in ["0h", "1h", "2h", "3h"]:
        core_cols.extend([f"target_{horizon}_file", f"label_{horizon}", f"dbz_{horizon}"])

    # Giữ lại các cột phụ trợ còn lại
    remaining_cols = [c for c in df_map.columns if c not in core_cols]
    final_cols = core_cols + remaining_cols
    df_output = df_map[final_cols]

    # 6. Lưu file kết quả
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_output.to_csv(output_csv, index=False)

    print(f"\n-> Đã xuất thành công Dataset Metadata: {output_csv}")
    print(f"-> Tổng số mẫu nhóm: {len(df_output)} groups")
    print("\n5 dòng đầu tiên:")
    preview_cols = ["group_id", "input_time", "label_0h", "label_1h", "label_2h", "label_3h"]
    print(df_output[preview_cols].head())


if __name__ == "__main__":
    mapping_path = Path("outputs/metadata/target_mapping_2025-08-01.csv")
    labels_path = Path("outputs/metadata/scan_labels_2025-08-01.csv")
    out_path = Path("outputs/metadata/dataset_2025-08-01.csv")

    build_labeled_dataset(mapping_path, labels_path, out_path)