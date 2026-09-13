"""
Ghép nối bảng nhóm thời gian toàn tháng (target_mapping_2025-08.csv),
chuỗi ảnh đầu vào (temporal_groups_2025-08.csv) và bảng nhãn (scan_labels_2025-08.csv)
để tạo ra Dataset Metadata hoàn chỉnh cho huấn luyện mô hình.
"""
from pathlib import Path
import pandas as pd


def build_month_dataset(mapping_csv: Path, temporal_csv: Path, labels_csv: Path, output_csv: Path):
    print("Đang đọc các bảng dữ liệu...")
    df_map = pd.read_csv(mapping_csv)
    df_temp = pd.read_csv(temporal_csv)
    df_lab = pd.read_csv(labels_csv)

    print(f"-> target_mapping: {len(df_map)} hàng")
    print(f"-> temporal_groups: {len(df_temp)} hàng")
    print(f"-> scan_labels   : {len(df_lab)} hàng")

    # 1. Ghép 4 file đầu vào (file_t0, file_t3, file_t10, file_t13) vào target_mapping
    input_cols = ["group_id", "file_t0", "file_t3", "file_t10", "file_t13"]
    df_merged = pd.merge(df_map, df_temp[input_cols], on="group_id", how="left")

    # 2. Tạo từ điển tra cứu nhanh
    label_dict = dict(zip(df_lab["filename"], df_lab["weather_class"]))
    dbz_dict = dict(zip(df_lab["filename"], df_lab["weighted_dbz"]))

    # 3. Ánh xạ nhãn và giá trị dBZ cho các mốc dự báo
    for horizon in ["0h", "1h", "2h", "3h"]:
        f_col = f"target_{horizon}_file"
        df_merged[f"label_{horizon}"] = df_merged[f_col].map(label_dict)
        df_merged[f"dbz_{horizon}"] = df_merged[f_col].map(dbz_dict)

    # 4. Sắp xếp các cột trọng tâm
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

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df_final.to_csv(output_csv, index=False)

    print(f"\n-> THÀNH CÔNG! Dataset Metadata toàn tháng đã lưu tại: {output_csv}")
    print(f"-> Tổng số mẫu nhóm (groups): {len(df_final)}")
    
    print("\nThống kê số lượng mẫu có nhãn theo từng mốc:")
    for h in ["0h", "1h", "2h", "3h"]:
        valid_cnt = df_final[f"label_{h}"].notna().sum()
        pct = (valid_cnt / len(df_final)) * 100
        print(f"  * {h:3s}: {valid_cnt:4d}/{len(df_final)} mẫu ({pct:5.1f}%)")


if __name__ == "__main__":
    mapping_file = Path("outputs/metadata/target_mapping_2025-08.csv")
    temporal_file = Path("outputs/metadata/temporal_groups_2025-08.csv")
    labels_file = Path("outputs/metadata/scan_labels_2025-08.csv")
    out_file = Path("outputs/metadata/dataset_2025-08.csv")

    build_month_dataset(mapping_file, temporal_file, labels_file, out_file)