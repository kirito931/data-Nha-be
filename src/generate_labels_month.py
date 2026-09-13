"""
Tính độ phản hồi trung bình có trọng số (X_label) và gán nhãn 5 lớp thời tiết
cho toàn bộ 8.916 file radar RAW trong tháng 08/2025 (31 ngày).
Có tính năng checkpoint: lưu tiến trình sau mỗi ngày và tự động tiếp tục nếu bị gián đoạn.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyart


def calculate_weighted_reflectivity(reflectivity_matrix) -> float:
    """
    Tính X_label từ ma trận dBZ:
    - Phân thành các bin 5 dBZ (<0, 0-5, ..., 55-60, >=60)
    - Trọng số w_i nghịch đảo tỷ lệ phần trăm p_k
    - Chuẩn hóa số mũ chống tràn số float
    """
    if np.ma.is_masked(reflectivity_matrix):
        valid_vals = reflectivity_matrix.compressed()
    else:
        valid_vals = reflectivity_matrix[~np.isnan(reflectivity_matrix)]

    if len(valid_vals) == 0:
        return 0.0

    edges = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    bin_indices = np.digitize(valid_vals, edges)

    n_total = len(valid_vals)
    unique_bins, counts = np.unique(bin_indices, return_counts=True)
    percentages = {b: count / n_total for b, count in zip(unique_bins, counts)}

    raw_exponents = np.array([100.0 * (1.0 - percentages[b]) for b in bin_indices], dtype=np.float64)
    max_exp = np.max(raw_exponents)
    normalized_weights = np.power(10.0, raw_exponents - max_exp)

    x_label = np.sum(normalized_weights * valid_vals) / np.sum(normalized_weights)
    return float(x_label)


def assign_weather_class(x_label: float) -> str:
    """
    Gán nhãn vào 5 lớp thời tiết theo bài báo
    """
    if x_label < 30.0:
        return "Clear"
    elif 30.0 <= x_label < 40.0:
        return "Light rain"
    elif 40.0 <= x_label < 47.5:
        return "Moderate rain"
    elif 47.5 <= x_label < 55.0:
        return "Heavy rain"
    else:
        return "Very heavy rain"


def process_month_labels(root_dir: Path, output_csv_path: Path):
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Kiểm tra checkpoint nếu đã có kết quả chạy dở
    processed_files = set()
    records = []
    if output_csv_path.exists():
        df_existing = pd.read_csv(output_csv_path)
        processed_files = set(df_existing["filename"].tolist())
        records = df_existing.to_dict("records")
        print(f"-> Tìm thấy file checkpoint cũ. Đã tải {len(processed_files)} file đã xử lý trước đó.")

    # 2. Lấy danh sách 31 thư mục ngày (01 đến 31)
    day_dirs = sorted([d for d in root_dir.iterdir() if d.is_dir() and d.name.isdigit()])
    print(f"Tìm thấy {len(day_dirs)} thư mục ngày trong: {root_dir}")

    total_all_files = sum(len(list(d.glob("*.RAW*"))) for d in day_dirs)
    print(f"Tổng số file cần quét: {total_all_files} files")

    global_start_time = time.time()
    processed_count = len(processed_files)

    for day_idx, day_dir in enumerate(day_dirs, start=1):
        day_raw_files = sorted(list(day_dir.glob("*.RAW*")))
        day_new_records = 0
        day_start = time.time()

        for file_path in day_raw_files:
            fname = file_path.name
            if fname in processed_files:
                continue

            try:
                radar = pyart.io.read_sigmet(str(file_path))
                ref_matrix = radar.fields["reflectivity"]["data"]

                nsweeps = radar.nsweeps
                mode = "Long Range" if nsweeps <= 4 else "Short Range"

                timestamp_str = f"20{fname[3:5]}-{fname[5:7]}-{fname[7:9]} {fname[9:11]}:{fname[11:13]}:{fname[13:15]}"

                x_label = calculate_weighted_reflectivity(ref_matrix)
                label_class = assign_weather_class(x_label)

                records.append({
                    "filename": fname,
                    "timestamp": timestamp_str,
                    "mode": mode,
                    "nsweeps": nsweeps,
                    "weighted_dbz": round(x_label, 2),
                    "weather_class": label_class
                })
                processed_files.add(fname)
                day_new_records += 1
                processed_count += 1

            except Exception as e:
                print(f"\n[LỖI] Không thể đọc file {fname}: {e}")

        # Lưu checkpoint ngay sau khi xử lý xong mỗi ngày
        if day_new_records > 0:
            pd.DataFrame(records).to_csv(output_csv_path, index=False)

        day_elapsed = time.time() - day_start
        total_elapsed = time.time() - global_start_time
        pct_done = (processed_count / total_all_files) * 100 if total_all_files > 0 else 100
        print(f"[Ngày {day_dir.name}/31] Xử lý {day_new_records} file mới ({day_elapsed:.1f}s) | Tổng tiến độ: {processed_count}/{total_all_files} ({pct_done:.1f}%) | Đã chạy: {total_elapsed/60:.1f}m")

    # Hoàn tất
    df_final = pd.DataFrame(records)
    df_final.to_csv(output_csv_path, index=False)
    print(f"\n-> HOÀN TẤT TOÀN THÁNG! Bảng nhãn đã lưu tại: {output_csv_path}")

    print("\n========== PHÂN BỐ NHÃN TOÀN THÁNG 08/2025 ==========")
    class_counts = df_final["weather_class"].value_counts()
    for cls, count in class_counts.items():
        pct = (count / len(df_final)) * 100
        print(f"  * {cls:15s}: {count:5d} mẫu ({pct:5.2f}%)")
    print("======================================================")


if __name__ == "__main__":
    # Thư mục gốc chứa 31 ngày
    root_raw_dir = Path(r"D:\Documents\Đồ án\2025 Pro-Raw T08")
    
    # Nơi lưu bảng nhãn toàn tháng
    output_month_csv = Path("outputs/metadata/scan_labels_2025-08.csv")

    process_month_labels(root_raw_dir, output_month_csv)