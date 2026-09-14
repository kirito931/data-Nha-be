"""
Tính độ phản hồi trung bình có trọng số (X_label) và gán nhãn 5 lớp thời tiết
cho toàn bộ các file radar RAW trong một ngày.
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
    - Chuẩn hóa số mũ để chống tràn số float
    """
    if np.ma.is_masked(reflectivity_matrix):
        valid_vals = reflectivity_matrix.compressed()
    else:
        valid_vals = reflectivity_matrix[~np.isnan(reflectivity_matrix)]

    if len(valid_vals) == 0:
        return 0.0

    # 1. Định nghĩa các khoảng bin 5 dBZ
    edges = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    bin_indices = np.digitize(valid_vals, edges)

    # 2. Tính tỷ lệ phần trăm p_k của từng bin
    n_total = len(valid_vals)
    unique_bins, counts = np.unique(bin_indices, return_counts=True)
    percentages = {b: count / n_total for b, count in zip(unique_bins, counts)}

    # 3. Tính log-weights để chống tràn số (Numerical Stability)
    raw_exponents = np.array([100.0 * (1.0 - percentages[b]) for b in bin_indices], dtype=np.float64)
    max_exp = np.max(raw_exponents)
    normalized_weights = np.power(10.0, raw_exponents - max_exp)

    # 4. Tính trung bình có trọng số
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


def process_day_labels(raw_dir: Path, output_csv_path: Path):
    """
    Quét qua tất cả file RAW trong thư mục ngày và xuất bảng nhãn metadata
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    raw_files = sorted(list(raw_dir.glob("*.RAW*")))
    total_files = len(raw_files)

    print(f"Tìm thấy tổng cộng {total_files} file RAW trong thư mục: {raw_dir.name}")
    if total_files == 0:
        print("Không tìm thấy file nào. Vui lòng kiểm tra lại đường dẫn!")
        return

    records = []
    start_time = time.time()

    for idx, file_path in enumerate(raw_files, start=1):
        try:
            radar = pyart.io.read_sigmet(str(file_path))
            ref_matrix = radar.fields["reflectivity"]["data"]

            # Xác định chế độ quét qua số sweeps
            nsweeps = radar.nsweeps
            mode = "Long Range" if nsweeps <= 4 else "Short Range"

            # Trích xuất timestamp từ tên file chuẩn Nhà Bè (NHByymmddhhmmss)
            fname = file_path.name
            timestamp_str = f"20{fname[3:5]}-{fname[5:7]}-{fname[7:9]} {fname[9:11]}:{fname[11:13]}:{fname[13:15]}"

            # Tính nhãn
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

            # In tiến trình mỗi 20 file để theo dõi
            if idx % 20 == 0 or idx == total_files:
                elapsed = time.time() - start_time
                print(f"[{idx:3d}/{total_files:3d}] Đã xử lý {fname} | {x_label:5.2f} dBZ | {label_class:12s} ({elapsed:.1f}s)")

        except Exception as e:
            print(f"[{idx:3d}/{total_files:3d}] LỖI đọc file {file_path.name}: {e}")

    # Xuất kết quả ra file CSV
    df = pd.DataFrame(records)
    df.to_csv(output_csv_path, index=False)
    print(f"\n-> Đã lưu bảng nhãn tại: {output_csv_path}")

    # Báo cáo tổng kết phân bố nhãn
    print("\n========== PHÂN BỐ NHÃN THỜI TIẾT (NGÀY 01/08/2025) ==========")
    class_counts = df["weather_class"].value_counts()
    for cls, count in class_counts.items():
        pct = (count / len(df)) * 100
        print(f"  * {cls:15s}: {count:3d} mẫu ({pct:5.1f}%)")
    print("==============================================================")


if __name__ == "__main__":
    # Thư mục ngày 01/08/2025
    day_dir = Path(r"D:\Documents\Đồ án\2025 Pro-Raw T08\01")
    
    # Nơi lưu bảng metadata nhãn
    output_csv = Path("outputs/metadata/scan_labels_2025-08-01.csv")

    process_day_labels(day_dir, output_csv)