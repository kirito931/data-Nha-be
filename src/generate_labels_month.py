"""
================================================================================
BƯỚC 3 TRONG PIPELINE: TÍNH ĐỘ PHẢN HỒI CÓ TRỌNG SỐ VÀ GÁN NHÃN 5 LỚP THỜI TIẾT
================================================================================
Mục đích:
- Đọc ma trận độ phản hồi (Reflectivity field - dBZ) từ từng file RAW SIGMET bằng Py-ART.
- Tính giá trị phản hồi trung bình có trọng số X_label theo công thức của bài báo:
    
         X_label = sum(w_i * X_i) / sum(w_i)
         với w_i = 10^(100 * (1 - p_k))
    
    Trong đó:
      * X_i: Giá trị độ phản xạ tại cổng đo thứ i (đơn vị dBZ).
      * p_k: Tỷ lệ phần trăm số điểm dữ liệu rơi vào khoảng (bin) 5 dBZ thứ k.
      * w_i: Trọng số của điểm đo đó.

- Ý NGHĨA KHOA HỌC CỦA CÔNG THỨC:
    Trong 1 lần quét radar, 85-95% diện tích là trời quang (0-15 dBZ). Nếu dùng trung bình cộng,
    các vùng trời quang này sẽ kéo sụt giá trị trung bình xuống rất thấp, làm "tàng hình" các
    ổ mây dông nguy hiểm (50-60 dBZ). Trọng số w_i nghịch đảo tỷ lệ p_k sẽ nhân hệ số cực lớn
    cho các đám mây dông hiếm gặp, giúp X_label phản ánh trung thực mức độ nguy hiểm của thời tiết.

- KỸ THUẬT CHỐNG TRÀN SỐ FLOAT (Float Overflow Prevention):
    Vì 100 * (1 - p_k) có thể lên tới 90-100, 10^100 sẽ vượt quá hoặc gần tới giới hạn số mũ float64.
    Code thông minh trừ đi max_exp trước khi tính lũy thừa:
         10^(exp - max_exp)
    Vì cả tử số và mẫu số đều chia cho 10^max_exp nên kết quả trung bình có trọng số không đổi,
    nhưng hoàn toàn triệt tiêu nguy cơ bị vô cực (inf) hay NaN.

- Gán nhãn vào 5 lớp thời tiết theo quy chuẩn:
    * Clear (Trời quang): X_label < 30.0 dBZ
    * Light rain (Mưa nhỏ): 30.0 <= X_label < 40.0 dBZ
    * Moderate rain (Mưa vừa): 40.0 <= X_label < 47.5 dBZ
    * Heavy rain (Mưa to): 47.5 <= X_label < 55.0 dBZ
    * Very heavy rain (Mưa rất to, dông sét): X_label >= 55.0 dBZ

- Xuất kết quả ra: outputs/metadata/scan_labels_2025-08.csv
================================================================================
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyart


def calculate_weighted_reflectivity(reflectivity_matrix) -> float:
    """
    Tính toán độ phản hồi trung bình có trọng số X_label từ ma trận dBZ 2D.
    
    Tham số:
        reflectivity_matrix: Ma trận numpy hoặc ma trận masked array chứa giá trị dBZ.
    
    Trả về:
        float: Giá trị X_label đại diện cho lần quét radar đó.
    """
    # 1. Bóc tách các điểm dữ liệu hợp lệ (loại bỏ các điểm bị masked hoặc NaN do radar không đo được)
    if np.ma.is_masked(reflectivity_matrix):
        valid_vals = reflectivity_matrix.compressed()
    else:
        valid_vals = reflectivity_matrix[~np.isnan(reflectivity_matrix)]

    # Nếu toàn bộ ma trận rỗng hoặc không có dữ liệu hợp lệ
    if len(valid_vals) == 0:
        return 0.0

    # 2. Phân thành 13 khoảng (bins) độ rộng 5 dBZ theo bài báo:
    # Các khoảng: <0, [0, 5), [5, 10), ..., [55, 60), >=60 dBZ
    edges = np.array([0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60])
    bin_indices = np.digitize(valid_vals, edges)

    # 3. Tính tỷ lệ phần trăm (percentage p_k) cho từng bin
    n_total = len(valid_vals)
    unique_bins, counts = np.unique(bin_indices, return_counts=True)
    percentages = {b: count / n_total for b, count in zip(unique_bins, counts)}

    # 4. Tính số mũ trọng số: 100 * (1 - p_k)
    raw_exponents = np.array([100.0 * (1.0 - percentages[b]) for b in bin_indices], dtype=np.float64)

    # Kỹ thuật chuẩn hóa trừ max_exp để chống tràn số float trong Python (Numerical Stability)
    max_exp = np.max(raw_exponents)
    normalized_weights = np.power(10.0, raw_exponents - max_exp)

    # 5. Tính trung bình có trọng số X = sum(w_i * X_i) / sum(w_i)
    x_label = np.sum(normalized_weights * valid_vals) / np.sum(normalized_weights)
    return float(x_label)


def assign_weather_class(x_label: float) -> str:
    """
    Gán nhãn 5 lớp phân loại thời tiết dựa trên giá trị X_label tính được:
        - Dưới 30 dBZ: Clear (Trời quang)
        - Từ 30 đến dưới 40 dBZ: Light rain (Mưa nhỏ)
        - Từ 40 đến dưới 47.5 dBZ: Moderate rain (Mưa vừa)
        - Từ 47.5 đến dưới 55 dBZ: Heavy rain (Mưa to)
        - Từ 55 dBZ trở lên: Very heavy rain (Mưa rất to / Dông sét)
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
    """
    Duyệt qua tất cả các file RAW trong tháng, tính X_label và lưu bảng nhãn.
    Có cơ chế checkpointing: lưu liên tục sau mỗi ngày để không bị mất dữ liệu nếu bị ngắt ngang.
    """
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Kiểm tra checkpoint nếu đã có kết quả chạy trước đó
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
                # Đọc file RAW bằng Py-ART
                radar = pyart.io.read_sigmet(str(file_path))
                ref_matrix = radar.fields["reflectivity"]["data"]

                nsweeps = radar.nsweeps
                mode = "Long Range" if nsweeps <= 4 else "Short Range"

                # Chuẩn hóa chuỗi thời gian timestamp
                timestamp_str = f"20{fname[3:5]}-{fname[5:7]}-{fname[7:9]} {fname[9:11]}:{fname[11:13]}:{fname[13:15]}"

                # Tính độ phản hồi có trọng số và gán lớp thời tiết
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

    # Hoàn tất lưu file cuối cùng
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
    root_raw_dir = Path(r"2025 Pro-Raw T08")

    # Nơi lưu bảng nhãn toàn tháng
    output_month_csv = Path("outputs/metadata/scan_labels_2025-08.csv")

    process_month_labels(root_raw_dir, output_month_csv)