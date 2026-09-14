"""
================================================================================
BƯỚC 2 TRONG PIPELINE: GOM NHÓM CHUỖI QUÉT THỜI GIAN 4 ẢNH (TEMPORAL GROUPING)
================================================================================
Mục đích:
- Bài báo nghiên cứu chỉ ra rằng: Tại mỗi thời điểm, trạm radar Nhà Bè có 4 lần quét
  liên tiếp cách nhau lần lượt 3 phút, 10 phút, 13 phút tính từ mốc t0.
  Cụ thể:
    * t0  : Lần quét 1 (Chế độ Long Range)
    * t3  : Lần quét 2 (Chế độ Short Range)  -> cách t0 khoảng 3 phút (3 - 0 = 3')
    * t10 : Lần quét 3 (Chế độ Long Range)   -> cách t3 khoảng 7 phút (10 - 3 = 7')
    * t13 : Lần quét 4 (Chế độ Short Range)  -> cách t10 khoảng 3 phút (13 - 10 = 3')
  Tổng thời gian của 1 chuỗi hoàn chỉnh là đúng 13 phút.
- Script này sử dụng thuật toán Cửa sổ trượt (Sliding Window):
  Duyệt qua danh sách các lần quét theo thứ tự thời gian, tìm ra chính xác các bộ 4 lần quét
  thỏa mãn cả 2 điều kiện:
    1. Đúng thứ tự chế độ quét: [Long Range, Short Range, Long Range, Short Range]
    2. Đúng dung sai thời gian: delta1 ~ 3', delta2 ~ 7', delta3 ~ 3'
- Xuất kết quả ra: outputs/metadata/temporal_groups_2025-08.csv
================================================================================
"""
from pathlib import Path
import pandas as pd


# Đường dẫn file bảng chỉ mục đầu vào (tạo ra từ Bước 1)
INPUT_PATH = Path("outputs/metadata/radar_index_2025-08.csv")

# Đường dẫn file kết quả nhóm chuỗi thời gian đầu ra
OUTPUT_PATH = Path("outputs/metadata/temporal_groups_2025-08.csv")


# 1. Nạp dữ liệu bảng chỉ mục và bảo đảm định dạng datetime
print("=" * 70)
print("BẮT ĐẦU GOM NHÓM CHUỖI THỜI GIAN (TEMPORAL GROUPING)")
print("=" * 70)

df = pd.read_csv(INPUT_PATH, parse_dates=["timestamp"])

# Sắp xếp lại theo thời gian quan trắc tăng dần và reset chỉ số hàng (index)
df = df.sort_values("timestamp").reset_index(drop=True)
print(f"-> Đã nạp {len(df)} lần quét radar từ bảng chỉ mục.")


# 2. Thuật toán cửa sổ trượt (Sliding Window) kích thước 4
groups = []
i = 0
group_id = 0

# Lặp qua từng vị trí i sao cho còn đủ ít nhất 4 file (i, i+1, i+2, i+3)
while i + 3 < len(df):
    # Lấy khối gồm 4 lần quét liên tiếp
    block = df.iloc[i : i + 4]

    # Danh sách chế độ quét của 4 file
    modes = block["mode"].tolist()

    # Tính khoảng cách thời gian (delta) giữa các lần quét liên tiếp (đơn vị: phút)
    # block["timestamp"].diff() tính t[k] - t[k-1]
    # .iloc[1:] bỏ phần tử đầu tiên (vì diff của phần tử đầu là NaN)
    deltas = (
        block["timestamp"]
        .diff()
        .dt.total_seconds()
        .div(60)
        .iloc[1:]
        .tolist()
    )

    # Điều kiện 1: Kiểm tra đúng chu kỳ luân phiên chế độ quét
    valid_modes = (
        modes == [
            "Long Range",
            "Short Range",
            "Long Range",
            "Short Range"
        ]
    )

    # Điều kiện 2: Kiểm tra dung sai thời gian (cho phép xê dịch nhẹ vài giây do đài radar vận hành)
    # delta[0]: khoảng cách giữa t0 và t3 (~ 3 phút: từ 2.7 đến 3.2 phút)
    # delta[1]: khoảng cách giữa t3 và t10 (~ 7 phút: từ 6.7 đến 7.3 phút)
    # delta[2]: khoảng cách giữa t10 và t13 (~ 3 phút: từ 2.7 đến 3.2 phút)
    valid_time = (
        2.7 <= deltas[0] <= 3.2
        and 6.7 <= deltas[1] <= 7.3
        and 2.7 <= deltas[2] <= 3.2
    )

    # Nếu thỏa mãn cả 2 điều kiện -> Đây là một nhóm chuỗi thời gian hoàn chỉnh
    if valid_modes and valid_time:
        groups.append({
            "group_id": group_id,

            # Lần quét thứ 1 (t0 - mốc thời gian gốc của nhóm)
            "t0": block.iloc[0]["timestamp"],
            "file_t0": block.iloc[0]["filename"],

            # Lần quét thứ 2 (t0 + 3 phút)
            "t3": block.iloc[1]["timestamp"],
            "file_t3": block.iloc[1]["filename"],

            # Lần quét thứ 3 (t0 + 10 phút)
            "t10": block.iloc[2]["timestamp"],
            "file_t10": block.iloc[2]["filename"],

            # Lần quét thứ 4 (t0 + 13 phút)
            "t13": block.iloc[3]["timestamp"],
            "file_t13": block.iloc[3]["filename"],
        })

        group_id += 1
        # Vì nhóm này đã tiêu thụ trọn vẹn 4 lần quét, nhảy cóc 4 bước để sang chuỗi tiếp theo
        i += 4

    else:
        # Nếu không hợp lệ (ví dụ: radar bị mất tín hiệu, bị ngắt quãng, hoặc sai thứ tự mode)
        # dịch chuyển con trỏ 1 bước để dò chuỗi hợp lệ tiếp theo
        i += 1


# 3. Tạo DataFrame và lưu ra tệp CSV
groups_df = pd.DataFrame(groups)
groups_df.to_csv(OUTPUT_PATH, index=False)

print("\n" + "=" * 70)
print("KẾT QUẢ GOM NHÓM CHUỖI THỜI GIAN")
print("=" * 70)
print(f"Tổng số nhóm hợp lệ tìm thấy: {len(groups_df)} nhóm (mỗi nhóm gồm 4 ảnh)")
print(f"Đã lưu bảng nhóm tại: {OUTPUT_PATH}")

print("\n--- 5 NHÓM ĐẦU TIÊN ---")
print(groups_df.head(5)[["group_id", "t0", "t3", "t10", "t13"]].to_string(index=False))

print("\n--- 5 NHÓM CUỐI CÙNG ---")
print(groups_df.tail(5)[["group_id", "t0", "t3", "t10", "t13"]].to_string(index=False))
print("=" * 70)