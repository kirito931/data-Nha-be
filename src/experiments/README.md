# 🧪 Early Exploratory & Single-Day Experiments

Thư mục này lưu trữ các mã nguồn thử nghiệm ban đầu và các script khảo sát đơn ngày (Ngày 01/08/2025) trước khi hoàn thiện pipeline toàn diện cho toàn tháng:

- **Các script xử lý riêng cho Ngày 01**:
  - `index_day.py`, `export_day_ppi.py`, `generate_labels.py`, `build_temporal_groups.py`, `build_target_mapping.py`, `build_dataset_metadata.py`, `train.py`, `evaluate.py`, `compare_horizons.py`.
- **Các script kiểm tra & phân tích đặc thù**:
  - `check_circles.py`: Kiểm tra các vòng tròn địa hình trên radar.
  - `check_missing.py`, `check_missing_by_range.py`: Khảo sát điểm khuyết dữ liệu theo cự ly.
  - `find_gaps.py`: Phát hiện các khoảng trống thời gian bất thường giữa các lần quét.
  - `inspect_radar.py`, `plot_ppi.py`: Xem cấu trúc metadata và trực quan hóa mẫu thử nghiệm ban đầu.

> **Lưu ý:** Để chạy pipeline chính thức của đề tài, vui lòng sử dụng các script chuẩn trong thư mục cha `src/` theo hướng dẫn tại [HUONG_DAN_PIPELINE.md](../../HUONG_DAN_PIPELINE.md).
