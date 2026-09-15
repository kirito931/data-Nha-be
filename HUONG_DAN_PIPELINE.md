# 🧭 CẨM NANG TOÀN DIỆN: THỨ TỰ VÀ QUY TRÌNH XỬ LÝ DỮ LIỆU RADAR NOWCASTING

> **Dành cho sinh viên thực hiện Đồ án Khoa học Dữ liệu (KHDL)**  
> **Đề tài:** Dự báo mưa cực ngắn (*Precipitation Nowcasting*) từ dữ liệu radar thời tiết trạm Nhà Bè bằng mô hình học sâu **ConvNeXt-B**.  
> **Dựa trên bài báo:** *"Radar Image Classification for Rainfall Nowcasting"* (Le Hong Trang, Nguyen Hoang Anh Thu, Nguyen Manh Dan, Nguyen Huy Hoang, Phan Thanh An, Pham Tran Vu - HCMUT, 2025).

---

## 📖 1. Các khái niệm cốt lõi cần hiểu trước khi đọc code

Trước khi mở code, bạn cần hiểu những thuật ngữ khí tượng và radar này:

1. **Radar thời tiết (Weather Radar - Doppler)**: Thiết bị phát chùm sóng điện từ vào khí quyển. Khi sóng gặp các hạt mưa/mây, một phần năng lượng bị phản xạ ngược lại radar.
2. **Độ phản hồi (Reflectivity - ký hiệu $Z$, đơn vị dBZ)**:
   - Đại lượng đo cường độ phản xạ của chùm sóng.
   - Giá trị dBZ càng cao thì mật độ và kích thước hạt mưa càng lớn.
   - Ví dụ: $< 30\text{ dBZ}$ (trời quang/mây nhẹ), $30 - 40\text{ dBZ}$ (mưa nhỏ), $40 - 50\text{ dBZ}$ (mưa vừa/mưa rào), $> 50\text{ dBZ}$ (mưa to, dông sét dữ dội).
3. **File RAW SIGMET (IRIS Format)**: Định dạng dữ liệu nhị phân do hệ thống radar của hãng Vaisala/Sigmet ghi lại. Mỗi file chứa cấu hình quét, tọa độ, và các ma trận số liệu (`reflectivity`, `velocity`, `total_power`, `spectrum_width`).
4. **Sweep (Lát cắt quét)**: Radar quay tròn $360^\circ$ theo góc phương vị tại một góc ngẩng (elevation angle) cố định. 
   - Trạm Nhà Bè thường quét 2 chế độ:
     - **Long Range (Tầm xa)**: 4 sweeps (góc ngẩng thấp $0.5^\circ, 1.5^\circ, ...$). Ma trận có dạng $(1388 \text{ tia}, 500 \text{ cổng cự ly})$.
     - **Short Range (Tầm gần)**: 8 sweeps (nhiều góc ngẩng để quét khối tích). Ma trận có dạng $(2832 \text{ tia}, 500 \text{ cổng cự ly})$.
5. **Ảnh PPI (Plan Position Indicator)**: Hình ảnh radar được chiếu phẳng theo tọa độ cực (khoảng cách và góc phương vị) lên mặt đất 2D, biểu diễn vùng mưa xung quanh trạm radar (bán kính 150 km).
6. **Nowcasting (Dự báo mưa cực ngắn)**: Dự báo tình trạng mưa trong khoảng thời gian rất gần (từ 0 đến 3 giờ tới).
7. **Temporal Group (Nhóm chuỗi thời gian)**: Radar Nhà Bè quét theo chu kỳ: $t_0 \to t_0+3' \to t_0+10' \to t_0+13'$. Chuỗi 4 lần quét liên tiếp này được ghép lại thành đầu vào $X$ gồm 12 kênh ảnh cho mô hình học sâu.
8. **Horizon (Mốc thời gian dự báo tương lai)**:
   - $0\text{h}$: Dự báo thời tiết ngay tại thời điểm quét $t_0$ (Nowcast tức thời).
   - $1\text{h}$: Dùng dữ liệu quá khứ tại $t_0$ để dự báo thời tiết sau đó $1$ tiếng ($t_0 + 60\text{ phút}$).
   - $2\text{h}$: Dự báo sau $2$ tiếng ($t_0 + 120\text{ phút}$).
   - $3\text{h}$: Dự báo sau $3$ tiếng ($t_0 + 180\text{ phút}$).

---

## 🗺️ 2. Bản đồ tổng thể: Thứ tự thực thi 11 bước trong `src/`

Dưới đây là thứ tự mở và chạy các file code trong dự án. **Tuyệt đối không chạy nhảy cóc** vì file sau luôn kế thừa kết quả (output) của file trước:

```
[ DỮ LIỆU GỐC: Thư mục 2025 Pro-Raw T08/ gồm 31 ngày (8.916 file RAW) ]
                                   │
                                   ▼
 [Bước 1] index_month.py           Quét toàn bộ thư mục, lập chỉ mục thời gian & chế độ quét
                                   │   Output: outputs/metadata/radar_index_2025-08.csv
                                   ▼
 [Bước 2] build_temporal_groups_month.py   Gom các chuỗi quét 4 ảnh liên tiếp (chu kỳ 3-7-3 phút)
                                   │   Output: outputs/metadata/temporal_groups_2025-08.csv
                                   ▼
 [Bước 3] generate_labels_month.py         Đọc ma trận dBZ, tính X_label có trọng số & gán 5 lớp
                                   │   Output: outputs/metadata/scan_labels_2025-08.csv
                                   ▼
 [Bước 4] export_month_ppi.py              Dùng Py-ART vẽ và xuất ảnh PPI RGB 224x224 cho từng lần quét
                                   │   Output: outputs/ppi/all_scans/*.png
                                   ▼
 [Bước 5] build_target_mapping_month.py    Tìm file radar tương lai ở các mốc +0h, +1h, +2h, +3h
                                   │   Output: outputs/metadata/target_mapping_2025-08.csv
                                   ▼
 [Bước 6] build_dataset_metadata_month.py  Ghép nối các bảng trên thành Dataset Metadata hoàn chỉnh
                                   │   Output: outputs/metadata/dataset_2025-08.csv
                                   ▼
 ┌─────────────────────────────────┴─────────────────────────────────┐
 │                                                                   │
 ▼                                                                   ▼
[Bước 7] radar_dataset.py                   [Bước 8] model.py / model_in22k.py
(PyTorch Dataset + Data Transformation:     (Tùy biến mạng ConvNeXt-B:
 RandAugment, MedianBlur, AutoContrast,      - model.py: Pretrained 1K (torchvision)
 Normalization theo NRD-1)                   - model_in22k.py: Pretrained 22k (timm))
 │                                           │
 └─────────────────┬─────────────────────────┘
                   │
                   ▼
 [Bước 9] train_month.py /                 Huấn luyện mô hình ConvNeXt-B trên GPU (mốc 2h):
          train_month_in22k.py             - train_month.py: Pretrained 1K
                                           - train_month_in22k.py: Chuẩn bài báo 22k (timm)
                                           Output: outputs/checkpoints/best_convnext_month_2h*.pth
                                                   outputs/checkpoints/train_history_month_2h*.csv
                                            │
                   ┌────────────────────────┴────────────────────────┐
                   ▼                                                 ▼
 [Bước 10] evaluate_month.py                       [Bước 11] compare_horizons_month.py
 (Đánh giá độc lập tập Test mốc 2h,                 (Huấn luyện & kiểm thử cả 4 mốc 0h, 1h, 2h, 3h,
  vẽ Ma trận nhầm lẫn 5 lớp)                         vẽ biểu đồ đối chiếu với bài báo gốc)
  Output: confusion_matrix_month_2h.png              Output: horizon_comparison_month.png
```

---

## 📋 3. Bảng tra cứu nhanh chức năng và Input/Output từng file

| Thứ tự | Tên File | Vai trò cốt lõi | Dữ liệu đầu vào (Input) | Kết quả đầu ra (Output) |
| :---: | :--- | :--- | :--- | :--- |
| **01** | [`index_month.py`](file:///d:/Documents/Đồ án/src/index_month.py) | Đọc metadata 8.916 file RAW, phân loại Long/Short Range | Thư mục `2025 Pro-Raw T08/` | `outputs/metadata/radar_index_2025-08.csv` |
| **02** | [`build_temporal_groups_month.py`](file:///d:/Documents/Đồ án/src/build_temporal_groups_month.py) | Tìm chuỗi 4 lần quét hợp lệ theo chu kỳ thời gian 3-7-3 phút | `radar_index_2025-08.csv` | `outputs/metadata/temporal_groups_2025-08.csv` |
| **03** | [`generate_labels_month.py`](file:///d:/Documents/Đồ án/src/generate_labels_month.py) | Tính độ phản hồi có trọng số $X_{label}$ & gán nhãn 5 lớp | Các file `.RAW*` + Py-ART | `outputs/metadata/scan_labels_2025-08.csv` |
| **04** | [`export_month_ppi.py`](file:///d:/Documents/Đồ án/src/export_month_ppi.py) | Vẽ ảnh phản hồi radar bề mặt (PPI) kích thước $224 \times 224$ | Các file `.RAW*` + Py-ART | Thư mục `outputs/ppi/all_scans/*.png` |
| **05** | [`build_target_mapping_month.py`](file:///d:/Documents/Đồ án/src/build_target_mapping_month.py) | Tìm nhãn thời tiết tương lai tại các mốc 0h, 1h, 2h, 3h | `temporal_groups_2025-08.csv` & `radar_index_2025-08.csv` | `outputs/metadata/target_mapping_2025-08.csv` |
| **06** | [`build_dataset_metadata_month.py`](file:///d:/Documents/Đồ án/src/build_dataset_metadata_month.py) | Ghép 4 file đầu vào và nhãn 4 mốc tương lai thành bảng tổng | Cả 3 file CSV metadata ở trên | `outputs/metadata/dataset_2025-08.csv` |
| **07** | [`radar_dataset.py`](file:///d:/Documents/Đồ án/src/radar_dataset.py) | Lớp Dataset nạp 4 ảnh (12 kênh) + Data Transformation | `dataset_2025-08.csv` và ảnh PPI | PyTorch DataLoader sinh tensor `(B, 12, 224, 224)` |
| **08** | [`model.py`](file:///d:/Documents/Đồ án/src/model.py) <br> *(và [`model_in22k.py`](file:///d:/Documents/Đồ án/src/model_in22k.py))* | Định nghĩa mạng ConvNeXt-B nhận 12 kênh và xuất 5 classes | Trọng số Pretrained ImageNet-1K (`torchvision`) hoặc ImageNet-22k (`timm`) | Mô hình PyTorch `nn.Module` sẵn sàng train |
| **09** | [`train_month.py`](file:///d:/Documents/Đồ án/src/train_month.py) | Vòng lặp huấn luyện tối ưu hóa GPU RTX 4050 (AMP FP16) | Dataset + Mô hình ConvNeXt-B | `best_convnext_month_2h.pth`, file log CSV, biểu đồ Loss |
| **10** | [`evaluate_month.py`](file:///d:/Documents/Đồ án/src/evaluate_month.py) | Đánh giá tập Test độc lập, xuất Classification Report | Checkpoint `.pth` + Test Set CSV | Báo cáo Precision, Recall, F1 + Ma trận nhầm lẫn |
| **11** | [`compare_horizons_month.py`](file:///d:/Documents/Đồ án/src/compare_horizons_month.py) | Tái lập thí nghiệm Table 5 của bài báo cho cả 4 mốc | Toàn bộ pipeline trên 4 mốc | `horizon_comparison_month.csv` & `horizon_comparison_month.png` |

---

## 🔍 4. Giải thích chi tiết bản chất từng bước xử lý

### Bước 1: `index_month.py`
- **Mục đích**: Tên file radar có dạng `NHB250801000007.RAWLTHU`. Trong đó:
  - `NHB`: Trạm Nhà Bè.
  - `250801`: Ngày quét (Năm 2025, Tháng 08, Ngày 01).
  - `000007`: Giờ phút giây (00 giờ 00 phút 07 giây).
  - `RAWLTHU`: Đuôi quy ước chế độ quét của hệ thống SIGMET.
- File này đọc tiêu đề (header) của từng file nhị phân bằng `pyart.io.read_sigmet` để lấy ra `nsweeps` (số lát cắt quét). Nếu `nsweeps <= 4` $\to$ **Long Range**, nếu `nsweeps > 4` $\to$ **Short Range**.

### Bước 2: `build_temporal_groups_month.py`
- **Mục đích**: Trạm radar Nhà Bè hoạt động liên tục nhưng luân phiên 2 chế độ quét theo chu kỳ:
  - $t_0$: Long Range
  - $t_0 + 3\text{ phút}$: Short Range
  - $t_0 + 10\text{ phút}$: Long Range (cách lần 2 khoảng 7 phút)
  - $t_0 + 13\text{ phút}$: Short Range (cách lần 3 khoảng 3 phút)
- Tổng thời gian của chuỗi này là **13 phút**.
- Script dùng thuật toán cửa sổ trượt (*sliding window*), kiểm tra khoảng cách thời gian $\Delta t$ giữa các file:
  - $2.7 \le \Delta t_1 \le 3.2$ phút
  - $6.7 \le \Delta t_2 \le 7.3$ phút
  - $2.7 \le \Delta t_3 \le 3.2$ phút
- Nếu thỏa mãn, 4 file này tạo thành một nhóm thời gian hợp lệ (`group_id`).

### Bước 3: `generate_labels_month.py`
- **Mục đích**: Làm sao máy tính biết một lần quét radar tương ứng với thời tiết gì?
- Ta tính giá trị độ phản hồi trung bình có trọng số:
  $$X_{label} = \frac{\sum w_i \cdot X_i}{\sum w_i}, \quad w_i = 10^{100 \cdot (1 - p_k)}$$
- **Tại sao cần công thức này?**  
  Nếu tính trung bình cộng thông thường, 95% diện tích là trời quang (0-15 dBZ) sẽ kéo trung bình xuống mức 5-10 dBZ, làm mất dấu hoàn toàn các ổ mây dông nguy hiểm (50-60 dBZ). Nhờ trọng số nghịch đảo tần suất $w_i$, bin nào càng ít xuất hiện (mưa cực to) thì trọng số càng nhân lên cực lớn, giúp $X_{label}$ phản ánh trung thực mức độ nguy hiểm của cơn mưa.
- Phân lớp nhãn:
  - $X_{label} < 30 \to$ `Clear` (Trời quang)
  - $30 \le X_{label} < 40 \to$ `Light rain` (Mưa nhỏ)
  - $40 \le X_{label} < 47.5 \to$ `Moderate rain` (Mưa vừa)
  - $47.5 \le X_{label} < 55 \to$ `Heavy rain` (Mưa to)
  - $X_{label} \ge 55 \to$ `Very heavy rain` (Mưa rất to, dông sét dữ dội)

### Bước 4: `export_month_ppi.py`
- **Mục đích**: Chuyển ma trận số liệu radar dạng tọa độ cực thành ảnh màu trực quan để đưa vào mạng tích chập (CNN/ConvNeXt).
- Thiết lập theo chuẩn bài báo:
  - Chọn lát cắt quét thấp nhất (`sweep = 0`) vì đây là tầng mây gần mặt đất nhất, quyết định lượng mưa rơi xuống.
  - Bảng màu `NWSRef` (bảng màu chuẩn của Cục Thời tiết Quốc gia Hoa Kỳ NWS).
  - Giới hạn bán kính: $150\text{ km}$ xung quanh trạm Nhà Bè.
  - Resize ảnh về kích thước chuẩn của ConvNeXt: **$224 \times 224$ pixels (3 kênh RGB)**.

### Bước 5 & Bước 6: `build_target_mapping_month.py` & `build_dataset_metadata_month.py`
- **Mục đích**: Bài toán Nowcasting không dự báo thời tiết của hiện tại, mà dự báo thời tiết của **tương lai**.
- Với mỗi nhóm 4 ảnh tại thời điểm $t_0$, script sẽ tìm file radar thực tế cách $t_0$ đúng:
  - $+0\text{h}$ (Nowcast tức thời)
  - $+1\text{h}$ ($t_0 + 60\text{ phút}$)
  - $+2\text{h}$ ($t_0 + 120\text{ phút}$)
  - $+3\text{h}$ ($t_0 + 180\text{ phút}$)
- Sau đó ghép nhãn thời tiết tương lai này vào mẫu để làm **Ground Truth (Nhãn thực tế $y$)** cho mô hình học máy. Kết quả được lưu tại `dataset_2025-08.csv`.

### Bước 7: `radar_dataset.py` (Data Pipeline)
- Nạp 4 file ảnh radar liên tiếp $(t_0, t_3, t_{10}, t_{13})$. Mỗi ảnh có kích thước $(3, 224, 224)$.
- Áp dụng chuỗi biến đổi **Data Transformation** (mục 4.2 của bài báo):
  1. **RandAugment**: Biến đổi hình học ngẫu nhiên giúp mạng chống học vẹt (chỉ dùng khi Train).
  2. **Median Blur (kernel 5×5)**: Lọc các đốm nhiễu hạt (speckle noise) do tín hiệu radar phản xạ từ địa hình hoặc chim chóc.
  3. **Auto Contrast**: Cân bằng độ tương phản giữa vùng đối lưu nóng và lạnh.
  4. **Chuẩn hóa kênh**: Chuẩn hóa theo phân phối chuẩn của dữ liệu radar trạm Nhà Bè (NRD-1):
     - Red: $\mu = 0.9844, \sigma = 0.0641$
     - Green: $\mu = 0.9930, \sigma = 0.0342$
     - Blue: $\mu = 0.9632, \sigma = 0.1163$
- Ghép 4 ảnh theo chiều kênh $\to$ thu được Tensor đầu vào $X$ kích thước **$(12, 224, 224)$**.

### Bước 8: `model.py` & `model_in22k.py` (Kiến trúc ConvNeXt-B)
- ConvNeXt-B là kiến trúc mạng tích chập hiện đại bậc nhất, kết hợp ưu điểm tính toán nhanh của CNN và cơ chế tiếp nhận không gian rộng của Vision Transformer (ViT).
- **2 Phiên bản kiến trúc trong mã nguồn**:
  - **Phương án 1 (`src/model.py` - Baseline hiện tại)**: Sử dụng trọng số Pretrained ImageNet-1K (`IMAGENET1K_V1`, 1.000 classes) tích hợp sẵn trong thư viện `torchvision`. Dùng cho toàn bộ các thực nghiệm và checkpoint đã báo cáo.
  - **Phương án 2 (`src/model_in22k.py` - Chuẩn 100% bài báo gốc)**: Sử dụng trọng số Pretrained ImageNet-22k (`convnext_base.fb_in22k`, 21.841 classes của Meta AI) thông qua thư viện `timm`, tích hợp `drop_path_rate = 0.2` (Stochastic depth) theo Table 2 của bài báo. Mô hình 22k học từ số lượng ảnh gấp 11 lần giúp năng lực trích xuất xoáy mây và đặc trưng khí quyển vượt trội hơn.
- **2 Điểm tùy biến cốt lõi cho bài toán Nowcasting**:
  - Bình thường ConvNeXt nhận ảnh RGB 3 kênh. Nhưng ta đưa vào 4 ảnh liên tiếp (12 kênh) $\to$ Lớp Conv2d đầu tiên (Stem) được mở rộng từ `in_channels=3` lên `in_channels=12`. Trọng số được sao chép từ ImageNet chia cho 4 để giữ nguyên cường độ truyền tín hiệu.
  - Lớp phân loại cuối (Classifier Head) được đổi thành 5 lớp thời tiết và nhân với tỉ lệ `head_init_scale = 0.001` (theo mục 4.2 của bài báo) để bảo đảm huấn luyện ổn định, tránh bùng nổ gradient ở các epoch đầu.

### Bước 9, 10, 11: Huấn luyện, Đánh giá và Đối chiếu (`train`, `evaluate`, `compare`)
- **Tối ưu phần cứng**: Sử dụng kỹ thuật tính toán hỗn hợp nửa độ chính xác (`torch.amp.autocast('cuda')` và `GradScaler`) giúp tiết kiệm VRAM và tăng tốc độ xử lý gấp đôi trên GPU RTX 4050.
- **Hàm mất mát**: `CrossEntropyLoss` kết hợp `label_smoothing = 0.1` để ngăn mô hình tự tin thái quá vào một lớp, giúp tăng tính tổng quát hóa.
- **Scheduler**: `CosineAnnealingLR` hạ learning rate theo dạng hình sin mềm mại, giúp mô hình thoát khỏi các cực tiểu cục bộ (local minima).
- **Đánh giá đa chiều**: Không chỉ nhìn vào Accuracy, ta tính toán **Precision, Recall, Macro F1-Score** và xuất **Ma trận nhầm lẫn (Confusion Matrix)** để phân tích chi tiết xem mô hình có bị nhầm lẫn giữa mưa vừa và mưa to hay không.

---

## 🎯 5. Cách trả lời khi Thầy Cô hỏi vấn đáp

1. **"Tại sao đầu vào lại là 12 kênh mà không phải 3 kênh?"**  
   $\to$ *Dạ thưa Thầy/Cô, bài toán Nowcasting đòi hỏi phải nắm bắt được xu hướng chuyển động và biến đổi của mây mưa theo thời gian. Do đó nhóm sử dụng một chuỗi 4 ảnh quét liên tiếp theo chu kỳ của radar Nhà Bè ($t_0, t_3, t_{10}, t_{13}$). Mỗi ảnh là một ảnh màu RGB 3 kênh, khi ghép nối dọc theo chiều kênh sẽ tạo thành tensor đầu vào có $4 \times 3 = 12$ kênh.*

2. **"Tại sao lại phải tính $X_{label}$ bằng trọng số mũ $10^{100(1-p)}$ thay vì lấy trung bình cộng dBZ?"**  
   $\to$ *Dạ vì trong một lần quét radar, các điểm trời quang hoặc mây loãng (dưới 20 dBZ) chiếm đại đa số diện tích (trên 85-90%). Nếu dùng trung bình cộng đơn thuần, các đám mây dông tích điện cực lớn (50-60 dBZ) sẽ bị kéo tụt giá trị và bị gán nhãn thành trời quang. Trọng số mũ nghịch đảo tần suất giúp khuếch đại sức ảnh hưởng của các vùng mưa dông hiếm gặp, bảo đảm phân loại trung thực mức độ nguy hiểm của thời tiết.*

3. **"Tại sao kết quả mốc 2h đạt ~74-76% trong khi bài báo là ~85%?"**  
   $\to$ *Dạ thưa Thầy/Cô, bài báo gốc sử dụng tập dữ liệu NRD-1 thu thập trong vòng **3 năm** (hơn 200.000 mẫu) và huấn luyện **150 epochs** trên trạm máy tính RTX A6000 (48GB VRAM). Trong khi đó, nhóm em đang thực nghiệm giai đoạn 1 trên dữ liệu **1 tháng** (tháng 08/2025 với 2.220 nhóm) và huấn luyện **15 epochs** trên laptop cá nhân (RTX 4050 6GB VRAM). Dù dữ liệu ít hơn gần 100 lần, mô hình vẫn tái lập đúng xu hướng phân rã độ chính xác theo thời gian ($0\text{h} > 1\text{h} > 2\text{h}$) và đạt F1-Score trên 73%, chứng minh tính đúng đắn của phương pháp luận.*
