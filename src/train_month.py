"""
================================================================================
BƯỚC 9 TRONG PIPELINE: HUẤN LUYỆN MÔ HÌNH CONVNEXT-B TRÊN GPU (MỐC 2H)
================================================================================
Mục đích:
- Huấn luyện mô hình ConvNeXt-B dự báo mưa cho mốc thời gian 2 giờ tới (+2h).
- TÍCH HỢP TOÀN BỘ DATA TRANSFORMATION THEO CHUẨN BÀI BÁO:
    * Tập Train: RandAugment (magnitude 9, 2 ops) -> Median Blur 5x5 -> Auto Contrast -> NRD-1 Normalization.
    * Tập Val/Test: Median Blur 5x5 -> Auto Contrast -> NRD-1 Normalization (không bị méo bởi RandAugment).
- TỐI ƯU HÓA KHAI THÁC PHẦN CỨNG (NVIDIA RTX 4050 6GB VRAM + CPU AMD RYZEN):
    1. Batch Size & Gradient Accumulation:
       - batch_size = 12: Đẩy mức tiêu thụ VRAM lên khoảng 4.8 - 5.2 GB (tận dụng tối đa 6GB của RTX 4050).
       - accum_steps = 3: Tích lũy gradient qua 3 mini-batch trước khi cập nhật trọng số.
         => Effective Batch Size = 12 * 3 = 36 mẫu (tương đương thiết lập của bài báo mà không bị tràn VRAM).
    2. PyTorch Mixed Precision (AMP FP16):
       - `torch.amp.autocast('cuda')`: Tính toán các phép nhân ma trận tích chập ở định dạng FP16 (nửa độ chính xác),
         giúp tăng tốc độ xử lý lên gấp 2 lần và giảm một nửa dung lượng bộ nhớ.
       - `GradScaler('cuda')`: Tự động phóng to gradient nhỏ trong FP16 để tránh bị triệt tiêu về 0 (Underflow).
    3. Đa luồng CPU (num_workers = 4, pin_memory = True):
       - 4 luồng CPU Ryzen đọc và biến đổi ảnh song song, khóa bộ nhớ RAM để nạp thẳng sang VRAM GPU mà không bị nghẽn cổ chai.
    4. Bộ tối ưu & Điều chỉnh Learning Rate:
       - AdamW: Tách biệt weight decay (0.01) khỏi gradient update, chống overfitting tốt hơn Adam thông thường.
       - CosineAnnealingLR (T_max = 5): Hạ dần learning rate từ 5e-5 theo đường cong cosine mềm mại, giúp mô hình hội tụ sâu.
       - CrossEntropyLoss(label_smoothing = 0.1): Giúp mô hình không quá bảo thủ với các nhãn ranh giới.

- Xuất kết quả ra:
    * Checkpoint mô hình tốt nhất: outputs/checkpoints/best_convnext_month_2h_transformed.pth
    * Lịch sử huấn luyện: outputs/checkpoints/train_history_month_2h_transformed.csv
    * Biểu đồ đường cong học tập: outputs/checkpoints/learning_curves_month_2h_transformed.png
================================================================================
"""
from pathlib import Path
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Nạp các module tự viết từ Bước 7 và Bước 8
from model import build_convnext_nowcasting
from radar_dataset import RadarNowcastingDataset, get_nowcasting_split_datasets


def train_full_month():
    """
    Hàm thực thi toàn bộ quy trình huấn luyện trên dữ liệu tháng 08/2025.
    """
    # Tự động nhận diện thiết bị GPU NVIDIA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"KHỞI ĐỘNG HUẤN LUYỆN TỐI ƯU CÔNG SUẤT GPU TRÊN: {device}")
    if device.type == "cuda":
        print(f"Card đồ họa : {torch.cuda.get_device_name(0)}")
        print(f"Tổng VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        # Bật benchmark cuDNN để PyTorch tự động tìm thuật toán tích chập nhanh nhất cho phần cứng
        torch.backends.cudnn.benchmark = True
    print("=" * 70)

    # ------------------------------------------------------------
    # 1. CẤU HÌNH SIÊU THAM SỐ (HYPERPARAMETERS)
    # ------------------------------------------------------------
    batch_size = 12           # Kích thước mini-batch nạp vào VRAM mỗi bước
    accum_steps = 3           # Số bước tích lũy gradient (12 * 3 = 36 mẫu / effective batch)
    base_lr = 5e-5            # Tốc độ học cơ sở theo chuẩn bài báo
    weight_decay = 1e-2       # Hệ số suy giảm trọng số L2 chống quá khớp (overfitting)
    epochs = 15               # Số lượt huấn luyện toàn bộ tập dữ liệu
    horizon = "2h"            # Mốc dự báo thời tiết sau 2 tiếng

    # Các đường dẫn thư mục
    metadata_path = Path("outputs/metadata/dataset_2025-08.csv")
    ppi_dir = Path("outputs/ppi/all_scans")
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # 2. KHỞI TẠO BỘ DỮ LIỆU ĐÃ TÍCH HỢP DATA TRANSFORMATION
    # ------------------------------------------------------------
    print("\n-> Đang khởi tạo bộ dữ liệu và phân chia 80% Train : 10% Val : 10% Test...")
    train_set, val_set, test_set, test_indices = get_nowcasting_split_datasets(
        metadata_csv=metadata_path,
        ppi_dir=ppi_dir,
        horizon=horizon,
        train_ratio=0.80,
        val_ratio=0.10,
        seed=42
    )

    train_size = len(train_set)
    val_size = len(val_set)
    test_size = len(test_set)
    total_samples = train_size + val_size + test_size

    # Lưu lại danh sách mẫu tập Test độc lập để dùng khi đánh giá
    base_data = RadarNowcastingDataset(metadata_path, ppi_dir, horizon=horizon, apply_paper_transform=False).data
    df_test_meta = base_data.iloc[test_indices].copy()
    df_test_meta.to_csv(checkpoint_dir / "test_set_month_2h.csv", index=False)

    print(f"Tổng số mẫu hợp lệ mốc [{horizon}]: {total_samples}")
    print(f"  * Tập Huấn luyện (Train set)   : {train_size} mẫu (80%) [Có RandAugment + MedianBlur + Normalization]")
    print(f"  * Tập Thẩm định  (Val set)     : {val_size} mẫu (10%)  [Có MedianBlur + Normalization]")
    print(f"  * Tập Kiểm thử   (Test set)    : {test_size} mẫu (10%)  [Có MedianBlur + Normalization]")

    # Khởi tạo DataLoader đa luồng
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,           # Xáo trộn dữ liệu sau mỗi epoch
        num_workers=4,          # 4 nhân CPU nạp ảnh song song
        pin_memory=True,        # Khóa trang bộ nhớ RAM để nạp thẳng sang GPU siêu tốc
        persistent_workers=True # Giữ luồng CPU sống suốt 15 epochs để tránh tạo lại luồng
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,          # Không xáo trộn khi thẩm định
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    # ------------------------------------------------------------
    # 3. KHỞI TẠO MÔ HÌNH CONVNEXT-B
    # ------------------------------------------------------------
    model = build_convnext_nowcasting(num_classes=5, in_channels=12, head_init_scale=0.001)
    model = model.to(device)

    # ------------------------------------------------------------
    # 4. THIẾT LẬP BỘ TỐI ƯU, SCHEDULER VÀ LOSS FUNCTION
    # ------------------------------------------------------------
    # Hàm mất mát CrossEntropyLoss với Label Smoothing 0.1
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer AdamW tách biệt weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # Cosine Annealing scheduler hạ dần learning rate
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    # Bộ phóng đại gradient cho tính toán hỗn hợp nửa độ chính xác (Mixed Precision)
    scaler = GradScaler('cuda')

    best_val_acc = 0.0
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    total_start = time.time()

    # Đường dẫn lưu checkpoint có Data Transformation mới
    best_ckpt_path = checkpoint_dir / "best_convnext_month_2h_transformed.pth"

    print(f"\n{'Epoch':<8} | {'Thời gian':<9} | {'Train Loss':<10} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'Trạng thái'}")
    print("-" * 85)

    # ------------------------------------------------------------
    # 5. VÒNG LẶP HUẤN LUYỆN CHÍNH (TRAINING LOOP)
    # ------------------------------------------------------------
    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --- GIAI ĐOẠN 1: HUẤN LUYỆN (TRAIN) ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        optimizer.zero_grad()

        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Tự động ép kiểu tính toán ma trận sang FP16
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / accum_steps  # Chia nhỏ loss theo số bước tích lũy

            # Lan truyền ngược với GradScaler
            scaler.scale(loss).backward()

            # Khi đã gom đủ accum_steps hoặc là batch cuối cùng trong epoch -> Cập nhật trọng số
            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()

        # Cập nhật learning rate theo chu kỳ Cosine
        scheduler.step()

        epoch_train_loss = train_loss / train_size
        epoch_train_acc = (train_correct / train_size) * 100.0

        # --- GIAI ĐOẠN 2: THẨM ĐỊNH (VALIDATION) ---
        model.eval()
        val_loss = 0.0
        val_correct = 0

        # Tắt tính toán gradient để tiết kiệm VRAM và tăng tốc thẩm định
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == targets).sum().item()

        epoch_val_loss = val_loss / val_size
        epoch_val_acc = (val_correct / val_size) * 100.0
        elapsed = time.time() - epoch_start

        # Ghi nhận lịch sử học tập
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        # Lưu checkpoint khi đạt độ chính xác Validation cao nhất
        status = ""
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_ckpt_path)
            status = "[ĐÃ LƯU BEST]"

        print(f"[{epoch:02d}/{epochs:02d}]   | {elapsed:6.1f}s    | {epoch_train_loss:10.4f} | {epoch_train_acc:9.2f}% | {epoch_val_loss:10.4f} | {epoch_val_acc:9.2f}% | {status}")

    total_time = (time.time() - total_start) / 60.0
    print("-" * 85)
    print(f"-> HOÀN TẤT HUẤN LUYỆN! Tổng thời gian chạy: {total_time:.1f} phút")
    print(f"-> Độ chính xác Validation cao nhất đạt: {best_val_acc:.2f}%")
    print(f"-> Checkpoint tốt nhất đã lưu tại: {best_ckpt_path}")

    # ------------------------------------------------------------
    # 6. LƯU BẢNG LỊCH SỬ VÀ VẼ ĐƯỜNG CONG HỌC TẬP (LEARNING CURVES)
    # ------------------------------------------------------------
    df_hist = pd.DataFrame(history)
    hist_csv_path = checkpoint_dir / "train_history_month_2h_transformed.csv"
    df_hist.to_csv(hist_csv_path, index=False)
    print(f"-> Đã lưu lịch sử huấn luyện tại: {hist_csv_path}")

    # Vẽ 2 biểu đồ: Loss và Accuracy theo từng Epoch
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    # Đồ thị Loss
    ax1.plot(df_hist["epoch"], df_hist["train_loss"], label="Train Loss", color="tab:red", linewidth=2)
    ax1.plot(df_hist["epoch"], df_hist["val_loss"], label="Val Loss", color="tab:orange", linestyle="--", linewidth=2)
    ax1.set_title("Hàm mất mát (CrossEntropy Loss)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", fontsize=10)
    ax1.set_ylabel("Loss", fontsize=10)
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    # Đồ thị Accuracy
    ax2.plot(df_hist["epoch"], df_hist["train_acc"], label="Train Acc", color="tab:blue", linewidth=2)
    ax2.plot(df_hist["epoch"], df_hist["val_acc"], label="Val Acc", color="tab:green", linestyle="--", linewidth=2)
    ax2.set_title("Độ chính xác (Accuracy %)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", fontsize=10)
    ax2.set_ylabel("Accuracy (%)", fontsize=10)
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = checkpoint_dir / "learning_curves_month_2h_transformed.png"
    plt.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"-> Đã xuất biểu đồ đường cong học tập tại: {chart_path}")
    print("=" * 85)


if __name__ == '__main__':
    train_full_month()