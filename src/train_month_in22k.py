"""
================================================================================
HUẤN LUYỆN NÂNG CẤP (PHƯƠNG ÁN 2): CONVNEXT-B PRETRAINED IMAGENET-22K TRÊN GPU
================================================================================
Mục đích:
- Huấn luyện mô hình ConvNeXt-B với trọng số Pretrained ImageNet-22k của Meta AI
  (convnext_base.fb_in22k) cho mốc thời gian 2 giờ tới (+2h).
- TÁI HIỆN CHUẨN 100% CẤU HÌNH BÀI BÁO GỐC:
    * Backbone: ImageNet-22k (21.841 nhãn gốc, 14.2 triệu ảnh).
    * Stochastic Depth: drop_path_rate = 0.2 (Table 2 của bài báo).
    * Lớp phân loại: head_init_scale = 0.001 (Mục 4.2).
    * Data Transformation: RandAugment + Median Blur 5x5 + Auto Contrast + NRD-1 Normalization.
- TỐI ƯU HÓA KHAI THÁC GPU LAPTOP (NVIDIA RTX 4050 6GB VRAM):
    * batch_size = 12, accum_steps = 3 -> Effective batch size = 36.
    * AMP FP16 (autocast + GradScaler).
    * AdamW: lr = 5e-5, weight_decay = 0.01.
    * CosineAnnealingLR (T_max = 5).
    * CrossEntropyLoss (label_smoothing = 0.1).

- Xuất kết quả ra:
    * Checkpoint: outputs/checkpoints/best_convnext_month_2h_in22k.pth
    * Lịch sử huấn luyện: outputs/checkpoints/train_history_month_2h_in22k.csv
    * Biểu đồ Loss & Acc: outputs/checkpoints/learning_curves_month_2h_in22k.png
================================================================================
"""
from pathlib import Path
import sys
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader

# Thiết lập mã hóa UTF-8 cho terminal Windows tránh lỗi UnicodeEncodeError cp1252
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# Nạp module mô hình ImageNet-22k và Dataset
from model_in22k import build_convnext_in22k_nowcasting
from radar_dataset import RadarNowcastingDataset, get_nowcasting_split_datasets


def train_month_in22k(epochs: int = 15, pretrained: bool = True):
    """
    Hàm thực thi huấn luyện mô hình ConvNeXt-B ImageNet-22k.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"KHỞI ĐỘNG HUẤN LUYỆN CONVNEXT-B IMAGENET-22K TRÊN: {device}")
    if device.type == "cuda":
        print(f"Card đồ họa : {torch.cuda.get_device_name(0)}")
        print(f"Tổng VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True
    print("=" * 75)

    # 1. CẤU HÌNH SIÊU THAM SỐ
    batch_size = 12
    accum_steps = 3
    base_lr = 5e-5
    weight_decay = 1e-2
    horizon = "2h"

    # 2. KHỞI TẠO DATASET VÀ DATALOADER
    metadata_path = Path("outputs/metadata/dataset_2025-08.csv")
    ppi_dir = Path("outputs/ppi/all_scans")

    print("\n1. Đang khởi tạo Dataset (Tích hợp RandAugment + Blur 5x5 + AutoContrast + NRD-1)...")
    train_ds, val_ds, test_ds, test_indices = get_nowcasting_split_datasets(
        metadata_csv=metadata_path,
        ppi_dir=ppi_dir,
        horizon=horizon,
        train_ratio=0.80,
        val_ratio=0.10,
        seed=42
    )

    num_workers = 4  # Sử dụng 4 luồng CPU nạp ảnh song song để tăng tốc gấp 3-4 lần
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda")
    )

    print(f"-> Phân chia mẫu: Train={len(train_ds)}, Val={len(val_ds)}, Test={len(test_ds)}")
    print(f"-> Batch Size={batch_size}, Tích lũy Gradient={accum_steps} (Effective Batch={batch_size * accum_steps})")

    # 3. XÂY DỰNG MÔ HÌNH IMAGENET-22K
    print("\n2. Đang xây dựng mạng ConvNeXt-B Pretrained ImageNet-22k...")
    model = build_convnext_in22k_nowcasting(
        num_classes=5,
        in_channels=12,
        head_init_scale=0.001,
        drop_path_rate=0.2,
        pretrained=pretrained
    ).to(device)

    # 4. THIẾT LẬP LOSS, TỐI ƯU VÀ SCHEDULER
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
    scaler = GradScaler('cuda')

    # Thư mục lưu checkpoint
    save_dir = Path("outputs/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / "best_convnext_month_2h_in22k.pth"

    best_val_acc = 0.0
    history = []
    total_start_time = time.time()

    print("\n" + "=" * 75)
    print("3. BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN (TRAINING LOOP)")
    print("=" * 75)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --- A. GIAI ĐOẠN HUẤN LUYỆN (TRAIN) ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        optimizer.zero_grad(set_to_none=True)

        for step, (inputs, targets) in enumerate(train_loader, start=1):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                scaled_loss = loss / accum_steps

            scaler.scale(scaled_loss).backward()

            if step % accum_steps == 0 or step == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            train_loss += loss.item() * inputs.size(0)
            preds = outputs.argmax(dim=1)
            train_correct += (preds == targets).sum().item()
            train_total += targets.size(0)

        scheduler.step()

        epoch_train_loss = train_loss / train_total
        epoch_train_acc = train_correct / train_total * 100

        # --- B. GIAI ĐOẠN ĐÁNH GIÁ (VALIDATION) ---
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)

                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == targets).sum().item()
                val_total += targets.size(0)

        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total * 100
        current_lr = optimizer.param_groups[0]['lr']
        duration = time.time() - epoch_start

        # Lưu checkpoint tốt nhất
        saved_mark = ""
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            saved_mark = "(*) ĐÃ LƯU BEST MODEL"

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({duration:.1f}s, lr={current_lr:.2e}) | "
            f"Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:5.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:5.2f}% | {saved_mark}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": epoch_train_loss,
            "train_acc": epoch_train_acc,
            "val_loss": epoch_val_loss,
            "val_acc": epoch_val_acc,
            "lr": current_lr,
            "time_sec": duration
        })

    # Lưu lịch sử ra CSV
    history_df = pd.DataFrame(history)
    csv_path = save_dir / "train_history_month_2h_in22k.csv"
    history_df.to_csv(csv_path, index=False)
    print(f"\n-> Đã lưu lịch sử huấn luyện vào: {csv_path}")

    # Vẽ và lưu biểu đồ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history_df["epoch"], history_df["train_loss"], 'r-', label="Train Loss")
    ax1.plot(history_df["epoch"], history_df["val_loss"], 'b--', label="Val Loss")
    ax1.set_title("Hàm mất mát (Loss) - ConvNeXt-B ImageNet-22k")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.plot(history_df["epoch"], history_df["train_acc"], 'r-', label="Train Acc")
    ax2.plot(history_df["epoch"], history_df["val_acc"], 'b--', label="Val Acc")
    ax2.set_title("Độ chính xác (Accuracy %) - ConvNeXt-B ImageNet-22k")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plot_path = save_dir / "learning_curves_month_2h_in22k.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"-> Đã lưu biểu đồ học tập vào: {plot_path}")

    total_time = time.time() - total_start_time
    print("=" * 75)
    print(f"HUẤN LUYỆN HOÀN TẤT TRONG {total_time / 60:.1f} PHÚT! VAL ACC TỐT NHẤT: {best_val_acc:.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    n_epochs = int(sys.argv[1]) if len(sys.argv) > 1 else 15
    train_month_in22k(epochs=n_epochs, pretrained=True)
