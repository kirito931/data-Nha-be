"""
================================================================================
HUẤN LUYỆN NÂNG CẤP VỚI HÀM MẤT MÁT CHỐNG MẤT CÂN BẰNG LỚP (CLASS IMBALANCE)
================================================================================
Mục đích:
- Huấn luyện mô hình ConvNeXt-B Pretrained ImageNet-22k (mốc +2h) giải quyết triệt để
  hiện tượng mất cân bằng lớp (Trời quang & Mưa nhỏ chiếm 60-80%, Mưa to & Dông chiếm thiểu số).
- CUNG CẤP 2 CƠ CHẾ LOSS LỰA CHỌN QUA CLI:
    1. --loss focal: Focal Loss (gamma = 2.0) giúp mô hình dồn sự chú ý vào các mẫu khó (mưa dông),
       triệt tiêu gradient từ các mẫu dễ đoán (trời quang).
    2. --loss weighted_ce: Weighted CrossEntropy với trọng số nghịch đảo tần suất có làm mượt,
       phạt nặng hơn khi mô hình bỏ sót các cơn mưa to nguy hiểm.
- GIỮ NGUYÊN TOÀN BỘ CẤU HÌNH BÀI BÁO ĐỂ ĐỐI CHỨNG KHOA HỌC (ABLATION STUDY):
    * Backbone: convnext_base.fb_in22k (Meta AI)
    * Stochastic Depth: drop_path_rate = 0.2
    * Head init scale = 0.001, Stem 12 kênh
    * Data Transformation: RandAugment + Median Blur 5x5 + Auto Contrast + NRD-1 Normalization
    * Tối ưu GPU RTX 4050: AMP FP16, batch_size = 12, accum_steps = 3 (Effective Batch = 36), num_workers = 4.
================================================================================
"""
import argparse
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

from model_in22k import build_convnext_in22k_nowcasting
from radar_dataset import RadarNowcastingDataset, get_nowcasting_split_datasets
from losses import compute_class_weights, FocalLoss


def train_month_in22k_imbalance(epochs: int = 15, loss_type: str = "focal", pretrained: bool = True):
    """
    Hàm thực thi huấn luyện mô hình ConvNeXt-B ImageNet-22k với hàm mất mát chống mất cân bằng lớp.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print(f"KHỞI ĐỘNG HUẤN LUYỆN CONVNEXT-B IMAGENET-22K CHỐNG MẤT CÂN BẰNG TRÊN: {device}")
    print(f"-> Cơ chế Loss được chọn : {loss_type.upper()}")
    print(f"-> Số epoch huấn luyện   : {epochs}")
    if device.type == "cuda":
        print(f"-> Card đồ họa            : {torch.cuda.get_device_name(0)}")
        print(f"-> Tổng VRAM              : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
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

    num_workers = 4
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
    print("\n2. Đang nạp mô hình ConvNeXt-B Pretrained ImageNet-22k...")
    model = build_convnext_in22k_nowcasting(
        num_classes=5,
        in_channels=12,
        head_init_scale=0.001,
        drop_path_rate=0.2,
        pretrained=pretrained
    ).to(device)

    # 4. TÍNH TOÁN TRỌNG SỐ LỚP VÀ THIẾT LẬP HÀM MẤT MÁT (LOSS FUNCTION)
    print("\n3. Đang tính toán trọng số lớp chống mất cân bằng...")
    class_weights = compute_class_weights(metadata_path, horizon=horizon, method="sqrt_inv", max_weight=4.0).to(device)
    labels_text = ["Clear", "Light rain", "Moderate rain", "Heavy rain", "Very heavy rain"]
    for i, (lbl, w) in enumerate(zip(labels_text, class_weights.cpu().numpy())):
        print(f"   - Lớp {i} ({lbl:15s}): Trọng số phạt = {w:.4f}")

    if loss_type == "focal":
        criterion = FocalLoss(alpha=class_weights, gamma=2.0, label_smoothing=0.1)
        suffix = "focal"
    elif loss_type == "weighted_ce":
        criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
        suffix = "weighted_ce"
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        suffix = "standard"

    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-6)
    scaler = GradScaler('cuda')

    # Thư mục lưu checkpoint
    save_dir = Path("outputs/checkpoints")
    save_dir.mkdir(parents=True, exist_ok=True)
    best_model_path = save_dir / f"best_convnext_month_2h_in22k_{suffix}.pth"

    best_val_acc = 0.0
    history = []
    total_start_time = time.time()

    print("\n" + "=" * 75)
    print(f"4. BẮT ĐẦU VÒNG LẶP HUẤN LUYỆN ({loss_type.upper()})")
    print("=" * 75)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # --- A. HUẤN LUYỆN (TRAINING PHASE) ---
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

        # --- B. ĐÁNH GIÁ (VALIDATION PHASE) ---
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

        saved_mark = ""
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), best_model_path)
            saved_mark = f"(*) ĐÃ LƯU BEST MODEL ({suffix})"

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] ({duration:.1f}s, lr={current_lr:.2e}) | "
            f"Train Loss: {epoch_train_loss:.4f} - Acc: {epoch_train_acc:5.2f}% | "
            f"Val Loss: {epoch_val_loss:.4f} - Acc: {epoch_val_acc:5.2f}% | {saved_mark}",
            flush=True
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
    csv_path = save_dir / f"train_history_month_2h_in22k_{suffix}.csv"
    history_df.to_csv(csv_path, index=False)
    print(f"\n-> Đã lưu lịch sử huấn luyện vào: {csv_path}")

    # Vẽ và lưu biểu đồ
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(history_df["epoch"], history_df["train_loss"], 'r-', label="Train Loss")
    ax1.plot(history_df["epoch"], history_df["val_loss"], 'b--', label="Val Loss")
    ax1.set_title(f"Hàm mất mát (Loss: {loss_type.upper()}) - ConvNeXt-B ImageNet-22k")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle=":", alpha=0.6)
    ax1.legend()

    ax2.plot(history_df["epoch"], history_df["train_acc"], 'r-', label="Train Acc")
    ax2.plot(history_df["epoch"], history_df["val_acc"], 'b--', label="Val Acc")
    ax2.set_title(f"Độ chính xác (Accuracy %) - ConvNeXt-B ({loss_type.upper()})")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.grid(True, linestyle=":", alpha=0.6)
    ax2.legend()

    plt.tight_layout()
    plot_path = save_dir / f"learning_curves_month_2h_in22k_{suffix}.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"-> Đã lưu biểu đồ học tập vào: {plot_path}")

    total_time = time.time() - total_start_time
    print("=" * 75)
    print(f"HUẤN LUYỆN HOÀN TẤT TRONG {total_time / 60:.1f} PHÚT! VAL ACC TỐT NHẤT: {best_val_acc:.2f}%")
    print("=" * 75)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện ConvNeXt-B ImageNet-22k với hàm mất mát chống mất cân bằng lớp.")
    parser.add_argument("--epochs", type=int, default=15, help="Số epoch huấn luyện (mặc định 15)")
    parser.add_argument("--loss", type=str, default="focal", choices=["focal", "weighted_ce", "standard"],
                        help="Loại hàm mất mát: 'focal' hoặc 'weighted_ce' (mặc định: focal)")
    args = parser.parse_args()

    train_month_in22k_imbalance(epochs=args.epochs, loss_type=args.loss, pretrained=True)
