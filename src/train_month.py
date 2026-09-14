"""
Huấn luyện tối đa công suất phần cứng (RTX 4050 6GB + Ryzen 5 7535HS):
- batch_size = 12 (đẩy mức ăn VRAM lên ~5GB)
- accum_steps = 3 (effective batch size = 36)
- num_workers = 4 (4 luồng CPU Ryzen nạp dữ liệu song song)
"""
from pathlib import Path
import time
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split

from model import build_convnext_nowcasting
from radar_dataset import RadarNowcastingDataset


def train_full_month():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"KHỞI ĐỘNG HUẤN LUYỆN TỐI ƯU CÔNG SUẤT GPU TRÊN: {device}")
    if device.type == "cuda":
        print(f"Card đồ họa : {torch.cuda.get_device_name(0)}")
        print(f"Tổng VRAM   : {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        torch.backends.cudnn.benchmark = True
    print("=" * 70)

    # 1. Cấu hình siêu tham số tối đa hóa hiệu năng phần cứng
    batch_size = 12           # Khai thác ~5.0 GB VRAM của RTX 4050
    accum_steps = 3           # 12 * 3 = 36 mẫu (tương đương effective batch của paper)
    base_lr = 5e-5            # Learning rate chuẩn bài báo
    weight_decay = 1e-2
    epochs = 15
    horizon = "2h"

    # Đường dẫn
    metadata_path = Path("outputs/metadata/dataset_2025-08.csv")
    ppi_dir = Path("outputs/ppi/all_scans")
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 2. Khởi tạo Dataset và phân chia 80% : 10% : 10%
    full_dataset = RadarNowcastingDataset(metadata_path, ppi_dir, horizon=horizon)
    total_samples = len(full_dataset)

    train_size = int(0.80 * total_samples)
    val_size = int(0.10 * total_samples)
    test_size = total_samples - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Lưu danh sách mẫu tập test độc lập để đánh giá khách quan
    test_indices = test_set.indices
    df_meta = full_dataset.data.iloc[test_indices].copy()
    df_meta.to_csv(checkpoint_dir / "test_set_month_2h.csv", index=False)

    # DataLoader đa luồng tận dụng CPU Ryzen 5 7535HS
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,          # 4 nhân CPU nạp ảnh song song
        pin_memory=True,        # Khóa bộ nhớ RAM để tăng tốc đẩy sang VRAM
        persistent_workers=True # Giữ luồng CPU sống suốt 15 epochs
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True
    )

    print(f"Tổng số mẫu hợp lệ [2h]: {total_samples}")
    print(f"  * Train set     : {train_size} mẫu (80%)")
    print(f"  * Validation set: {val_size} mẫu (10%)")
    print(f"  * Test set      : {test_size} mẫu (10%)")

    # 3. Khởi tạo mô hình
    model = build_convnext_nowcasting(num_classes=5, in_channels=12, head_init_scale=0.001)
    model = model.to(device)

    # 4. Thiết lập bộ tối ưu
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scaler = GradScaler('cuda')

    best_val_acc = 0.0
    history = {"epoch": [], "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    total_start = time.time()

    print(f"\n{'Epoch':<8} | {'Thời gian':<9} | {'Train Loss':<10} | {'Train Acc':<10} | {'Val Loss':<10} | {'Val Acc':<10} | {'Trạng thái'}")
    print("-" * 80)

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        optimizer.zero_grad()

        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss = loss / accum_steps

            scaler.scale(loss).backward()

            if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            train_loss += loss.item() * accum_steps * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            train_correct += (preds == targets).sum().item()

        scheduler.step()

        epoch_train_loss = train_loss / train_size
        epoch_train_acc = (train_correct / train_size) * 100.0

        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0

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

        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_train_loss)
        history["train_acc"].append(epoch_train_acc)
        history["val_loss"].append(epoch_val_loss)
        history["val_acc"].append(epoch_val_acc)

        status = ""
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), checkpoint_dir / "best_convnext_month_2h.pth")
            status = "[ĐÃ LƯU BEST]"

        print(f"[{epoch:02d}/{epochs:02d}]   | {elapsed:6.1f}s    | {epoch_train_loss:10.4f} | {epoch_train_acc:9.2f}% | {epoch_val_loss:10.4f} | {epoch_val_acc:9.2f}% | {status}")

    total_time = (time.time() - total_start) / 60.0
    print("-" * 80)
    print(f"-> HOÀN TẤT HUẤN LUYỆN! Tổng thời gian: {total_time:.1f} phút")
    print(f"-> Độ chính xác Validation tốt nhất: {best_val_acc:.2f}%")

    df_hist = pd.DataFrame(history)
    df_hist.to_csv(checkpoint_dir / "train_history_month_2h.csv", index=False)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(df_hist["epoch"], df_hist["train_loss"], label="Train Loss", color="tab:red")
    ax1.plot(df_hist["epoch"], df_hist["val_loss"], label="Val Loss", color="tab:orange", linestyle="--")
    ax1.set_title("Hàm mất mát (Loss)", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("CrossEntropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.5)

    ax2.plot(df_hist["epoch"], df_hist["train_acc"], label="Train Acc", color="tab:blue")
    ax2.plot(df_hist["epoch"], df_hist["val_acc"], label="Val Acc", color="tab:green", linestyle="--")
    ax2.set_title("Độ chính xác (Accuracy %)", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout()
    chart_path = checkpoint_dir / "learning_curves_month_2h.png"
    plt.savefig(chart_path, dpi=150)
    plt.close(fig)


if __name__ == '__main__':
    train_full_month()