"""
Huấn luyện và so sánh hiệu năng của ConvNeXt-B trên 4 mốc dự báo:
0h (tức thời), 1h (3600s), 2h (7200s), 3h (10800s) cho dữ liệu ngày 01/08/2025.
Tối ưu hóa tài nguyên cho GPU RTX 4050 6GB VRAM.
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


def train_single_horizon(horizon: str, epochs: int = 10, device=None):
    print(f"\n==================================================")
    print(f"   BẮT ĐẦU HUẤN LUYỆN MỐC DỰ BÁO: {horizon.upper()}")
    print(f"==================================================")

    batch_size = 4
    accum_steps = 8
    base_lr = 5e-5
    weight_decay = 1e-2

    metadata_path = Path("outputs/metadata/dataset_2025-08-01.csv")
    ppi_dir = Path("outputs/ppi/2025-08-01")
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 1. Nạp Dataset cho mốc dự báo hiện tại
    full_dataset = RadarNowcastingDataset(metadata_path, ppi_dir, horizon=horizon)
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    # 2. Tạo mô hình ConvNeXt-B mới
    model = build_convnext_nowcasting(num_classes=5, in_channels=12, head_init_scale=0.001)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
    scaler = GradScaler('cuda')

    best_val_acc = 0.0
    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        # Huấn luyện
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

        # Đánh giá validation
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

        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            best_val_loss = epoch_val_loss
            ckpt_path = checkpoint_dir / f"best_convnext_0801_{horizon}.pth"
            torch.save(model.state_dict(), ckpt_path)

        if epoch == epochs:
            print(f"-> Hoàn tất {epochs} epochs | Val Acc tốt nhất: {best_val_acc:5.2f}% | Val Loss: {best_val_loss:.4f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    return {
        "horizon": horizon,
        "total_samples": total_samples,
        "train_samples": train_size,
        "val_samples": val_size,
        "best_val_acc": round(best_val_acc, 2),
        "best_val_loss": round(best_val_loss, 4)
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị sử dụng: {device} ({torch.cuda.get_device_name(0)})")

    horizons = ["0h", "1h", "2h", "3h"]
    results = []
    total_start = time.time()

    for h in horizons:
        res = train_single_horizon(horizon=h, epochs=10, device=device)
        results.append(res)

    # 1. Hiển thị bảng tổng kết kết quả
    df_res = pd.DataFrame(results)
    print("\n" + "=" * 65)
    print("      BẢNG SO SÁNH HIỆU NĂNG CÁC MỐC DỰ BÁO (NGÀY 01/08/2025)")
    print("=" * 65)
    print(df_res.to_string(index=False))
    print("=" * 65)

    # Lưu kết quả ra CSV
    summary_csv = Path("outputs/checkpoints/horizon_comparison_0801.csv")
    df_res.to_csv(summary_csv, index=False)
    print(f"-> Đã lưu bảng thống kê tại: {summary_csv}")

    # 2. Vẽ biểu đồ đối chiếu xu hướng
    fig, ax1 = plt.subplots(figsize=(8, 5))

    x_labels = df_res["horizon"].tolist()
    accuracies = df_res["best_val_acc"].tolist()

    color = 'tab:blue'
    ax1.set_xlabel('Mốc thời gian dự báo (Forecast Horizon)', fontsize=11, fontweight='bold')
    ax1.set_ylabel('Độ chính xác Validation (%)', color=color, fontsize=11, fontweight='bold')
    bars = ax1.bar(x_labels, accuracies, color=color, alpha=0.6, width=0.45)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.set_ylim(0, 100)

    # Điền giá trị % lên đầu mỗi cột
    for bar in bars:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 1.5, f"{yval:.1f}%", ha='center', va='bottom', fontweight='bold')

    plt.title('So sánh độ chính xác theo khoảng cách dự báo (Nowcasting Horizons)\nDữ liệu Radar Nhà Bè 01/08/2025', fontsize=12, fontweight='bold')
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()

    chart_path = Path("outputs/checkpoints/horizon_comparison_0801.png")
    plt.savefig(chart_path, dpi=150)
    plt.close(fig)
    print(f"-> Đã lưu biểu đồ trực quan tại: {chart_path}")
    print(f"-> Tổng thời gian chạy toàn bộ 4 mốc: {(time.time() - total_start)/60:.1f} phút")


if __name__ == "__main__":
    main()