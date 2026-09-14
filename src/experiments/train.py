"""
Huấn luyện thử nghiệm mô hình ConvNeXt-B dự báo mưa (Nowcasting 2h)
trên tập dữ liệu ngày 01/08/2025, tối ưu hóa cho GPU RTX 4050 6GB VRAM.
"""
from pathlib import Path
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.amp import autocast, GradScaler

from radar_dataset import RadarNowcastingDataset
from model import build_convnext_nowcasting


def train_pipeline():
    # 1. Cấu hình thiết bị và tham số
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Bắt đầu quy trình huấn luyện trên thiết bị: {device}")
    if device.type == "cuda":
        print(f"Tên card đồ họa: {torch.cuda.get_device_name(0)}")
        torch.backends.cudnn.benchmark = True

    # Siêu tham số tối ưu cho RTX 4050 6GB
    batch_size = 4            # Kích thước lô nhỏ chống tràn VRAM
    accum_steps = 8           # Tích lũy 8 bước (effective batch size = 4 * 8 = 32)
    base_lr = 5e-5            # Learning rate theo bài báo
    weight_decay = 1e-2       # Trọng số phạt theo AdamW
    epochs = 10               # Số epoch chạy thử nghiệm
    horizon = "2h"            # Mốc dự báo đánh giá chính của bài báo

    # Đường dẫn dữ liệu
    metadata_path = Path("outputs/metadata/dataset_2025-08-01.csv")
    ppi_dir = Path("outputs/ppi/2025-08-01")
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 2. Khởi tạo Dataset và chia Train / Validation (80% / 20%)
    full_dataset = RadarNowcastingDataset(metadata_path, ppi_dir, horizon=horizon)
    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    # Phân chia ngẫu nhiên nhưng cố định hạt giống để tái lập kết quả
    generator = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(full_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True)

    print(f"Tổng số mẫu: {total_samples} | Train set: {train_size} | Validation set: {val_size}")

    # 3. Khởi tạo mô hình ConvNeXt-B
    model = build_convnext_nowcasting(num_classes=5, in_channels=12, head_init_scale=0.001)
    model = model.to(device)

    # 4. Hàm mất mát, Bộ tối ưu và Bộ điều chỉnh tốc độ học
    # CrossEntropy kết hợp Label Smoothing = 0.1 theo bài báo
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # AdamW theo thiết lập của bài báo
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)

    # Cosine Annealing Scheduler (T_max = 5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)

    # Bộ chia tỉ lệ gradient cho độ chính xác hỗn hợp FP16
    scaler = GradScaler('cuda')

    best_val_acc = 0.0

    print("\n========== BẮT ĐẦU HUẤN LUYỆN (10 EPOCHS) ==========")
    for epoch in range(1, epochs + 1):
        start_time = time.time()

        # --- Giai đoạn HUẤN LUYỆN (Training) ---
        model.train()
        train_loss = 0.0
        train_correct = 0
        optimizer.zero_grad()

        for step, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            # Chạy forward pass với chuẩn FP16 để tiết kiệm bộ nhớ
            with autocast('cuda'):
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                # Chia loss theo số bước tích lũy để tỉ lệ gradient chuẩn xác
                loss = loss / accum_steps

            # Lan truyền ngược với gradient scaling
            scaler.scale(loss).backward()

            # Tích lũy đủ accum_steps hoặc hết dữ liệu thì cập nhật trọng số
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

        # --- Giai đoạn KIỂM TRA (Validation) ---
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
        elapsed = time.time() - start_time

        # Lưu checkpoint khi mô hình đạt độ chính xác validation cao nhất
        saved_tag = ""
        if epoch_val_acc > best_val_acc:
            best_val_acc = epoch_val_acc
            torch.save(model.state_dict(), checkpoint_dir / "best_convnext_0801.pth")
            saved_tag = "[ĐÃ LƯU BEST]"

        print(f"Epoch [{epoch:02d}/{epochs:02d}] ({elapsed:.1f}s) | "
              f"Train Loss: {epoch_train_loss:.4f}, Acc: {epoch_train_acc:5.1f}% | "
              f"Val Loss: {epoch_val_loss:.4f}, Acc: {epoch_val_acc:5.1f}% | "
              f"LR: {scheduler.get_last_lr()[0]:.2e} {saved_tag}")

        # Dọn dẹp cache VRAM rác sau mỗi epoch
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print("\n-> Hoàn thành thử nghiệm huấn luyện!")
    print(f"-> Độ chính xác Validation tốt nhất: {best_val_acc:.2f}%")
    print(f"-> Trọng số mô hình tốt nhất lưu tại: {checkpoint_dir / 'best_convnext_0801.pth'}")


if __name__ == '__main__':
    train_pipeline()