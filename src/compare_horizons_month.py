"""
================================================================================
BƯỚC 11 TRONG PIPELINE: HUẤN LUYỆN & ĐỐI CHIẾU ĐA MỐC THỜI GIAN (0h, 1h, 2h, 3h)
================================================================================
Mục đích:
- Tái lập toàn diện thực nghiệm trong BẢNG 5 (Table 5) của bài báo gốc:
  Khảo sát hiệu năng dự báo của mô hình ConvNeXt-B trên cả 4 khoảng thời gian:
    * 0h (Nowcast tức thời)
    * 1h (Sau 1 giờ)
    * 2h (Sau 2 giờ)
    * 3h (Sau 3 giờ)
- QUY LUẬT VẬT LÝ VÀ KHÍ TƯỢNG HỌC CẦN QUAN SÁT:
  Khoảng cách thời gian dự báo càng xa (từ 0h lên 3h) thì sự thay đổi hình thái mây và
  hướng gió càng lớn, dẫn đến tương quan giữa ảnh radar quá khứ và nhãn tương lai giảm dần.
  Do đó, Test Accuracy sẽ có xu hướng giảm dần (0h cao nhất, 1h, 2h, 3h giảm tương ứng).
- TỰ ĐỘNG HÓA HOÀN TOÀN:
    1. Kiểm tra nếu mốc nào đã có checkpoint .pth lưu sẵn thì bỏ qua bước train để tiết kiệm thời gian.
    2. Đánh giá độc lập trên tập Test (10% độc lập).
    3. Trích xuất Loss, Accuracy, Macro F1.
    4. Xuất Bảng thống kê đối chiếu trực tiếp với số liệu công bố trong bài báo gốc.
    5. Vẽ biểu đồ cột đôi so sánh trực quan (Our Model vs Paper).

- Xuất kết quả ra:
    * Bảng thống kê: outputs/checkpoints/horizon_comparison_month.csv
    * Biểu đồ đối chiếu: outputs/checkpoints/horizon_comparison_month.png
================================================================================
"""
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import accuracy_score, f1_score

from model import build_convnext_nowcasting
from radar_dataset import RadarNowcastingDataset, LABEL_MAPPING


def train_and_eval_horizon(horizon: str, epochs: int = 15, device=None):
    """
    Huấn luyện (nếu chưa có checkpoint) và đánh giá độc lập cho 1 mốc thời gian cụ thể.
    """
    print("\n" + "=" * 70)
    print(f"       TIẾN HÀNH XỬ LÝ MỐC THỜI GIAN DỰ BÁO: +{horizon.upper()}")
    print("=" * 70)

    # Cấu hình siêu tham số tối ưu hóa phần cứng
    batch_size = 12
    accum_steps = 3
    base_lr = 5e-5
    weight_decay = 1e-2

    metadata_path = Path("outputs/metadata/dataset_2025-08.csv")
    ppi_dir = Path("outputs/ppi/all_scans")
    checkpoint_dir = Path("outputs/checkpoints")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = checkpoint_dir / f"best_convnext_month_{horizon}.pth"
    test_csv_path = checkpoint_dir / f"test_set_month_{horizon}.csv"

    # 1. Nạp Dataset và chia tỉ lệ chuẩn 80% Train : 10% Val : 10% Test
    full_dataset = RadarNowcastingDataset(metadata_path, ppi_dir, horizon=horizon, apply_paper_transform=False)
    total_samples = len(full_dataset)
    train_size = int(0.80 * total_samples)
    val_size = int(0.10 * total_samples)
    test_size = total_samples - train_size - val_size

    generator = torch.Generator().manual_seed(42)
    train_set, val_set, test_set = random_split(
        full_dataset, [train_size, val_size, test_size], generator=generator
    )

    # Lưu danh sách mẫu tập Test nếu chưa có
    if not test_csv_path.exists():
        df_meta = full_dataset.data.iloc[test_set.indices].copy()
        df_meta.to_csv(test_csv_path, index=False)

    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=True, persistent_workers=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True
    )

    print(f"Tổng số mẫu mốc [{horizon}]: {total_samples} | Train: {train_size} | Val: {val_size} | Test: {test_size}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # 2. Huấn luyện (Nếu đã có file checkpoint lưu sẵn thì bỏ qua để tiết kiệm thời gian)
    if ckpt_path.exists():
        print(f"-> Đã tìm thấy checkpoint có sẵn: {ckpt_path.name}. Bỏ qua bước train!")
    else:
        print(f"-> Bắt đầu huấn luyện mới {epochs} epochs trên GPU...")
        model = build_convnext_nowcasting(num_classes=5, in_channels=12, head_init_scale=0.001)
        model = model.to(device)

        optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5)
        scaler = GradScaler('cuda')

        best_val_acc = 0.0

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            model.train()
            train_loss = 0.0
            train_correct = 0
            optimizer.zero_grad()

            for step, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with autocast('cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets) / accum_steps

                scaler.scale(loss).backward()

                if (step + 1) % accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                train_loss += loss.item() * accum_steps * inputs.size(0)
                preds = torch.argmax(outputs, dim=1)
                train_correct += (preds == targets).sum().item()

            scheduler.step()

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

            ep_val_acc = (val_correct / val_size) * 100.0
            ep_time = time.time() - t0

            status = ""
            if ep_val_acc > best_val_acc:
                best_val_acc = ep_val_acc
                torch.save(model.state_dict(), ckpt_path)
                status = "[ĐÃ LƯU BEST]"

            if epoch % 3 == 0 or epoch == epochs:
                print(f"Epoch [{epoch:02d}/{epochs:02d}] ({ep_time:4.1f}s) | "
                      f"Train Loss: {train_loss/train_size:.4f}, Acc: {train_correct/train_size*100:5.2f}% | "
                      f"Val Loss: {val_loss/val_size:.4f}, Acc: {ep_val_acc:5.2f}% {status}")

            if device.type == "cuda":
                torch.cuda.empty_cache()

    # 3. Đánh giá độc lập trên tập Test Set (10%)
    print(f"-> Đang đánh giá độc lập trên tập Test ({test_size} mẫu)...")
    eval_model = build_convnext_nowcasting(num_classes=5, in_channels=12)
    eval_model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    eval_model = eval_model.to(device)
    eval_model.eval()

    all_preds = []
    all_targets = []
    test_loss = 0.0

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            with autocast('cuda'):
                outputs = eval_model(inputs)
                loss = criterion(outputs, targets)

            test_loss += loss.item() * inputs.size(0)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    test_acc = accuracy_score(all_targets, all_preds) * 100.0
    macro_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0) * 100.0
    avg_test_loss = test_loss / test_size

    print(f"-> KẾT QUẢ TEST [{horizon}]: Loss = {avg_test_loss:.4f} | Accuracy = {test_acc:.2f}% | Macro F1 = {macro_f1:.2f}%")

    return {
        "horizon": horizon,
        "total_samples": total_samples,
        "test_loss": round(avg_test_loss, 4),
        "test_acc": round(test_acc, 2),
        "macro_f1": round(macro_f1, 2)
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 75)
    print("CHƯƠNG TRÌNH SO SÁNH ĐA MỐC THỜI GIAN TOÀN THÁNG (0h, 1h, 2h, 3h)")
    print(f"Phần cứng thực nghiệm: {torch.cuda.get_device_name(0)} (RTX 4050 6GB)")
    print("=" * 75)

    horizons = ["0h", "1h", "2h", "3h"]
    results = []
    global_start = time.time()

    for h in horizons:
        res = train_and_eval_horizon(horizon=h, epochs=15, device=device)
        results.append(res)

    # 1. Hiển thị bảng tổng hợp so sánh trực tiếp với bài báo
    df_res = pd.DataFrame(results)

    # Số liệu công bố của bài báo gốc (ConvNeXt-B, Table 5) để đối chiếu
    paper_benchmarks = {
        "0h": {"paper_loss": 0.2830, "paper_acc": 91.10},
        "1h": {"paper_loss": 0.4653, "paper_acc": 85.04},
        "2h": {"paper_loss": 0.4984, "paper_acc": 84.92},
        "3h": {"paper_loss": 0.4961, "paper_acc": 84.66}
    }
    df_res["paper_loss"] = df_res["horizon"].map(lambda x: paper_benchmarks[x]["paper_loss"])
    df_res["paper_acc"] = df_res["horizon"].map(lambda x: paper_benchmarks[x]["paper_acc"])

    print("\n" + "=" * 80)
    print("      BẢNG ĐỐI CHIẾU KẾT QUẢ ĐA MỐC DỰ BÁO VỚI BÀI BÁO GỐC (TABLE 5)")
    print("=" * 80)
    cols_display = ["horizon", "total_samples", "test_loss", "test_acc", "macro_f1", "paper_loss", "paper_acc"]
    print(df_res[cols_display].to_string(index=False))
    print("=" * 80)

    # Lưu bảng kết quả
    summary_path = Path("outputs/checkpoints/horizon_comparison_month.csv")
    df_res.to_csv(summary_path, index=False)
    print(f"-> Đã lưu bảng thống kê đối chiếu tại: {summary_path}")

    # 2. Vẽ biểu đồ so sánh xu hướng (Our Model vs Paper)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(horizons))
    width = 0.35

    our_accs = df_res["test_acc"].tolist()
    paper_accs = df_res["paper_acc"].tolist()

    bars1 = ax.bar(x - width/2, our_accs, width, label='Mô hình nhóm (Tháng 08/2025 - 15 epochs)', color='#1f77b4', alpha=0.85)
    bars2 = ax.bar(x + width/2, paper_accs, width, label='Bài báo gốc NRD-1 (3 năm - 150 epochs)', color='#ff7f0e', alpha=0.85)

    ax.set_xlabel('Mốc thời gian dự báo (Forecast Horizon)', fontsize=11, fontweight='bold')
    ax.set_ylabel('Độ chính xác Test Accuracy (%)', fontsize=11, fontweight='bold')
    ax.set_title('Đối chiếu độ chính xác theo khoảng cách dự báo (Nowcasting Horizons)\nConvNeXt-B: Radar Nhà Bè 08/2025 vs Bài báo gốc (NRD-1)', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(horizons, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10, loc='lower left')
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    for bar in bars1:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold')

    for bar in bars2:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2.0, yval + 1.2, f"{yval:.1f}%", ha='center', va='bottom', fontsize=9, fontweight='bold', color='#d95f02')

    plt.tight_layout()
    chart_path = Path("outputs/checkpoints/horizon_comparison_month.png")
    plt.savefig(chart_path, dpi=150)
    plt.close(fig)

    total_min = (time.time() - global_start) / 60.0
    print(f"-> Đã lưu biểu đồ đối chiếu tại: {chart_path}")
    print(f"-> Toàn bộ quy trình hoàn tất trong: {total_min:.1f} phút")


if __name__ == '__main__':
    main()