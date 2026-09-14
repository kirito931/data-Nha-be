"""
Đánh giá chi tiết mô hình ConvNeXt-B đã huấn luyện:
- Tính Accuracy, Precision, Recall, F1-score cho từng lớp
- Vẽ và lưu biểu đồ trực quan Ma trận nhầm lẫn (Confusion Matrix)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report, confusion_matrix

from radar_dataset import RadarNowcastingDataset, LABEL_MAPPING
from model import build_convnext_nowcasting


def evaluate_best_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Đang đánh giá mô hình trên thiết bị: {device}")

    checkpoint_path = Path("outputs/checkpoints/best_convnext_0801.pth")
    if not checkpoint_path.exists():
        print("Lỗi: Không tìm thấy file checkpoint đã lưu!")
        return

    # 1. Nạp lại dữ liệu kiểm thử (Validation set)
    metadata_path = Path("outputs/metadata/dataset_2025-08-01.csv")
    ppi_dir = Path("outputs/ppi/2025-08-01")
    full_dataset = RadarNowcastingDataset(metadata_path, ppi_dir, horizon="2h")

    total_samples = len(full_dataset)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size

    # Dùng đúng seed 42 để lấy đúng tập Validation đã kiểm tra trong lúc train
    generator = torch.Generator().manual_seed(42)
    _, val_set = random_split(full_dataset, [train_size, val_size], generator=generator)
    val_loader = DataLoader(val_set, batch_size=4, shuffle=False)

    # 2. Khởi tạo mạng và nạp bộ trọng số tối ưu từ file .pth
    model = build_convnext_nowcasting(num_classes=5, in_channels=12)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    print("Đang tiến hành suy luận trên tập Validation...")
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    # 3. Tính toán các chỉ số thống kê
    target_names = list(LABEL_MAPPING.keys())
    present_labels = sorted(list(set(all_targets) | set(all_preds)))
    present_names = [target_names[i] for i in present_labels]

    print("\n========== BÁO CÁO ĐÁNH GIÁ CHI TIẾT (CLASSIFICATION REPORT) ==========")
    report = classification_report(all_targets, all_preds, labels=present_labels, target_names=present_names, zero_division=0)
    print(report)

    # 4. Vẽ biểu đồ Ma trận nhầm lẫn (Confusion Matrix)
    cm = confusion_matrix(all_targets, all_preds, labels=present_labels)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=present_names,
        yticklabels=present_names,
        title="Ma trận nhầm lẫn (Confusion Matrix) - Nowcasting 2h",
        ylabel="Nhãn thực tế (Ground Truth)",
        xlabel="Nhãn mô hình dự đoán (Predicted Label)"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Điền con số cụ thể vào từng ô
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=12, weight='bold')

    plt.tight_layout()
    cm_output_path = Path("outputs/checkpoints/confusion_matrix_0801.png")
    plt.savefig(cm_output_path, dpi=150)
    plt.close(fig)

    print(f"-> Đã lưu biểu đồ Ma trận nhầm lẫn tại: {cm_output_path}")


if __name__ == '__main__':
    evaluate_best_model()