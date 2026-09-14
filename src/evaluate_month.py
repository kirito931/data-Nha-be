"""
Đánh giá độc lập mô hình ConvNeXt-B Nowcasting 2h trên tập Test toàn tháng (222 mẫu):
- Tính toán Accuracy, Precision, Recall, Macro F1-score
- Xuất biểu đồ Ma trận nhầm lẫn (Confusion Matrix) 5 lớp thời tiết
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix

from radar_dataset import RadarNowcastingDataset, LABEL_MAPPING
from model import build_convnext_nowcasting


def evaluate_test_set():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"BẮT ĐẦU ĐÁNH GIÁ ĐỘC LẬP TẬP TEST TOÀN THÁNG TRÊN: {device}")
    print("=" * 70)

    checkpoint_path = Path("outputs/checkpoints/best_convnext_month_2h.pth")
    test_meta_path = Path("outputs/checkpoints/test_set_month_2h.csv")
    ppi_dir = Path("outputs/ppi/all_scans")
    output_dir = Path("outputs/checkpoints")

    if not checkpoint_path.exists() or not test_meta_path.exists():
        print("Lỗi: Không tìm thấy file checkpoint hoặc bảng test_set_month_2h.csv!")
        return

    # 1. Nạp Dataset từ danh sách mẫu Test đã lưu
    test_dataset = RadarNowcastingDataset(test_meta_path, ppi_dir, horizon="2h")
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=2)
    print(f"-> Đã nạp {len(test_dataset)} mẫu kiểm tra độc lập (Test Set).")

    # 2. Khởi tạo mô hình và nạp trọng số tối ưu
    model = build_convnext_nowcasting(num_classes=5, in_channels=12)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    print("Đang tiến hành suy luận trên toàn bộ tập Test...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # 3. Tính toán các chỉ số thống kê
    target_names = list(LABEL_MAPPING.keys())
    present_labels = sorted(list(set(all_targets) | set(all_preds)))
    present_names = [target_names[i] for i in present_labels]

    print("\n========== BÁO CÁO PHÂN LOẠI CHI TIẾT TRÊN TẬP TEST ==========")
    report = classification_report(
        all_targets, all_preds, labels=present_labels, target_names=present_names, digits=4, zero_division=0
    )
    print(report)

    # 4. Vẽ biểu đồ Ma trận nhầm lẫn (Confusion Matrix)
    cm = confusion_matrix(all_targets, all_preds, labels=present_labels)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=present_names,
        yticklabels=present_names,
        title="Ma trận nhầm lẫn (Confusion Matrix) - Test Set Tháng 08/2025 (Nowcasting 2h)",
        ylabel="Nhãn thực tế (Ground Truth)",
        xlabel="Nhãn mô hình dự đoán (Predicted Label)"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black",
                fontsize=11, weight='bold'
            )

    plt.tight_layout()
    cm_path = output_dir / "confusion_matrix_month_2h.png"
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)

    print(f"-> Đã lưu biểu đồ Ma trận nhầm lẫn tại: {cm_path}")


if __name__ == '__main__':
    evaluate_test_set()