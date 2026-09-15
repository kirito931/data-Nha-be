"""
================================================================================
BƯỚC 10 TRONG PIPELINE: ĐÁNH GIÁ ĐỘC LẬP TẬP TEST & VẼ MA TRẬN NHẦM LẪN (EVALUATION)
================================================================================
Mục đích:
- Đánh giá khách quan mô hình ConvNeXt-B Nowcasting trên tập Kiểm thử độc lập (Test Set gồm 222 mẫu).
  *Tập Test này hoàn toàn không được mô hình nhìn thấy trong suốt quá trình huấn luyện.*
- CÁC CHỈ SỐ THỐNG KÊ KHOA HỌC:
    1. Accuracy (Độ chính xác tổng thể): Tỷ lệ số mẫu dự đoán đúng trên tổng số mẫu.
    2. Precision (Độ chuẩn xác từng lớp): Khi mô hình cảnh báo "Mưa to", có bao nhiêu % thực sự là mưa to?
       (Tránh báo động giả - False Alarm).
    3. Recall (Độ nhạy từng lớp): Trong tất cả các cơn mưa to thực tế diễn ra, mô hình bắt trúng được bao nhiêu %?
       (Tránh bỏ sót nguy cơ thiên tai - Missed Detection).
    4. Macro F1-Score: Trung bình cộng F1 của cả 5 lớp, không bị chi phối bởi các lớp đông dân (Clear, Light rain).
       Đây là chỉ số quan trọng nhất để đánh giá bài toán mất cân bằng lớp (Class Imbalance).
    5. Confusion Matrix (Ma trận nhầm lẫn): Bảng $5 \times 5$ đối chiếu chéo giữa Nhãn thực tế (Ground Truth)
       và Nhãn mô hình dự đoán (Predicted Label). Giúp phát hiện mô hình hay bị nhầm lẫn ở cặp lớp nào nhất.

- Xuất kết quả ra: outputs/checkpoints/confusion_matrix_month_2h.png
================================================================================
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
from model_in22k import build_convnext_in22k_nowcasting


def evaluate_test_set(
    checkpoint_name: str = "best_convnext_month_2h.pth",
    apply_paper_transform: bool = False,
    is_in22k: bool = False,
    output_cm_name: str = "confusion_matrix_month_2h.png"
):
    """
    Hàm thực thi đánh giá mô hình trên tập Test và vẽ ma trận nhầm lẫn.
    
    Tham số:
        checkpoint_name: Tên file trọng số mô hình cần đánh giá.
        apply_paper_transform: True nếu mô hình được train với Data Transformation mới,
                               False nếu mô hình được train với raw pixels cũ.
        is_in22k: True nếu mô hình là ConvNeXt-B ImageNet-22k (timm).
        output_cm_name: Tên file ảnh lưu Ma trận nhầm lẫn.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 70)
    print(f"BẮT ĐẦU ĐÁNH GIÁ ĐỘC LẬP TẬP TEST TRÊN: {device}")
    print(f"-> File Checkpoint: {checkpoint_name}")
    print(f"-> Áp dụng Data Transformation: {apply_paper_transform}")
    print("=" * 70)

    checkpoint_path = Path("outputs/checkpoints") / checkpoint_name
    test_meta_path = Path("outputs/checkpoints/test_set_month_2h.csv")
    ppi_dir = Path("outputs/ppi/all_scans")
    output_dir = Path("outputs/checkpoints")

    if not checkpoint_path.exists() or not test_meta_path.exists():
        print(f"Lỗi: Không tìm thấy file checkpoint {checkpoint_path} hoặc file test_set_month_2h.csv!")
        return

    # ------------------------------------------------------------
    # 1. NẠP TẬP TEST ĐỘC LẬP
    # ------------------------------------------------------------
    test_dataset = RadarNowcastingDataset(
        test_meta_path,
        ppi_dir,
        horizon="2h",
        is_train=False,
        apply_paper_transform=apply_paper_transform
    )
    test_loader = DataLoader(test_dataset, batch_size=12, shuffle=False, num_workers=2)
    print(f"-> Đã nạp thành công {len(test_dataset)} mẫu kiểm tra độc lập (Test Set).")

    # ------------------------------------------------------------
    # 2. KHỞI TẠO MÔ HÌNH VÀ NẠP TRỌNG SỐ TỐI ƯU
    # ------------------------------------------------------------
    if is_in22k:
        model = build_convnext_in22k_nowcasting(
            num_classes=5, in_channels=12, drop_path_rate=0.2, pretrained=False
        )
    else:
        model = build_convnext_nowcasting(num_classes=5, in_channels=12)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()  # Chuyển mô hình sang chế độ suy luận (tắt Dropout/Stochastic Depth)

    all_preds = []
    all_targets = []

    print("\nĐang tiến hành suy luận dự báo trên toàn bộ tập Test...")
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            # Lấy vị trí lớp có xác suất dự đoán cao nhất (Argmax)
            preds = torch.argmax(outputs, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    # ------------------------------------------------------------
    # 3. TÍNH TOÁN BÁO CÁO PHÂN LOẠI CHI TIẾT (CLASSIFICATION REPORT)
    # ------------------------------------------------------------
    target_names = list(LABEL_MAPPING.keys())
    present_labels = sorted(list(set(all_targets) | set(all_preds)))
    present_names = [target_names[i] for i in present_labels]

    print("\n" + "=" * 75)
    print("BÁO CÁO KẾT QUẢ ĐÁNH GIÁ CHI TIẾT TRÊN TẬP TEST (NOWCASTING +2H)")
    print("=" * 75)
    report = classification_report(
        all_targets, all_preds, labels=present_labels, target_names=present_names, digits=4, zero_division=0
    )
    print(report)
    print("=" * 75)

    # ------------------------------------------------------------
    # 4. VẼ VÀ XUẤT BIỂU ĐỒ MA TRẬN NHẦM LẪN (CONFUSION MATRIX)
    # ------------------------------------------------------------
    cm = confusion_matrix(all_targets, all_preds, labels=present_labels)

    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)

    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=present_names,
        yticklabels=present_names,
        title="Ma trận nhầm lẫn (Confusion Matrix) - Test Set Tháng 08/2025 (Nowcasting +2h)",
        ylabel="Nhãn thực tế quan sát (Ground Truth)",
        xlabel="Nhãn mô hình dự đoán (Predicted Label)"
    )
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Điền số lượng mẫu vào từng ô của ma trận
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
    cm_path = output_dir / output_cm_name
    plt.savefig(cm_path, dpi=150)
    plt.close(fig)

    print(f"-> Đã lưu biểu đồ Ma trận nhầm lẫn tại: {cm_path}")
    print("=" * 75)


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "in22k":
        evaluate_test_set(
            checkpoint_name="best_convnext_month_2h_in22k.pth",
            apply_paper_transform=True,
            is_in22k=True,
            output_cm_name="confusion_matrix_month_2h_in22k.png"
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "focal":
        evaluate_test_set(
            checkpoint_name="best_convnext_month_2h_in22k_focal.pth",
            apply_paper_transform=True,
            is_in22k=True,
            output_cm_name="confusion_matrix_month_2h_in22k_focal.png"
        )
    elif len(sys.argv) > 1 and sys.argv[1] == "weighted_ce":
        evaluate_test_set(
            checkpoint_name="best_convnext_month_2h_in22k_weighted_ce.pth",
            apply_paper_transform=True,
            is_in22k=True,
            output_cm_name="confusion_matrix_month_2h_in22k_weighted_ce.png"
        )
    else:
        new_ckpt = Path("outputs/checkpoints/best_convnext_month_2h_transformed.pth")
        if new_ckpt.exists():
            evaluate_test_set(checkpoint_name=new_ckpt.name, apply_paper_transform=True)
        else:
            evaluate_test_set(checkpoint_name="best_convnext_month_2h.pth", apply_paper_transform=False)