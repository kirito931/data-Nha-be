"""
================================================================================
MODULE HÀM MẤT MÁT CHUYÊN DỤNG (LOSS FUNCTIONS CHO CLASS IMBALANCE)
================================================================================
Mục đích:
- Khắc phục hiện tượng mất cân bằng lớp (Class Imbalance) trong dữ liệu radar thời tiết trạm Nhà Bè.
  * Lớp đa số: Mưa nhỏ (829 mẫu ~ 37%), Mưa vừa (637 mẫu ~ 29%), Trời quang (447 mẫu ~ 20%).
  * Lớp thiểu số nhưng nguy hiểm: Mưa to (302 mẫu ~ 14%), Mưa rất to / Dông (5 mẫu ~ 0.2%).
- Cung cấp 2 cơ chế tiên tiến:
  1. compute_class_weights(): Tính trọng số nghịch đảo tần suất có làm mượt (Square-root Inverse Frequency).
  2. FocalLoss: Hàm mất mát Focal Loss (Lin et al. - ICCV 2017) tập trung học các mẫu khó phân biệt,
     triệt tiêu gradient từ các mẫu dễ đoán.
================================================================================
"""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

# Thiết lập mã hóa UTF-8 cho terminal Windows tránh lỗi UnicodeEncodeError cp1252
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def compute_class_weights(
    metadata_csv: str | Path = "outputs/metadata/dataset_2025-08.csv",
    horizon: str = "2h",
    method: str = "sqrt_inv",
    max_weight: float = 4.0
) -> torch.Tensor:
    """
    Tính toán vector trọng số (Class Weights) cho 5 lớp thời tiết dựa trên tần suất mẫu.
    
    Tham số:
        metadata_csv: Đường dẫn file CSV mục lục tập dữ liệu.
        horizon: Mốc thời gian dự báo ('0h', '1h', '2h', '3h').
        method: Phương pháp tính trọng số:
                - 'sqrt_inv': Căn bậc hai nghịch đảo tần suất (Khuyến nghị, ổn định nhất).
                - 'inv': Nghịch đảo tần suất đơn thuần (N / (K * N_c)).
        max_weight: Ngưỡng trần cắt (clipping) để tránh bùng nổ gradient đối với lớp cực hiếm.
        
    Trả về:
        torch.Tensor: Vector trọng số kích thước (num_classes,) kiểu Float32.
    """
    df = pd.read_csv(metadata_csv)
    col_name = f"label_{horizon}"
    
    if col_name not in df.columns:
        raise ValueError(f"Không tìm thấy cột nhãn {col_name} trong file {metadata_csv}!")
        
    label_map = {
        "Clear": 0,
        "Light rain": 1,
        "Moderate rain": 2,
        "Heavy rain": 3,
        "Very heavy rain": 4
    }
    
    # Đếm số lượng mẫu của từng lớp
    counts = np.zeros(5, dtype=np.float32)
    series = df[col_name].dropna()
    for label_str, class_id in label_map.items():
        counts[class_id] = (series == label_str).sum()
        
    # Xử lý an toàn nếu có lớp không xuất hiện mẫu nào
    counts = np.maximum(counts, 1.0)
    max_count = np.max(counts)
    
    if method == "sqrt_inv":
        # Làm mượt bằng căn bậc hai: w_c = sqrt(N_max / N_c)
        weights = np.sqrt(max_count / counts)
    elif method == "inv":
        # Nghịch đảo tuyến tính: w_c = N_max / N_c
        weights = max_count / counts
    else:
        weights = np.ones(5, dtype=np.float32)
        
    # Giới hạn ngưỡng trần clipping để bảo vệ độ ổn định khi huấn luyện
    weights = np.clip(weights, a_min=0.5, a_max=max_weight)
    
    # Chuẩn hóa sao cho trung bình cộng của các trọng số bằng 1.0
    weights = weights / np.mean(weights)
    
    weight_tensor = torch.tensor(weights, dtype=torch.float32)
    return weight_tensor


class FocalLoss(nn.Module):
    """
    Cài đặt hàm mất mát Focal Loss cho bài toán phân loại đa lớp (Multi-class Classification).
    Dựa trên bài báo: "Focal Loss for Dense Object Detection" (Tsung-Yi Lin et al., ICCV 2017).
    
    Công thức:
        FL(p_t) = - alpha_t * (1 - p_t)^gamma * log(p_t)
        
    Trong đó:
        - p_t: Xác suất mô hình dự đoán đúng vào nhãn thực tế.
        - gamma (Focusing Parameter): Hệ số tập trung (mặc định = 2.0). Mẫu càng dễ đoán (p_t lớn),
          thì hệ số (1 - p_t)^gamma càng triệt tiêu về 0, ép mạng phải tập trung vào các mẫu khó (p_t nhỏ).
        - alpha: Vector trọng số điều hòa giữa các lớp (Class Weights).
    """
    def __init__(
        self,
        alpha: torch.Tensor | None = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = "mean"
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction
        
    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Tham số:
            inputs: Logits dự đoán của mô hình, kích thước (Batch_Size, Num_Classes).
            targets: Nhãn thực tế, kích thước (Batch_Size,).
        """
        # Tính Cross Entropy Loss tiêu chuẩn theo từng mẫu (không reduction)
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha.to(inputs.device) if self.alpha is not None else None,
            label_smoothing=self.label_smoothing,
            reduction="none"
        )
        
        # Tính xác suất dự đoán p_t từ cross entropy: p_t = exp(-ce_loss)
        p_t = torch.exp(-ce_loss)
        
        # Tính hệ số điều chế Focal: (1 - p_t)^gamma
        focal_factor = (1.0 - p_t) ** self.gamma
        
        # Focal Loss của từng mẫu
        focal_loss = focal_factor * ce_loss
        
        if self.reduction == "mean":
            return focal_loss.mean()
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:
            return focal_loss


# ==============================================================================
# KHỐI TEST KIỂM TRA ĐỘC LẬP
# ==============================================================================
if __name__ == "__main__":
    print("=" * 70)
    print("KIỂM TRA TÍNH TOÁN TRỌNG SỐ LỚP VÀ HÀM MẤT MÁT (LOSS FUNCTIONS)")
    print("=" * 70)
    
    weights = compute_class_weights(horizon="2h", method="sqrt_inv", max_weight=4.0)
    labels = ["Clear", "Light rain", "Moderate rain", "Heavy rain", "Very heavy rain"]
    
    print("\n1. Trọng số lớp tính được (Square-root Inverse Frequency, Normalized):")
    for i, (lbl, w) in enumerate(zip(labels, weights.numpy())):
        print(f"   - Lớp {i} ({lbl:15s}): Trọng số = {w:.4f}")
        
    # Giả lập mini-batch 4 mẫu
    dummy_logits = torch.randn(4, 5, requires_grad=True)
    dummy_targets = torch.tensor([0, 1, 3, 3])  # Mẫu 3 là Heavy rain
    
    # Test Weighted CrossEntropy
    criterion_weighted = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    loss_ce = criterion_weighted(dummy_logits, dummy_targets)
    loss_ce.backward()
    print(f"\n2. Test Weighted CrossEntropy Loss thành công: {loss_ce.item():.4f}")
    
    # Test Focal Loss
    criterion_focal = FocalLoss(alpha=weights, gamma=2.0, label_smoothing=0.1)
    loss_fl = criterion_focal(dummy_logits, dummy_targets)
    print(f"3. Test Focal Loss thành công               : {loss_fl.item():.4f}")
    print("=" * 70)
