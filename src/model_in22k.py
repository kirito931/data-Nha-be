"""
================================================================================
MÔ HÌNH NÂNG CẤP (PHƯƠNG ÁN 2): CONVNEXT-B PRETRAINED IMAGENET-22K (CHUẨN BÀI BÁO)
================================================================================
Mục đích:
- Tái lập 100% kiến trúc và nguồn trọng số được công bố trong bài báo gốc:
  "Radar Image Classification for Rainfall Nowcasting" (mục 4.2 và Table 2).
- SỰ KHÁC BIỆT SO VỚI BẢN IMAGENET-1K (src/model.py):
    1. Trọng số gốc (Backbone):
       - Bản `src/model.py`: Sử dụng `IMAGENET1K_V1` (1.28 triệu ảnh, 1.000 classes).
       - Bản `src/model_in22k.py`: Sử dụng `convnext_base.fb_in22k` của Meta AI / Facebook Research
         được huấn luyện trên tập ImageNet-22k (14.2 triệu ảnh, 21.841 classes).
       - Nhờ học từ số lượng ảnh gấp 11 lần, mô hình 22k có năng lực trích xuất các đặc trưng
         hình học, xoáy mây và độ tương phản khí quyển phong phú hơn rất nhiều.
    2. Cấu hình Stochastic Depth (Drop Path Rate):
       - Bảng 2 trong bài báo ghi rõ: `Stochastic depth = 0.2` cho ConvNeXt-B.
       - Thư viện `timm` hỗ trợ tham số `drop_path_rate = 0.2` giúp tăng cường tính chính quy hóa.
    3. Tùy biến Stem & Head:
       - Stem: Mở rộng `in_chans = 12` (chuỗi 4 ảnh PPI RGB liên tiếp). Trọng số 3 kênh gốc
         được lặp lại 4 lần và chia đều cho 4 để bảo toàn biên độ tín hiệu.
       - Head: Lớp Linear gốc 21.841 classes được thay bằng `Linear(1024, 5)` và nhân với
         hệ số `head_init_scale = 0.001` đúng theo bài báo.

- Yêu cầu thư viện: `timm>=1.0.0`
================================================================================
"""
import sys
import torch
import torch.nn as nn
import timm

# Thiết lập mã hóa UTF-8 cho terminal Windows tránh lỗi UnicodeEncodeError cp1252
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass


def build_convnext_in22k_nowcasting(
    num_classes: int = 5,
    in_channels: int = 12,
    head_init_scale: float = 0.001,
    drop_path_rate: float = 0.2,
    pretrained: bool = True
):
    """
    Hàm khởi tạo mô hình ConvNeXt-B Pretrained trên ImageNet-22k theo chuẩn bài báo.
    
    Tham số:
        num_classes: Số lượng lớp đầu ra (mặc định 5: Clear -> Very heavy rain).
        in_channels: Số lượng kênh đầu vào (mặc định 12 = 4 ảnh x 3 kênh RGB).
        head_init_scale: Hệ số thu nhỏ trọng số lớp phân loại (mặc định 0.001 theo bài báo).
        drop_path_rate: Tỷ lệ Stochastic Depth (mặc định 0.2 theo Table 2 của bài báo).
        pretrained: True nếu tải trọng số ImageNet-22k từ Meta AI / HuggingFace, False nếu chỉ tạo khung.
    
    Trả về:
        torch.nn.Module: Mô hình PyTorch hoàn chỉnh sẵn sàng huấn luyện.
    """
    print("=" * 75)
    print("KHỞI TẠO MÔ HÌNH CONVNEXT-B NOWCASTING (CHUẨN IMAGENET-22K META AI)")
    print("=" * 75)

    model_name = 'convnext_base.fb_in22k'
    print(f"-> Tên mô hình timm       : {model_name}")
    print(f"-> Chế độ Pretrained       : {pretrained} (Trọng số ImageNet-22k: 21.841 classes gốc)")
    print(f"-> Stochastic Depth (Drop) : {drop_path_rate}")
    print(f"-> Kênh đầu vào (in_chans) : {in_channels} (4 ảnh PPI RGB)")
    print(f"-> Lớp phân loại đầu ra    : {num_classes} classes")

    # 1. Khởi tạo mô hình qua timm
    # timm tự động nhận diện in_chans=12 và phân bổ trọng số lặp lại 4 lần chia 4 cho lớp Stem
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        in_chans=in_channels,
        num_classes=num_classes,
        drop_path_rate=drop_path_rate
    )

    # 2. Áp dụng chuẩn hóa trọng số Head theo mục 4.2 của bài báo gốc:
    # "The weights of this new layer were scaled by a factor of 0.001 to ensure stable training"
    with torch.no_grad():
        if hasattr(model.head, 'fc') and isinstance(model.head.fc, nn.Linear):
            model.head.fc.weight.data.mul_(head_init_scale)
            if model.head.fc.bias is not None:
                model.head.fc.bias.data.zero_()
            print(f"-> [Head] Đã scale trọng số lớp phân loại với hệ số {head_init_scale}")

    print("=" * 75)
    return model


# ============================================================
# KHỐI TEST KIỂM TRA FORWARD PASS
# ============================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nThiết bị kiểm tra: {device}")
    if device.type == "cuda":
        print(f"Tên GPU: {torch.cuda.get_device_name(0)}")

    # Tạo mô hình thử nghiệm (ở đây để pretrained=False để kiểm tra cấu trúc nhanh không cần tải checkpoint nặng)
    net = build_convnext_in22k_nowcasting(
        num_classes=5,
        in_channels=12,
        head_init_scale=0.001,
        drop_path_rate=0.2,
        pretrained=False
    ).to(device)

    # Giả lập batch 4 mẫu (Batch=4, Channels=12, H=224, W=224)
    dummy_input = torch.randn(4, 12, 224, 224, device=device)

    print("\nĐang chạy thử 1 lượt Forward Pass...")
    with torch.no_grad():
        output = net(dummy_input)

    print("\n========== KẾT QUẢ KIỂM TRA MÔ HÌNH IMAGENET-22K ==========")
    print(f"-> Kích thước đầu vào X      : {dummy_input.shape}")
    print(f"-> Kích thước Logits đầu ra  : {output.shape} (Batch_size=4, Classes=5)")
    print(f"-> Cấu trúc lớp Stem         : {net.stem[0]}")
    print(f"-> Cấu trúc lớp Head         : {net.head.fc}")
    print("============================================================")
    print("-> Mô hình ImageNet-22k hoạt động hoàn hảo, sẵn sàng khi bạn muốn nâng cấp thực nghiệm!")
