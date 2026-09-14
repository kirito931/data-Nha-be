"""
================================================================================
BƯỚC 8 TRONG PIPELINE: KIẾN TRÚC MÔ HÌNH HỌC SÂU CONVNEXT-B NOWCASTING
================================================================================
Mục đích:
- Định nghĩa và khởi tạo mô hình ConvNeXt-B (Pretrained từ ImageNet-1K qua torchvision) được tùy biến
  chuyên biệt cho bài toán Dự báo mưa cực ngắn (Nowcasting).
- LƯU Ý KHOA HỌC QUAN TRỌNG:
  * Trong giai đoạn 1 này, mô hình sử dụng trọng số Pretrained ImageNet-1K (1.000 classes) tích hợp sẵn trong torchvision
    để kiểm chứng tính khả thi của toàn bộ pipeline và tối ưu tài nguyên tính toán.
  * Bài báo gốc sử dụng trọng số ImageNet-22k (21.841 classes của Meta AI). Phiên bản nâng cấp ImageNet-22k
    sử dụng thư viện `timm` được tách riêng tại file: `src/model_in22k.py`.
- TẠI SAO BÀI BÁO LỰA CHỌN CONVNEXT?
  ConvNeXt (Facebook Research / CVPR 2022) là kiến trúc mạng tích chập hiện đại được thiết kế
  lại dựa trên các ưu điểm của Vision Transformer (ViT) và Swin Transformer:
    1. Kích thước nhân tích chập lớn (7x7 Depthwise Conv): Mở rộng vùng tiếp nhận không gian
       (Receptive Field), rất thích hợp để nhận diện các dải mây quy mô lớn trên bản đồ radar.
    2. Cấu trúc Inverted Bottleneck: Tăng độ sâu đặc trưng và giảm chi phí tính toán FLOPs.
    3. Hàm kích hoạt GELU thay cho ReLU: Làm mượt đạo hàm, giúp gradient lan truyền ổn định hơn.
    4. Layer Normalization thay cho Batch Normalization: Tránh phụ thuộc vào batch size nhỏ.

- 2 ĐIỂM TÙY BIẾN CỐT LÕI CHO BÀI TOÁN RADAR:
    1. TÙY BIẾN LỚP ĐẦU VÀO (STEM CONVOLUTION):
       - ConvNeXt gốc nhận ảnh 3 kênh màu RGB.
       - Bài toán của ta nhận chuỗi 4 ảnh liên tiếp (mỗi ảnh 3 kênh RGB) -> Tổng là 12 kênh.
       - Thay lớp `nn.Conv2d(3, 128, kernel_size=4, stride=4)` bằng `nn.Conv2d(12, 128, 4, 4)`.
       - Khởi tạo trọng số 12 kênh: Sao chép bộ trọng số 3 kênh gốc của ImageNet lặp lại 4 lần
         và chia đều cho 4 (W_new = repeat(W_orig, 4) / 4).
         *Lý do:* Giữ cho tổng năng lượng kích hoạt ban đầu của tín hiệu đi vào mạng không bị
         phóng đại lên gấp 4 lần, giúp mô hình hội tụ ổn định ngay từ epoch đầu tiên.

    2. TÙY BIẾN LỚP ĐẦU RA (CLASSIFIER HEAD):
       - Thay lớp Linear 1.000 lớp của ImageNet-1K thành `nn.Linear(1024, 5)` (5 lớp thời tiết).
       - Khởi tạo lại trọng số và nhân với hệ số `head_init_scale = 0.001` (theo mục 4.2 của bài báo).
         *Lý do:* Ngăn ngừa các giá trị ngẫu nhiên ban đầu của lớp Head tạo ra gradient quá lớn
         làm phá vỡ các đặc trưng thị giác quý giá đã được học sẵn trong phần thân Backbone.
================================================================================
"""
import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


def build_convnext_nowcasting(num_classes: int = 5, in_channels: int = 12, head_init_scale: float = 0.001):
    """
    Hàm xây dựng và tùy biến mạng ConvNeXt-B cho bài toán Nowcasting.
    
    Tham số:
        num_classes: Số lượng lớp thời tiết đầu ra (mặc định 5 lớp: Clear -> Very heavy rain).
        in_channels: Số lượng kênh đầu vào (mặc định 12 = 4 ảnh x 3 kênh RGB).
        head_init_scale: Hệ số thu nhỏ trọng số lớp phân loại (mặc định 0.001 theo bài báo).
    
    Trả về:
        torch.nn.Module: Mô hình PyTorch hoàn chỉnh sẵn sàng huấn luyện hoặc suy luận.
    """
    print("=" * 70)
    print("KHỞI TẠO MÔ HÌNH CONVNEXT-B NOWCASTING (IMAGENET-1K)")
    print("=" * 70)
    print("-> Đang tải trọng số Pretrained ImageNet-1K từ torchvision (ConvNeXt_Base_Weights.IMAGENET1K_V1)...")
    weights = ConvNeXt_Base_Weights.DEFAULT
    model = convnext_base(weights=weights)

    # ------------------------------------------------------------
    # 1. TÙY BIẾN LỚP STEM (Đầu vào từ 3 kênh -> 12 kênh)
    # ------------------------------------------------------------
    # model.features[0][0] là lớp tích chập đầu tiên hạ mẫu ảnh 4x4
    old_conv = model.features[0][0]  # Conv2d(3, 128, kernel_size=4, stride=4)
    out_channels = old_conv.out_channels
    kernel_size = old_conv.kernel_size
    stride = old_conv.stride

    # Tạo lớp tích chập mới với in_channels = 12
    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride
    )

    # Sao chép và chuẩn hóa trọng số từ 3 kênh gốc của ImageNet
    with torch.no_grad():
        orig_weight = old_conv.weight.data  # Shape ban đầu: (128, 3, 4, 4)
        repeat_factor = in_channels // 3    # 12 // 3 = 4 lần lặp
        # Lặp lại 4 lần theo chiều kênh (dim=1) và chia đều cho 4
        repeated_weight = orig_weight.repeat(1, repeat_factor, 1, 1) / repeat_factor
        new_conv.weight.data = repeated_weight

        # Sao chép bias (nếu có)
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()

    # Gán lớp Conv mới vào kiến trúc mạng
    model.features[0][0] = new_conv
    print(f"-> [Stem] Đã chuyển đổi lớp Conv đầu vào: 3 channels -> {in_channels} channels (Weights lặp 4 lần / 4)")

    # ------------------------------------------------------------
    # 2. TÙY BIẾN LỚP CLASSIFIER HEAD (Đầu ra 5 classes)
    # ------------------------------------------------------------
    # model.classifier[2] là lớp Linear cuối cùng kết nối với softmax
    in_features = model.classifier[2].in_features  # 1024 chiều đặc trưng
    new_classifier = nn.Linear(in_features, num_classes)

    # Khởi tạo trọng số lớp Head với hệ số tỉ lệ 0.001 theo chuẩn bài báo
    with torch.no_grad():
        new_classifier.weight.data.mul_(head_init_scale)
        new_classifier.bias.data.zero_()

    model.classifier[2] = new_classifier
    print(f"-> [Head] Đã chuyển đổi lớp Linear phân loại: in_features={in_features} -> out_classes={num_classes}")
    print(f"         Áp dụng tỉ lệ khởi tạo head_init_scale = {head_init_scale}")
    print("=" * 70)

    return model


# ============================================================
# KHỐI TEST THỬ NGHIỆM FORWARD PASS (CHẠY THỬ MÔ HÌNH)
# ============================================================
if __name__ == "__main__":
    # Tự động chọn GPU NVIDIA nếu khả dụng
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nThiết bị đang sử dụng để kiểm tra: {device}")
    if device.type == "cuda":
        print(f"Tên GPU: {torch.cuda.get_device_name(0)}")

    # Khởi tạo mô hình
    net = build_convnext_nowcasting(num_classes=5, in_channels=12).to(device)

    # Tạo 1 batch giả lập gồm 4 mẫu (Batch=4, Channels=12, H=224, W=224)
    dummy_input = torch.randn(4, 12, 224, 224, device=device)

    print("\nĐang chạy thử nghiệm 1 lượt Forward Pass...")
    with torch.no_grad():
        output = net(dummy_input)

    print("\n========== KẾT QUẢ KIỂM TRA FORWARD PASS ==========")
    print(f"-> Kích thước đầu vào (X)    : {dummy_input.shape} (Batch, 12 Channels, 224, 224)")
    print(f"-> Kích thước Logits đầu ra : {output.shape} (Batch=4, 5 Classes)")
    print(f"-> Giá trị dự đoán mẫu (Logits):\n{output}")
    print("===================================================")
    print("-> Mô hình ConvNeXt-B hoạt động hoàn hảo, sẵn sàng cho bước huấn luyện!")