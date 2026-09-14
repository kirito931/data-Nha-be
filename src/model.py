"""
Khởi tạo và tùy biến mạng ConvNeXt-B cho bài toán Nowcasting:
- Đầu vào: 12 channels (chuỗi 4 ảnh PPI RGB)
- Đầu ra: 5 classes thời tiết
- Khởi tạo Head với tỉ lệ 0.001 theo thiết lập của bài báo
"""
import torch
import torch.nn as nn
from torchvision.models import convnext_base, ConvNeXt_Base_Weights


def build_convnext_nowcasting(num_classes: int = 5, in_channels: int = 12, head_init_scale: float = 0.001):
    """
    Tạo mô hình ConvNeXt-B tùy biến theo đúng đặc tả của bài báo
    """
    print("Đang nạp mô hình ConvNeXt-B pretrained từ ImageNet...")
    weights = ConvNeXt_Base_Weights.DEFAULT
    model = convnext_base(weights=weights)

    # 1. Tùy biến Stem: đổi từ 3 channels -> 12 channels
    old_conv = model.features[0][0]  # Lớp Conv2d(3, 128, kernel_size=4, stride=4)
    out_channels = old_conv.out_channels
    kernel_size = old_conv.kernel_size
    stride = old_conv.stride

    new_conv = nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride
    )

    # Khởi tạo trọng số 12 channels bằng cách sao chép trọng số 3 channels gốc (lặp 4 lần và chia 4)
    with torch.no_grad():
        orig_weight = old_conv.weight.data  # Shape: (128, 3, 4, 4)
        repeated_weight = orig_weight.repeat(1, in_channels // 3, 1, 1) / (in_channels // 3)
        new_conv.weight.data = repeated_weight
        if old_conv.bias is not None:
            new_conv.bias.data = old_conv.bias.data.clone()

    model.features[0][0] = new_conv
    print(f"-> Đã chuyển đổi lớp Stem đầu vào: in_channels={in_channels}, out_channels={out_channels}")

    # 2. Tùy biến Head: đổi lớp Linear cuối thành 5 classes
    in_features = model.classifier[2].in_features
    new_classifier = nn.Linear(in_features, num_classes)

    # Scale trọng số lớp phân loại với hệ số 0.001 theo paper
    with torch.no_grad():
        new_classifier.weight.data.mul_(head_init_scale)
        new_classifier.bias.data.zero_()

    model.classifier[2] = new_classifier
    print(f"-> Đã chuyển đổi lớp Head phân loại: in_features={in_features}, out_classes={num_classes} (head_init_scale={head_init_scale})")

    return model


# ============================================================
# TEST THỬ NGHIỆM FORWARD PASS TRÊN GPU HOẶC CPU
# ============================================================
if __name__ == "__main__":
    # Tự động chọn GPU nếu khả dụng
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Thiết bị đang sử dụng: {device}")
    if device.type == "cuda":
        print(f"Tên GPU: {torch.cuda.get_device_name(0)}")

    # Khởi tạo mô hình
    net = build_convnext_nowcasting(num_classes=5, in_channels=12).to(device)

    # Tạo 1 batch giả lập gồm 4 mẫu (Batch=4, Channels=12, H=224, W=224)
    dummy_input = torch.randn(4, 12, 224, 224, device=device)

    print("\nĐang chạy thử Forward Pass qua mô hình...")
    with torch.no_grad():
        output = net(dummy_input)

    print("\n========== KẾT QUẢ KIỂM TRA MÔ HÌNH ==========")
    print(f"-> Kích thước đầu vào (X) : {dummy_input.shape}")
    print(f"-> Kích thước đầu ra (Logits): {output.shape} (Batch_size, Num_classes)")
    print(f"-> Giá trị dự đoán mẫu:\n{output}")
    print("===============================================")
    print("-> Mô hình hoạt động hoàn hảo, sẵn sàng cho bước huấn luyện!")