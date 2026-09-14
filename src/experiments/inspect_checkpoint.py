"""
Xem cấu trúc các lớp và tham số được lưu trong file .pth
"""
from pathlib import Path
import torch

checkpoint_path = Path("outputs/checkpoints/best_convnext_0801.pth")

if not checkpoint_path.exists():
    print("Chưa tìm thấy file checkpoint!")
else:
    # Nạp file nhị phân vào bộ nhớ
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    
    print(f"-> Nạp thành công file: {checkpoint_path.name}")
    print(f"-> Tổng số lớp có chứa trọng số: {len(state_dict)}")
    
    total_params = sum(p.numel() for p in state_dict.values())
    print(f"-> Tổng số tham số (parameters) đã lưu: {total_params:,} tham số (~{total_params / 1e6:.1f} triệu)")
    
    print("\nSoi thử 5 lớp tiêu biểu bên trong:")
    for idx, (layer_name, tensor) in enumerate(state_dict.items()):
        if idx < 5:
            print(f"  * Lớp: {layer_name:<40s} | Kích thước ma trận: {str(list(tensor.shape)):<20s}")
            
    # Kiểm tra lớp đầu vào Stem và lớp đầu ra Head tùy biến
    print("\nKiểm tra 2 lớp ta đã tùy biến:")
    stem_shape = list(state_dict["features.0.0.weight"].shape)
    head_shape = list(state_dict["classifier.2.weight"].shape)
    print(f"  * Stem Conv2d (Đầu vào 12 kênh): {stem_shape} -> [128 filters, 12 channels, 4x4 kernel]")
    print(f"  * Head Linear (Đầu ra 5 classes) : {head_shape} -> [5 classes, 1024 features]")