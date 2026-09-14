"""
Xây dựng PyTorch Dataset cho bài toán Nowcasting:
Nạp chuỗi 4 ảnh PPI (t0, t3, t10, t13) và ghép kênh thành tensor (12, 224, 224).
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T


# Bảng mã hóa 5 lớp thời tiết theo bài báo
LABEL_MAPPING = {
    "Clear": 0,
    "Light rain": 1,
    "Moderate rain": 2,
    "Heavy rain": 3,
    "Very heavy rain": 4
}


class RadarNowcastingDataset(Dataset):
    def __init__(self, metadata_csv: Path, ppi_dir: Path, horizon: str = "2h", transform=None):
        """
        Args:
            metadata_csv: Đường dẫn file metadata (VD: dataset_2025-08-01.csv)
            ppi_dir: Thư mục chứa các ảnh PPI (.png)
            horizon: Mốc dự báo ('0h', '1h', '2h', '3h')
            transform: Các phép biến đổi bổ sung cho ảnh
        """
        self.ppi_dir = ppi_dir
        self.horizon = horizon
        self.label_col = f"label_{horizon}"
        self.transform = transform

        # Đọc metadata và lọc bỏ các mẫu không có nhãn tương lai
        df = pd.read_csv(metadata_csv)
        self.data = df.dropna(subset=[self.label_col]).reset_index(drop=True)

        print(f"Khởi tạo Dataset [Horizon={horizon}]: {len(self.data)} mẫu hợp lệ / {len(df)} nhóm gốc.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # Lấy tên 4 file quét liên tiếp
        raw_files = [row["file_t0"], row["file_t3"], row["file_t10"], row["file_t13"]]
        image_tensors = []

        for raw_name in raw_files:
            # Tên file ảnh tương ứng (thay đuôi .RAW* thành .png)
            img_name = Path(raw_name).stem + ".png"
            img_path = self.ppi_dir / img_name

            if not img_path.exists():
                raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")

            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                # Chuyển PIL Image -> Tensor shape (3, 224, 224) với giá trị [0.0, 1.0]
                tensor_img = T.functional.to_tensor(img_rgb)
                image_tensors.append(tensor_img)

        # Ghép 4 tensor (mỗi tensor 3 kênh) dọc theo chiều channel (dim=0)
        # Kết quả thu được tensor kích thước: (12, 224, 224)
        x = torch.cat(image_tensors, dim=0)

        # Lấy nhãn số nguyên
        label_str = row[self.label_col]
        y = torch.tensor(LABEL_MAPPING[label_str], dtype=torch.long)

        return x, y


# ============================================================
# KHỐI TEST THỬ NGHIỆM DATASET VÀ DATALOADER
# ============================================================
if __name__ == "__main__":
    meta_path = Path("outputs/metadata/dataset_2025-08-01.csv")
    ppi_folder = Path("outputs/ppi/2025-08-01")

    # Thử nghiệm với mốc dự báo 2 giờ (2h)
    dataset = RadarNowcastingDataset(meta_path, ppi_folder, horizon="2h")

    # Tạo DataLoader mini-batch size = 4
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Lấy thử 1 batch đầu tiên để kiểm tra kích thước
    for batch_x, batch_y in dataloader:
        print("\n========== KẾT QUẢ KIỂM TRA BATCH ĐẦU TIÊN ==========")
        print(f"-> Kích thước batch đầu vào X : {batch_x.shape} (Batch_size, Channels, Height, Width)")
        print(f"-> Kích thước batch nhãn y   : {batch_y.shape}")
        print(f"-> Giá trị nhãn trong batch  : {batch_y.tolist()}")
        print(f"-> Kiểu dữ liệu X            : {batch_x.dtype}, Min={batch_x.min():.2f}, Max={batch_x.max():.2f}")
        print("======================================================")
        break