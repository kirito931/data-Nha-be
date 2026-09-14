"""
Xây dựng PyTorch Dataset cho bài toán Nowcasting:
- Nạp chuỗi 4 ảnh PPI (t0, t3, t10, t13) và ghép kênh thành tensor (12, 224, 224).
- Tích hợp đầy đủ Data Transformation theo mục 4.2 và Hình 6 của bài báo gốc:
  1. RandAugment (2 operations, magnitude 7-11)
  2. Median Blur (kernel 5x5)
  3. Auto Contrast
  4. Chuẩn hóa kênh theo NRD-1 (Table 3)
"""
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as T


# Bảng mã hóa 5 lớp thời tiết theo bài báo
LABEL_MAPPING = {
    "Clear": 0,
    "Light rain": 1,
    "Moderate rain": 2,
    "Heavy rain": 3,
    "Very heavy rain": 4
}


class RadarTransform:
    """
    Hiện thực hóa toàn bộ chuỗi Data Transformation theo bài báo gốc (mục 4.2 và Hình 6):
    - is_train=True: RandAugment -> Median Blur 5x5 -> Auto Contrast -> ToTensor -> Normalize
    - is_train=False: Median Blur 5x5 -> Auto Contrast -> ToTensor -> Normalize (dành cho Val/Test)
    """
    def __init__(self, is_train: bool = True):
        self.is_train = is_train

        # 1. RandAugment: 2 phép biến đổi ngẫu nhiên, magnitude trong khoảng 7-11
        if is_train:
            self.randaug = T.RandAugment(num_ops=2, magnitude=9)
        else:
            self.randaug = None

        # 5. Chuẩn hóa kênh theo phân phối thống kê NRD-1 (Table 3 trong paper)
        # Red: mean=0.9844, std=0.0641 | Green: mean=0.9930, std=0.0342 | Blue: mean=0.9632, std=0.1163
        self.normalize = T.Normalize(
            mean=[0.9844, 0.9930, 0.9632],
            std=[0.0641, 0.0342, 0.1163]
        )

    def __call__(self, img: Image.Image) -> torch.Tensor:
        # 1. RandAugment (chỉ áp dụng cho tập Train)
        if self.is_train and self.randaug is not None:
            img = self.randaug(img)

        # 2. Median Blur 5x5: giảm nhiễu hạt speckle noise của radar
        img = img.filter(ImageFilter.MedianFilter(size=5))

        # 3. Auto Contrast: tăng tương phản giữa các vùng thời tiết
        img = ImageOps.autocontrast(img)

        # 4. Chuyển sang PyTorch Tensor [0.0, 1.0]
        tensor_img = T.functional.to_tensor(img)

        # 5. Normalization theo phân phối NRD-1
        tensor_img = self.normalize(tensor_img)

        return tensor_img


class RadarNowcastingDataset(Dataset):
    def __init__(
        self,
        metadata_csv: Path,
        ppi_dir: Path,
        horizon: str = "2h",
        transform=None,
        is_train: bool = False,
        apply_paper_transform: bool = True
    ):
        """
        Args:
            metadata_csv: Đường dẫn file metadata (VD: dataset_2025-08.csv)
            ppi_dir: Thư mục chứa các ảnh PPI (.png)
            horizon: Mốc dự báo ('0h', '1h', '2h', '3h')
            transform: Custom transform callable.
            is_train: True nếu dùng cho tập Train (bật RandAugment), False nếu dùng cho Val/Test
            apply_paper_transform: True để áp dụng đầy đủ RandAugment, MedianBlur, AutoContrast, Normalize theo bài báo.
        """
        self.ppi_dir = Path(ppi_dir)
        self.horizon = horizon
        self.label_col = f"label_{horizon}"
        self.is_train = is_train
        self.apply_paper_transform = apply_paper_transform

        if transform is not None:
            self.transform = transform
        elif apply_paper_transform:
            self.transform = RadarTransform(is_train=is_train)
        else:
            self.transform = None

        # Đọc metadata và lọc bỏ các mẫu không có nhãn tương lai
        df = pd.read_csv(metadata_csv)
        self.data = df.dropna(subset=[self.label_col]).reset_index(drop=True)

        print(f"Khởi tạo Dataset [Horizon={horizon} | is_train={is_train} | PaperTransform={apply_paper_transform}]: {len(self.data)} mẫu hợp lệ / {len(df)} nhóm gốc.")

    def __len__(self):
        return len(self.data)

    def get_item_with_transform(self, idx: int, custom_transform=None):
        row = self.data.iloc[idx]

        # Lấy tên 4 file quét liên tiếp
        raw_files = [row["file_t0"], row["file_t3"], row["file_t10"], row["file_t13"]]
        image_tensors = []

        active_transform = custom_transform if custom_transform is not None else self.transform

        for raw_name in raw_files:
            img_name = Path(raw_name).stem + ".png"
            img_path = self.ppi_dir / img_name

            if not img_path.exists():
                raise FileNotFoundError(f"Không tìm thấy ảnh: {img_path}")

            with Image.open(img_path) as img:
                img_rgb = img.convert("RGB")
                if active_transform is not None:
                    tensor_img = active_transform(img_rgb)
                else:
                    tensor_img = T.functional.to_tensor(img_rgb)
                image_tensors.append(tensor_img)

        # Ghép 4 tensor (mỗi tensor 3 kênh) dọc theo chiều channel (dim=0) -> (12, 224, 224)
        x = torch.cat(image_tensors, dim=0)

        # Lấy nhãn số nguyên
        label_str = row[self.label_col]
        y = torch.tensor(LABEL_MAPPING[label_str], dtype=torch.long)

        return x, y

    def __getitem__(self, idx):
        return self.get_item_with_transform(idx, self.transform)


class TransformedSubset(Dataset):
    """
    Bọc một Subset từ random_split và áp dụng transform riêng
    (ví dụ: Train dùng RandAugment, Val/Test dùng eval transform)
    """
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform

    def __len__(self):
        return len(self.subset)

    def __getitem__(self, idx):
        orig_idx = self.subset.indices[idx]
        return self.subset.dataset.get_item_with_transform(orig_idx, self.transform)


def get_nowcasting_split_datasets(
    metadata_csv: Path,
    ppi_dir: Path,
    horizon: str = "2h",
    train_ratio: float = 0.80,
    val_ratio: float = 0.10,
    seed: int = 42
):
    """
    Hàm tiện ích chia tập dữ liệu 80:10:10 và áp dụng đúng Data Transformation:
    - train_set: có RandAugment, Median Blur, Auto Contrast, Normalization
    - val_set, test_set: có Median Blur, Auto Contrast, Normalization (không RandAugment)
    """
    # Khởi tạo dataset gốc (dùng transform mặc định)
    base_dataset = RadarNowcastingDataset(metadata_csv, ppi_dir, horizon=horizon, is_train=False)
    total_samples = len(base_dataset)

    train_size = int(train_ratio * total_samples)
    val_size = int(val_ratio * total_samples)
    test_size = total_samples - train_size - val_size

    generator = torch.Generator().manual_seed(seed)
    raw_train, raw_val, raw_test = random_split(
        base_dataset, [train_size, val_size, test_size], generator=generator
    )

    train_set = TransformedSubset(raw_train, RadarTransform(is_train=True))
    val_set = TransformedSubset(raw_val, RadarTransform(is_train=False))
    test_set = TransformedSubset(raw_test, RadarTransform(is_train=False))

    return train_set, val_set, test_set, raw_test.indices


# ============================================================
# KHỐI TEST THỬ NGHIỆM DATASET VÀ TRANSFORMATION
# ============================================================
if __name__ == "__main__":
    meta_path = Path("outputs/metadata/dataset_2025-08.csv")
    ppi_folder = Path("outputs/ppi/all_scans")

    print("\n--- KIỂM TRA DATA TRANSFORM HOÀN CHỈNH ---")
    train_set, val_set, test_set, test_indices = get_nowcasting_split_datasets(
        meta_path, ppi_folder, horizon="2h"
    )

    print(f"-> Train set: {len(train_set)} mẫu | Val set: {len(val_set)} mẫu | Test set: {len(test_set)} mẫu")

    # DataLoader thử nghiệm
    loader = DataLoader(train_set, batch_size=4, shuffle=True)
    for batch_x, batch_y in loader:
        print("\n========== KẾT QUẢ KIỂM TRA BATCH ĐẦU TIÊN VỚI TRANSFORM ==========")
        print(f"-> Kích thước batch đầu vào X : {batch_x.shape} (Batch_size, Channels=12, H=224, W=224)")
        print(f"-> Kích thước batch nhãn y   : {batch_y.shape}")
        print(f"-> Giá trị nhãn trong batch  : {batch_y.tolist()}")
        print(f"-> Kiểu dữ liệu X            : {batch_x.dtype}")
        print(f"-> Phân phối X sau chuẩn hóa : Min={batch_x.min():.2f}, Max={batch_x.max():.2f}, Mean={batch_x.mean():.2f}")
        print("===================================================================")
        break