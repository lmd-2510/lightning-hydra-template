from typing import Optional, Tuple
import torch
from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, Dataset, Subset
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Giả sử file dataset của bạn đặt tại src/data/components/WFLW_dataset.py
from src.data.components.WFLW_dataset import WFLWDataset 

class WFLWDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/",
        train_val_split: Tuple[float, float] = (0.9, 0.1), 
        batch_size: int = 32,
        num_workers: int = 2, # Đã chỉnh xuống 2 cho an toàn (ví dụ trên Kaggle)
        pin_memory: bool = True, # Nên bật True nếu dùng GPU
        image_size: int = 256, # Kích thước ảnh đưa vào mô hình
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

        self.train_transforms = A.Compose([
            A.Resize(self.hparams.image_size, self.hparams.image_size),
            
            # 1. BIẾN ĐỔI KHÔNG GIAN (SPATIAL TRANSFORMS)
            # Dịch chuyển, thu phóng, xoay. Giữ nguyên border_mode=0 (viền đen) để tránh bị nhân đôi nửa khuôn mặt ở viền.
            A.ShiftScaleRotate(shift_limit=0.06, scale_limit=0.15, rotate_limit=10, p=0.5, border_mode=0),
            
            # Thêm Perspective: Cực kỳ hữu ích cho WFLW để giả lập các góc chụp nghiêng của camera
            A.Perspective(scale=(0.02, 0.08), p=0.3),

            # 2. BIẾN ĐỔI CẤP ĐỘ PIXEL (MÀU SẮC & ÁNH SÁNG)
            A.OneOf([
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=1.0),
                A.RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.25, p=1.0),
                A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=1.0), # Cân bằng sáng cục bộ, rất tốt cho ảnh bị ngược sáng
            ], p=0.6),

            # 3. GIẢ LẬP ẢNH KÉM CHẤT LƯỢNG (NHIỄU & MỜ)
            A.OneOf([
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=5, p=1.0), # Giả lập chụp mặt khi đang chuyển động
                A.GaussNoise(p=1.0), # Nhiễu hạt camera
                A.ImageCompression(quality_lower=50, quality_upper=95, p=1.0), # Giả lập ảnh bị nén thấp trên web
            ], p=0.5),

            # 4. CHE KHUẤT (OCCLUSION) - Cải thiện khả năng chống chịu khi bị che mặt (đeo kính, tóc che, vật cản)
            A.CoarseDropout(
                max_holes=4, # Tạo tối đa 4 vùng che
                max_height=int(self.hparams.image_size * 0.2), # Vùng che tối đa bằng 20% kích thước ảnh
                max_width=int(self.hparams.image_size * 0.2),
                min_holes=1,
                min_height=int(self.hparams.image_size * 0.05),
                min_width=int(self.hparams.image_size * 0.05),
                fill_value=0, # Phủ màu đen (pixel value = 0)
                p=0.4 # Xác suất 40% ảnh sẽ bị che
            ),

            # 5. CHUẨN HÓA & TENSOR
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
            
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

        # 2. Transform cho tập Val & Test (Chỉ Resize, Normalize và ToTensor)
        self.val_test_transforms = A.Compose([
            A.Resize(self.hparams.image_size, self.hparams.image_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False))

    def setup(self, stage: Optional[str] = None):
        if not self.data_train and not self.data_val and not self.data_test:
                        
            # Khởi tạo dataset cơ bản để lấy tổng chiều dài
            base_train_dataset = WFLWDataset(data_dir=self.hparams.data_dir, split="train", transform=None)
            total_train_len = len(base_train_dataset)
            
            train_len = int(total_train_len * self.hparams.train_val_split[0])
            val_len = total_train_len - train_len
            
            # Chỉ dùng random_split để lấy ra các index (chỉ mục)
            indices = list(range(total_train_len))
            train_idx, val_idx = random_split(
                dataset=indices,
                lengths=[train_len, val_len],
                generator=torch.Generator().manual_seed(42),
            )
            
            # Khởi tạo lại 2 dataset riêng biệt với 2 transform khác nhau, dùng Subset để bọc lại
            dataset_with_train_tf = WFLWDataset(self.hparams.data_dir, split="train", transform=self.train_transforms)
            dataset_with_val_tf = WFLWDataset(self.hparams.data_dir, split="train", transform=self.val_test_transforms)
            
            self.data_train = Subset(dataset_with_train_tf, train_idx.indices)
            self.data_val = Subset(dataset_with_val_tf, val_idx.indices)

            # --- LOAD TẬP TEST ---
            self.data_test = WFLWDataset(
                data_dir=self.hparams.data_dir, 
                split="test", 
                transform=self.val_test_transforms
            )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=True, 
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=False,
        )