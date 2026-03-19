from lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# Gọi "Anh Đầu Bếp" từ thư mục components ra để dùng
from src.data.components.muct_dataset import MUCTDataset

class MUCTDataModule(LightningDataModule):
    def __init__(self, data_dir: str, num_landmarks: int = 75, batch_size: int = 16, num_workers: int = 0):
        super().__init__()
        self.save_hyperparameters(logger=False)
        
        # Khai báo các phép biến đổi ảnh (Albumentations)
        self.transform = A.Compose([
            A.Resize(height=224, width=224), 
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)), 
            ToTensorV2() 
        ], keypoint_params=A.KeypointParams(format='xy', remove_invisible=False)) 

    def setup(self, stage=None):
        # Truyền thông số vào cho anh Đầu bếp
        full_dataset = MUCTDataset(
            data_dir=self.hparams.data_dir, 
            num_landmarks=self.hparams.num_landmarks,
            transform=self.transform
        )

        generator = torch.Generator().manual_seed(12345)

        # Chia lô 80% học, 20% thi
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size], generator=generator)

    def train_dataloader(self):
        return DataLoader(
        self.train_dataset,
        batch_size=self.hparams.batch_size,
        num_workers=self.hparams.num_workers,
        shuffle=True,
        pin_memory=True,
        persistent_workers=self.hparams.num_workers > 0
    )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.val_dataset, # Mượn tạm tập val_dataset (20%) để chấm điểm
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,    # Lưu ý: Khi test tuyệt đối không xáo trộn ảnh (shuffle=False)
            pin_memory=True,
            persistent_workers=self.hparams.num_workers > 0
        )