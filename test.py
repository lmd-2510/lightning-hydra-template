import matplotlib.pyplot as plt
import torch
import numpy as np
# Import các class của bạn (đảm bảo đúng đường dẫn)
from src.data.WFLW_datamodule import WFLWDataModule 

def visualize_sample(data_dir, image_size=256):
    # 1. Khởi tạo DataModule
    dm = WFLWDataModule(data_dir=data_dir, batch_size=4, image_size=image_size)
    dm.setup("fit")
    
    # 2. Lấy 1 batch từ train_dataloader
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    images, landmarks = batch # images: [B, 3, 256, 256], landmarks: [B, 196]

    # 3. Chọn 1 mẫu đầu tiên trong batch để vẽ
    img = images[1].permute(1, 2, 0).numpy() # Chuyển từ [C, H, W] -> [H, W, C]
    
    # Nếu trong transforms bạn có dùng A.Normalize, bạn cần "un-normalize" để ảnh hiện đúng màu
    # Nếu chỉ dùng ToTensorV2 thì chỉ cần clip về [0, 1]
    img = np.clip(img, 0, 1) 

    # 4. Xử lý tọa độ Landmark
    # Đưa về dạng [98, 2] và nhân ngược với image_size
    pts = landmarks[1].view(-1, 2).numpy()
    pts[:, 0] *= image_size # Nhân ngược với chiều rộng
    pts[:, 1] *= image_size # Nhân ngược với chiều cao

    # 5. Vẽ ảnh
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.scatter(pts[:, 0], pts[:, 1], s=10, c='r', marker='o')
    plt.title("Check Landmark: Red points should be on face features")
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    # Thay đường dẫn tới thư mục WFLW của bạn
    visualize_sample(data_dir="data/WFLW")