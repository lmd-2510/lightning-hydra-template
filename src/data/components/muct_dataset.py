import os
import torch
import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

class MUCTDataset(Dataset):
    def __init__(self, data_dir, num_landmarks=75, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.num_landmarks = num_landmarks
        
        # 1. Định vị file CSV
        landmark_dir = os.path.join(data_dir, "muct-landmarks-v1/muct-landmarks")
        csv_file = [f for f in os.listdir(landmark_dir) if f.endswith('.csv')][0]
        csv_path = os.path.join(landmark_dir, csv_file)
        self.data_frame = pd.read_csv(csv_path)
        
        # 2. Định vị 5 thư mục ảnh
        self.img_dirs = [
            os.path.join(data_dir, "muct-a-jpg-v1/jpg"),
            os.path.join(data_dir, "muct-b-jpg-v1/jpg"),
            os.path.join(data_dir, "muct-c-jpg-v1/jpg"),
            os.path.join(data_dir, "muct-d-jpg-v1/jpg"),
            os.path.join(data_dir, "muct-e-jpg-v1/jpg")
        ]

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        # 3. Tìm ảnh
        img_name = str(self.data_frame.iloc[idx, 0])
        if not img_name.endswith('.jpg'):
            img_name += '.jpg'
            
        img_path = None
        for d in self.img_dirs:
            temp_path = os.path.join(d, img_name)
            if os.path.exists(temp_path):
                img_path = temp_path
                break
                
        if img_path is None:
            raise FileNotFoundError(f"Không tìm thấy ảnh {img_name}")

        image = Image.open(img_path).convert("RGB")
        
        # 4. Lấy tọa độ và gom nhóm
        num_coords = self.num_landmarks * 2
        landmarks = self.data_frame.iloc[idx, -num_coords:].values.astype('float32')
        keypoints = landmarks.reshape(-1, 2).tolist()
        
        # 5. Biến đổi ảnh và tọa độ
        if self.transform:
            image_np = np.array(image) 
            transformed = self.transform(image=image_np, keypoints=keypoints)
            image = transformed['image']
            landmarks = torch.tensor(transformed['keypoints'], dtype=torch.float32).flatten()
            
        return image, landmarks