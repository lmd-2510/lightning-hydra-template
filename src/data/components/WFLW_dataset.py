import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class WFLWDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
        # ... (Phần __init__ giữ nguyên như cũ) ...
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        anno_dir = os.path.join(data_dir, "WFLW_annotations", "WFLW_annotations")
        if self.split == "train":
            txt_path = os.path.join(anno_dir, "list_98pt_rect_attr_train_test", "list_98pt_rect_attr_train.txt")
        else:
            txt_path = os.path.join(anno_dir, "list_98pt_rect_attr_train_test", "list_98pt_rect_attr_test.txt")
            
        if not os.path.exists(txt_path):
            raise FileNotFoundError(f"Không tìm thấy file annotation tại: {txt_path}")

        with open(txt_path, 'r') as f:
            self.annotations = f.readlines()
            
        self.img_base_dir = os.path.join(data_dir, "WFLW_images", "WFLW_images")

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        line = self.annotations[idx].strip().split()
        
        img_name = line[-1] 
        img_path = os.path.join(self.img_base_dir, img_name)
        
        if not os.path.exists(img_path):
            raise FileNotFoundError(f"Không tìm thấy ảnh {img_path}")

        image = Image.open(img_path).convert("RGB")
        img_width, img_height = image.size
        
        # 1. LẤY BOUNDING BOX VÀ TỌA ĐỘ LANDMARK GỐC
        # Box nằm ở vị trí 196, 197, 198, 199 (xmin, ymin, xmax, ymax)
        bbox = np.array(line[196:200], dtype=np.float32)
        xmin, ymin, xmax, ymax = bbox
        
        # Landmarks là 196 giá trị đầu tiên
        landmarks = np.array(line[:196], dtype=np.float32).reshape(-1, 2)

        # Đảm bảo box không bị văng ra khỏi kích thước ảnh thực tế
        xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
        xmax, ymax = min(img_width, int(xmax)), min(img_height, int(ymax))

        # Tính toán chiều rộng và cao của vùng crop
        crop_w = xmax - xmin
        crop_h = ymax - ymin

        # 2. CROP ẢNH
        image = image.crop((xmin, ymin, xmax, ymax))
        
        # 3. CHUẨN HÓA LANDMARKS VỀ [0, 1] (Sửa lại đoạn này)
        # Thay vì chỉ trừ xmin, ta chia cho kích thước vùng crop
        landmarks[:, 0] = (landmarks[:, 0] - xmin) / (crop_w + 1e-6)
        landmarks[:, 1] = (landmarks[:, 1] - ymin) / (crop_h + 1e-6)
        
        # Chuyển về list để đưa vào Albumentations
        keypoints = landmarks.tolist()
        
        # 4. ĐƯA QUA ALBUMENTATIONS BÌNH THƯỜNG
        if self.transform:
            image_np = np.array(image) 
            transformed = self.transform(image=image_np, keypoints=keypoints)
            image = transformed['image']
            
            if len(transformed['keypoints']) > 0:
                landmarks = torch.tensor(transformed['keypoints'], dtype=torch.float32).flatten()
            else:
                landmarks = torch.zeros(196, dtype=torch.float32)
        else:
            landmarks = torch.tensor(landmarks, dtype=torch.float32).flatten()
            
        return image, landmarks