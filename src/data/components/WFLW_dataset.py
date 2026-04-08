import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class WFLWDataset(Dataset):
    def __init__(self, data_dir, split="train", transform=None):
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
        bbox = np.array(line[196:200], dtype=np.float32)
        xmin, ymin, xmax, ymax = bbox
        
        landmarks = np.array(line[:196], dtype=np.float32).reshape(-1, 2)

        # Giới hạn box trong phạm vi ảnh
        xmin, ymin = max(0, int(xmin)), max(0, int(ymin))
        xmax, ymax = min(img_width, int(xmax)), min(img_height, int(ymax))

        # 2. CROP ẢNH
        image = image.crop((xmin, ymin, xmax, ymax))
        
        # 3. CHUYỂN LANDMARKS VỀ TỌA ĐỘ CỦA VÙNG CROP (PIXEL - KHÔNG CHUẨN HÓA [0,1] Ở ĐÂY)
        # Quan trọng: Albumentations cần tọa độ pixel thực tế để resize/rotate chính xác.
        landmarks[:, 0] -= xmin
        landmarks[:, 1] -= ymin
        
        keypoints = landmarks.tolist()
        
        # 4. ĐƯA QUA ALBUMENTATIONS
        if self.transform:
            image_np = np.array(image) 
            transformed = self.transform(image=image_np, keypoints=keypoints)
            image = transformed['image'] # Lúc này image đã là Tensor và đã Resize (VD: 256x256)
            
            # Sau khi transform (đã resize), ta lấy tọa độ pixel mới và chuẩn hóa về [0, 1]
            if len(transformed['keypoints']) > 0:
                new_landmarks = np.array(transformed['keypoints'])
                
                # Lấy kích thước ảnh sau khi biến đổi (thường là 256 từ datamodule)
                # Image ở đây đã là Tensor [C, H, W] do ToTensorV2()
                _, final_h, final_w = image.shape
                
                new_landmarks[:, 0] /= final_w
                new_landmarks[:, 1] /= final_h
                landmarks_tensor = torch.tensor(new_landmarks, dtype=torch.float32).flatten()
            else:
                # Trường hợp hi hữu keypoints bị văng ra khỏi ảnh hoàn toàn
                landmarks_tensor = torch.zeros(196, dtype=torch.float32)
        else:
            # Nếu không có transform, chuẩn hóa dựa trên kích thước vùng crop gốc
            crop_w, crop_h = image.size
            landmarks[:, 0] /= (crop_w + 1e-6)
            landmarks[:, 1] /= (crop_h + 1e-6)
            landmarks_tensor = torch.tensor(landmarks, dtype=torch.float32).flatten()
            
        return image, landmarks_tensor