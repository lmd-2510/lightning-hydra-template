import torch
from torch import nn
from torchvision.models import resnet18, ResNet18_Weights

class ResNet18Landmark(nn.Module):
    def __init__(self, num_landmarks: int = 98, pretrained: bool = True):
        super().__init__()
        
        # 1. Khởi tạo mạng ResNet18
        # Dùng weights=ResNet18_Weights.DEFAULT để lấy tạ đã được train sẵn trên ImageNet
        # Giúp mô hình hội tụ nhanh hơn rất nhiều so với việc học lại từ đầu
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        self.backbone = resnet18(weights=weights)

        # 2. Lấy số lượng nơ-ron đầu vào của lớp FC cũ (đối với ResNet18 thì số này là 512)
        in_features = self.backbone.fc.in_features

        # 3. "Đập đi xây lại" lớp FC cuối cùng
        # num_landmarks * 2 vì mỗi điểm có 2 tọa độ (x, y)
        self.backbone.fc = nn.Linear(in_features, num_landmarks * 2)

    def forward(self, x: torch.Tensor):
        # Ảnh đi qua toàn bộ mạng backbone và trả về luôn 196 giá trị
        return self.backbone(x)