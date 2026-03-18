import torch
from torch import nn
from lightning import LightningModule

class CVLitModule(LightningModule):
    def __init__(self, net: nn.Module, optimizer: torch.optim.Optimizer, num_landmarks: int = 75, **kwargs):
        super().__init__()
        self.save_hyperparameters(logger=False, ignore=['net']) 
        
        self.net = net 
        # Sửa lại cái đầu ra (Last Layer) của ResNet để nó trả về 150 con số (75 cặp x,y)
        in_features = self.net.fc.in_features 
        self.net.fc = nn.Linear(in_features, num_landmarks * 2) 
        
        # Dùng MSELoss vì đây là bài toán dự đoán số (tọa độ)
        self.criterion = nn.MSELoss()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch # x là ảnh, y là 150 con số tọa độ
        preds = self.forward(x) 
        loss = self.criterion(preds, y) # So sánh dự đoán với thực tế
        self.log("train/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        self.log("test/loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)
        loss = self.criterion(preds, y)
        self.log("val/loss", loss, on_step=True, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        # Lấy cái Adam hoặc SGD mà bạn cấu hình trong YAML để chạy
        return self.hparams.optimizer(params=self.parameters())