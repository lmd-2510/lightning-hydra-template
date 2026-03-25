import numpy as np

class PointEMA:
    def __init__(self, alpha=0.3):
        """
        Args:
            alpha (float): Hệ số làm mượt (0 < alpha <= 1). 
                           Càng nhỏ càng mượt nhưng càng trễ.
        """
        self.alpha = alpha
        self.shadow = None

    def update(self, data):
        """
        data: Tọa độ mới từ model (numpy array shape [98, 2])
        """
        if self.shadow is None:
            # Frame đầu tiên: khởi tạo shadow bằng chính data
            self.shadow = data.copy()
            return self.shadow
        
        # Công thức EMA: P_new = alpha * p_current + (1 - alpha) * P_old
        self.shadow = self.alpha * data + (1.0 - self.alpha) * self.shadow
        return self.shadow

    def reset(self):
        """Gọi hàm này khi đổi sang khuôn mặt mới hoặc mất dấu khuôn mặt"""
        self.shadow = None