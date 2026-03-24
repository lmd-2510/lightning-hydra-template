import cv2
import torch
import numpy as np
from src.models.components.resnet18 import ResNet18Landmark

print("--- ĐANG KHỞI TẠO AI VỚI CHẾ ĐỘ PRETRAINED ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

weight_path = "checkpoints/resnet18_wflw_weights.pth" 

# 1. BẬT PRETRAINED=TRUE 
# Lúc này ResNet18 sẽ tự tải trọng số từ ImageNet về (nếu máy có mạng)
model = ResNet18Landmark(num_landmarks=98, pretrained=True)

# 2. NẠP TRỌNG SỐ CỦA BẠN ĐỂ "DẠY" NÓ CÁCH CHẤM 98 ĐIỂM
model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=False))
model = model.to(device)
model.eval()
print("✅ Đã kích hoạt Sức mạnh Pretrained + Trọng số của bạn!")

# 3. BỘ CẮT MẶT OPENCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, (256, 256))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    
    # CHÚ Ý: Khi dùng Pretrained, thường người ta sẽ chuẩn hóa theo chuẩn ImageNet
    # Nhưng để so sánh công bằng với bản cũ của bạn, ta giữ nguyên cách xử lý cũ:
    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
    return face_tensor.unsqueeze(0).to(device)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    h_frame, w_frame, _ = frame.shape
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray_frame, 1.1, 5, minSize=(100, 100))
    
    for (x_min, y_min, width, height) in faces:
        pad_w, pad_h = int(width * 0.1), int(height * 0.1)
        x1, y1 = max(0, x_min - pad_w), max(0, y_min - pad_h)
        x2, y2 = min(w_frame, x_min + width + pad_w), min(h_frame, y_min + height + pad_h)
        
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0: continue
        
        input_tensor = preprocess_face(face_crop)
        with torch.no_grad():
            landmarks = model(input_tensor) 
        
        pts = landmarks.view(-1, 2).cpu().numpy()
        pts = pts / 256.0 
        pts[:, 0] = pts[:, 0] * (x2 - x1) + x1
        pts[:, 1] = pts[:, 1] * (y2 - y1) + y1
        
        for (x, y) in pts:
            cv2.circle(frame, (int(x), int(y)), 2, (0, 255, 0), -1) # Chuyển sang chấm XANH LÁ để phân biệt

    cv2.imshow('SO SÁNH: BẢN PRETRAINED (GREEN POINTS)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()