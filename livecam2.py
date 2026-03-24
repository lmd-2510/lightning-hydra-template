import cv2
import torch
import numpy as np

# Nhập (Import) cấu trúc Mạng ResNet18 của bạn từ Template
from src.models.components.resnet18 import ResNet18Landmark

print("--- ĐANG KHỞI TẠO AI TỪ TEMPLATE ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Đường dẫn tới file trọng số .pth của bạn
weight_path = "checkpoints/resnet18_wflw_weights.pth" 

# 1. NẠP MODEL RESNET18
model = ResNet18Landmark(num_landmarks=98, pretrained=False)
# Nạp trọng số, thêm weights_only=False để tránh cảnh báo bảo mật của PyTorch
model.load_state_dict(torch.load(weight_path, map_location=device, weights_only=False))
model = model.to(device)
model.eval()
print("✅ Tải Model ResNet18 thành công!")

# 2. KHỞI TẠO BỘ TÌM MẶT (OPENCV HAAR CASCADE)
# Sử dụng bộ tìm mặt mặc định có sẵn trong OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
print("✅ Khởi động bộ cắt mặt thành công!")

def preprocess_face(face_img):
    """Xử lý ảnh khuôn mặt trước khi đưa vào AI chấm điểm"""
    # Resize về kích thước model yêu cầu (WFLW thường dùng 256x256)
    face_resized = cv2.resize(face_img, (256, 256))
    # Chuyển BGR (OpenCV) sang RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    # Chuyển thành Tensor [C, H, W] và chuẩn hóa về dải [0, 1]
    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
    # Thêm batch dimension: [1, 3, 256, 256]
    return face_tensor.unsqueeze(0).to(device)

# --- BẬT WEBCAM ---
cap = cv2.VideoCapture(0) # 0 là camera mặc định
print("🎥 Bật Webcam! Nhấn phím 'q' trên bàn phím để thoát.")

# Thiết lập kích thước cửa sổ hiển thị (tùy chọn, giúp ảnh đỡ bị vỡ)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
        
    # Lật ảnh theo chiều ngang như soi gương cho tự nhiên
    frame = cv2.flip(frame, 1)
    h_frame, w_frame, _ = frame.shape
    
    # Chuyển ảnh sang trắng đen để bộ cắt mặt OpenCV hoạt động tốt hơn
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 3. OPENCV NHẬN DIỆN KHUÔN MẶT
    # detectMultiScale trả về danh sách các ô vuông (x, y, w, h) chứa khuôn mặt
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100))
    
    for (x_min, y_min, width, height) in faces:
        # Mở rộng khung mặt (padding) một chút (10%) để lấy trọn vẹn cằm/trán
        pad_w, pad_h = int(width * 0.1), int(height * 0.1)
        
        x1 = max(0, x_min - pad_w)
        y1 = max(0, y_min - pad_h)
        x2 = min(w_frame, x_min + width + pad_w)
        y2 = min(h_frame, y_min + height + pad_h)
        
        # Cắt ảnh khuôn mặt
        face_crop = frame[y1:y2, x1:x2]
        # Kiểm tra nếu ảnh cắt bị rỗng (do nằm sát mép camera) thì bỏ qua
        if face_crop.size == 0: continue
        
        # 4. Đưa ảnh khuôn mặt vào ResNet18 dự đoán 98 điểm
        input_tensor = preprocess_face(face_crop)
        with torch.no_grad():
            landmarks = model(input_tensor) 
        
        # 5. Xử lý toạ độ điểm ảnh (CHUẨN THEO CÁCH BẠN TRAIN)
        # Output landmarks thường có shape [1, 196], ta reshape về [98, 2]
        pts = landmarks.view(-1, 2).cpu().numpy()
        
        # CHUẨN HOÁ VỀ TỶ LỆ [0 -> 1]: Chia cho kích thước ảnh lúc Train (256)
        # (Dựa trên file WFLWDataset bạn gửi, model dự đoán toạ độ tuyệt đối 0-256)
        pts = pts / 256.0 
        
        # NHÂN TỶ LỆ VỚI KÍCH THƯỚC THỰC TẾ của khuôn mặt trên Webcam
        pts[:, 0] *= (x2 - x1)
        pts[:, 1] *= (y2 - y1)
        
        # CỘNG THÊM OFFSET (toạ độ góc trên trái khung mặt) để điểm nằm đúng vị trí trên khung hình to
        pts[:, 0] += x1
        pts[:, 1] += y1
        
        # 6. VẼ KẾT QUẢ
        # --- ĐÃ XÓA DÒNG LỆNH VẼ Ô VUÔNG MÀU XANH TẠI ĐÂY ---
        
        # Vẽ 98 điểm Landmark (chấm tròn đỏ)
        for (x, y) in pts:
            # cv2.circle(ảnh, tâm, bán kính, màu BGR, độ dày (-1 là tô đặc))
            # Chỉnh bán kính thành 2 để chấm nhỏ gọn, sắc nét
            cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

    # Thêm tiêu đề cho cửa sổ và hiển thị
    cv2.imshow('ResNet18 WFLW - Livecam Landmark Only', frame)
    
    # Nhấn 'q' để tắt Camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()