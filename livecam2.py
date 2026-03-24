import cv2
import torch
import numpy as np
# 1. Thêm các thư viện cần thiết cho YOLO từ Hugging Face
from huggingface_hub import hf_hub_download
from ultralytics import YOLO

# Nhập (Import) cấu trúc Mạng ResNet18 của bạn từ Template
# Đảm bảo file src/models/components/resnet18.py tồn tại
from src.models.components.resnet18 import ResNet18Landmark

print("--- ĐANG KHỞI TẠO HỆ THỐNG AI ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang chạy trên thiết bị: {device}")

# --- A. KHỞI TẠO MODEL RESNET18 DỰ ĐOÁN LANDMARK ---
# Đường dẫn tới file trọng số .pth của bạn (sau khi train)
resnet_weight_path = "checkpoints/resnet18_wflw_weights.pth" 

model_landmark = ResNet18Landmark(num_landmarks=98, pretrained=False)

# Nạp trọng số Landmark
try:
    model_landmark.load_state_dict(torch.load(resnet_weight_path, map_location=device, weights_only=False))
    model_landmark = model_landmark.to(device)
    model_landmark.eval()
    print("✅ Tải Model ResNet18 Landmark thành công!")
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file trọng số ResNet tại {resnet_weight_path}. Vui lòng kiểm tra lại.")
    exit()

# --- B. KHỞI TẠO MODEL YOLOv8 FACE DETECTION TỪ HUGGING FACE ---
print("⏳ Đang tải YOLOv8 Face Detection model từ Hugging Face (có thể mất chút thời gian lần đầu)...")
try:
    # Tải model từ repo chỉ định
    yolo_model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    # Nạp model YOLO
    yolo_detector = YOLO(yolo_model_path)
    # Đẩy model YOLO lên GPU nếu có để chạy nhanh hơn
    yolo_detector.to(device) 
    print("✅ Khởi động YOLOv8 Face Detector thành công!")
except Exception as e:
    print(f"❌ Lỗi khi tải hoặc nạp model YOLO: {e}")
    exit()


def preprocess_face(face_img):
    """Xử lý ảnh khuôn mặt trước khi đưa vào AI chấm điểm"""
    # Resize về kích thước model ResNet yêu cầu (WFLW dùng 256x256)
    face_resized = cv2.resize(face_img, (256, 256))
    # Chuyển BGR (OpenCV) sang RGB
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    # Chuyển thành Tensor [C, H, W] và chuẩn hóa về dải [0, 1]
    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
    # Thêm batch dimension: [1, 3, 256, 256]
    return face_tensor.unsqueeze(0).to(device)

# --- C. BẬT WEBCAM VÀ CHẠY REAL-TIME ---
cap = cv2.VideoCapture(0) # 0 là camera mặc định
print("🎥 Bật Webcam! Nhấn phím 'q' trên bàn phím để thoát.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
        
    # Lật ảnh theo chiều ngang như soi gương cho tự nhiên
    frame = cv2.flip(frame, 1)
    h_frame, w_frame, _ = frame.shape
    
    # --- 3. DÙNG YOLOv8 ĐỂ NHẬN DIỆN KHUÔN MẶT ---
    # YOLOv8 chạy trực tiếp trên ảnh màu BGR, không cần chuyển ảnh xám
    # verbose=False để tắt log in ra terminal lúc chạy, conf=0.5 để đặt ngưỡng tin cậy
    yolo_results = yolo_detector(frame, verbose=False, conf=0.5)
    
    # Trích xuấtBounding Boxes [xmin, ymin, xmax, ymax] dạng numpy
    # YOLOv8 có thể trả về nhiều khuôn mặt
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    
    for box in boxes:
        # Lấy tọa độ gốc từ YOLO
        xmin_raw, ymin_raw, xmax_raw, ymax_raw = box
        
        # Tính toán chiều rộng, chiều cao khuôn mặt gốc
        width_yolo = xmax_raw - xmin_raw
        height_yolo = ymax_raw - ymin_raw
        
        # Mở rộng khung mặt (padding) một chút (10%) giống như lúc train để lấy trọn vẹn cằm/trán
        pad_w, pad_h = int(width_yolo * 0.2), int(height_yolo * 0.2)
        
        # Tọa độ khung mặt sau khi mở rộng và ép kiểu nguyên
        x1 = max(0, int(xmin_raw - pad_w))
        y1 = max(0, int(ymin_raw - pad_h))
        x2 = min(w_frame, int(xmax_raw + pad_w))
        y2 = min(h_frame, int(ymax_raw + pad_h))
        
        # Cắt ảnh khuôn mặt (Face Crop)
        face_crop = frame[y1:y2, x1:x2]
        # Kiểm tra nếu ảnh cắt bị rỗng (do nằm sát mép camera) thì bỏ qua
        if face_crop.size == 0: continue
        
        # --- 4. ĐƯA KHUÔN MẶT ĐÃ CẮT VÀO RESNET18 ĐỂ DỰ ĐOÁN 98 ĐIỂM ---
        input_tensor = preprocess_face(face_crop)
        with torch.no_grad():
            landmarks = model_landmark(input_tensor) 
        
        # --- 5. XỬ LÝ TỌA ĐỘ ĐIỂM LANDMARK ---
        # Reshape về [98, 2]
        pts = landmarks.view(-1, 2).cpu().numpy()
        
        # CHUẨN HOÁ VỀ TỶ LỆ [0 -> 1]: Chia cho kích thước ảnh lúc Train (256)
        pts = pts / 256.0 
        
        # NHÂN TỶ LỆ VỚI KÍCH THƯỚC THỰC TẾ của khuôn mặt đã mở rộng trên Webcam
        # pts[:, 0] là x, pts[:, 1] là y
        pts[:, 0] *= (x2 - x1)
        pts[:, 1] *= (y2 - y1)
        
        # CỘNG THÊM OFFSET (tọa độ góc trên trái khung mặt y1, x1) để điểm nằm đúng vị trí trên khung hình to
        pts[:, 0] += x1
        pts[:, 1] += y1
        
        # --- 6. VẼ KẾT QUẢ ---
        # Không vẽ ô vuông (đã xóa), chỉ vẽ 98 điểm Landmark màu đỏ
        for (x, y) in pts:
            # cv2.circle(ảnh, tâm, bán kính, màu BGR, độ dày (-1 là tô đặc))
            cv2.circle(frame, (int(x), int(y)), 2, (0, 0, 255), -1)

    # Thêm tiêu đề cho cửa sổ và hiển thị
    cv2.imshow('YOLOv8 Face + ResNet18 WFLW - Livecam', frame)
    
    # Nhấn 'q' để tắt Camera
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Giải phóng tài nguyên
cap.release()
cv2.destroyAllWindows()