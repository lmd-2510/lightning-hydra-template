import cv2
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from src.models.components.resnet18 import ResNet18Landmark

print("--- ĐANG KHỞI TẠO HỆ THỐNG AI ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang chạy trên thiết bị: {device}")

# --- A. KHỞI TẠO MODEL RESNET18 DỰ ĐOÁN LANDMARK ---
resnet_weight_path = "checkpoints/resnet18_wflw_weights.pth" 
model_landmark = ResNet18Landmark(num_landmarks=98, pretrained=False)

try:
    model_landmark.load_state_dict(torch.load(resnet_weight_path, map_location=device, weights_only=False))
    model_landmark = model_landmark.to(device)
    model_landmark.eval()
    print("✅ Tải Model ResNet18 Landmark thành công!")
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file trọng số ResNet tại {resnet_weight_path}.")
    exit()

# --- B. KHỞI TẠO MODEL YOLOv8 FACE DETECTION ---
print("⏳ Đang tải YOLOv8 Face Detection model...")
try:
    yolo_model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    yolo_detector = YOLO(yolo_model_path)
    yolo_detector.to(device) 
    print("✅ Khởi động YOLOv8 Face Detector thành công!")
except Exception as e:
    print(f"❌ Lỗi khi tải YOLO: {e}")
    exit()

def preprocess_face(face_img):
    """Xử lý ảnh khuôn mặt trước khi đưa vào ResNet"""
    face_resized = cv2.resize(face_img, (256, 256))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
    return face_tensor.unsqueeze(0).to(device)

# --- C. CHẠY REAL-TIME ---
cap = cv2.VideoCapture(0)
print("🎥 Bật Webcam! Nhấn 'q' để thoát.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h_frame, w_frame, _ = frame.shape
    
    # Chạy YOLOv8 phát hiện khuôn mặt
    yolo_results = yolo_detector(frame, verbose=False, conf=0.5)
    
    # Lấy danh sách boxes và confidences
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    confidences = yolo_results[0].boxes.conf.cpu().numpy()
    
    for i, box in enumerate(boxes):
        # 1. Lấy tọa độ và độ tự tin
        xmin_raw, ymin_raw, xmax_raw, ymax_raw = box
        conf_score = confidences[i]
        
        # 2. Tính toán Padding (Mở rộng khung 20% để khớp với cách train WFLW)
        width_yolo = xmax_raw - xmin_raw
        height_yolo = ymax_raw - ymin_raw
        pad_w, pad_h = int(width_yolo * 0.2), int(height_yolo * 0.2)
        
        x1, y1 = max(0, int(xmin_raw - pad_w)), max(0, int(ymin_raw - pad_h))
        x2, y2 = min(w_frame, int(xmax_raw + pad_w)), min(h_frame, int(ymax_raw + pad_h))
        
        # 3. Cắt và Dự đoán Landmark
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0: continue
        
        input_tensor = preprocess_face(face_crop)
        with torch.no_grad():
            landmarks = model_landmark(input_tensor) 
        
        # 4. Chuyển đổi tọa độ Landmark về khung hình gốc
        pts = landmarks.view(-1, 2).cpu().numpy()
        pts = pts / 256.0  # Normalize về 0-1 (dựa trên size 256 lúc train)
        pts[:, 0] = pts[:, 0] * (x2 - x1) + x1
        pts[:, 1] = pts[:, 1] * (y2 - y1) + y1
        
        # --- 5. VẼ KẾT QUẢ ---
        # Vẽ Bounding Box màu Xanh Lá (BGR: 0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        # Vẽ text Độ tự tin (Confidence)
        label = f"Confidence: {conf_score:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Vẽ 98 điểm Landmark màu Đỏ (BGR: 0, 0, 255)
        for (px, py) in pts:
            cv2.circle(frame, (int(px), int(py)), 2, (0, 0, 255), -1)

    cv2.imshow('YOLOv8 + ResNet18 WFLW', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()