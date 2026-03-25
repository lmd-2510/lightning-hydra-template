import cv2
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from src.models.components.resnet18 import ResNet18Landmark
# Import class EMA
from src.utils.smoothing import PointEMA

print("--- ĐANG KHỞI TẠO HỆ THỐNG AI (CÓ LÀM MƯỢT EMA) ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang chạy trên thiết bị: {device}")

# --- A. KHỞI TẠO MODEL RESNET18 ---
resnet_weight_path = "checkpoints/resnet18_wflw_weights.pth" 
model_landmark = ResNet18Landmark(num_landmarks=98, pretrained=False)

try:
    model_landmark.load_state_dict(torch.load(resnet_weight_path, map_location=device, weights_only=False))
    model_landmark = model_landmark.to(device)
    model_landmark.eval()
    print("✅ Tải Model ResNet18 Landmark thành công!")
except FileNotFoundError:
    print(f"❌ Lỗi: Không tìm thấy file trọng số tại {resnet_weight_path}.")
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

# --- C. KHỞI TẠO BỘ LỌC EMA ---
# Bạn có thể điều chỉnh alpha ở đây: 0.2 là khá mượt, 0.4 là nhạy hơn
smoother = PointEMA(alpha=0.25)

def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, (256, 256))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
    return face_tensor.unsqueeze(0).to(device)

# --- D. CHẠY REAL-TIME ---
cap = cv2.VideoCapture(0)
print("🎥 Bật Webcam! Nhấn 'q' để thoát.")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    h_frame, w_frame, _ = frame.shape
    
    yolo_results = yolo_detector(frame, verbose=False, conf=0.5)
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    confidences = yolo_results[0].boxes.conf.cpu().numpy()
    
    # Nếu không phát hiện khuôn mặt nào, reset bộ lọc để tránh kéo vệt điểm
    if len(boxes) == 0:
        smoother.reset()

    for i, box in enumerate(boxes):
        # Chỉ áp dụng EMA cho khuôn mặt đầu tiên (hoặc lớn nhất) để demo ổn định nhất
        # Nếu muốn làm mượt nhiều mặt, cần một danh sách smoother riêng cho từng ID
        
        xmin_raw, ymin_raw, xmax_raw, ymax_raw = box
        conf_score = confidences[i]
        
        width_yolo = xmax_raw - xmin_raw
        height_yolo = ymax_raw - ymin_raw
        pad_w, pad_h = int(width_yolo * 0.2), int(height_yolo * 0.2)
        
        x1, y1 = max(0, int(xmin_raw - pad_w)), max(0, int(ymin_raw - pad_h))
        x2, y2 = min(w_frame, int(xmax_raw + pad_w)), min(h_frame, int(ymax_raw + pad_h))
        
        face_crop = frame[y1:y2, x1:x2]
        if face_crop.size == 0: continue
        
        input_tensor = preprocess_face(face_crop)
        with torch.no_grad():
            landmarks = model_landmark(input_tensor) 
        
        # Chuyển đổi tọa độ Landmark về khung hình gốc
        pts = landmarks.view(-1, 2).cpu().numpy()
        pts = pts / 256.0
        pts[:, 0] = pts[:, 0] * (x2 - x1) + x1
        pts[:, 1] = pts[:, 1] * (y2 - y1) + y1
        
        # --- BƯỚC QUAN TRỌNG: ÁP DỤNG EMA ---
        # Làm mượt tọa độ pts trước khi vẽ
        pts = smoother.update(pts)
        
        # --- VẼ KẾT QUẢ ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Confidence: {conf_score:.2f} (Smooth: ON)"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for (px, py) in pts:
            cv2.circle(frame, (int(px), int(py)), 2, (0, 0, 255), -1)
        
        # Trong ví dụ này ta chỉ xử lý mặt đầu tiên để smoother hoạt động chính xác
        break 

    cv2.imshow('YOLOv8 + ResNet18 WFLW (Stabilized)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()