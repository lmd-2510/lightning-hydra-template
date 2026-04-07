import cv2
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from src.models.components.resnet18 import ResNet18Landmark

# --- A. ĐỊNH NGHĨA CLASS KALMAN FILTER CHO LANDMARKS ---
class KalmanLandmarkSmoother:
    def __init__(self, num_landmarks=98, process_noise=0.1, measurement_noise=0.05):
        self.num_landmarks = num_landmarks
        self.filters = []
        
        # process_noise (Q): Độ tin tưởng vào mô hình chuyển động (càng nhỏ càng mượt nhưng lag)
        # measurement_noise (R): Độ tin tưởng vào kết quả AI (càng lớn càng mượt nhưng lag)
        self.q = process_noise 
        self.r = measurement_noise

        for _ in range(num_landmarks):
            # State: [x, vx, y, vy] (vị trí và vận tốc cho x và y)
            kf = cv2.KalmanFilter(4, 2) 
            
            # Ma trận chuyển trạng thái (Transition Matrix)
            # x_new = x + vx * dt
            # vx_new = vx
            kf.transitionMatrix = np.array([
                [1, 1, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 0, 1]
            ], np.float32)
            
            # Ma trận đo lường (Measurement Matrix) - Chúng ta chỉ đo được x và y
            kf.measurementMatrix = np.array([
                [1, 0, 0, 0],
                [0, 0, 1, 0]
            ], np.float32)
            
            # Sai số quy trình (Process Noise Covariance)
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * self.q
            
            # Sai số đo lường (Measurement Noise Covariance)
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * self.r
            
            self.filters.append(kf)

    def update(self, pts):
        """
        pts: numpy array dạng (98, 2)
        """
        smoothed_pts = []
        for i in range(self.num_landmarks):
            kf = self.filters[i]
            
            # 1. Dự đoán vị trí tiếp theo
            prediction = kf.predict()
            
            # 2. Cập nhật với giá trị đo thực tế từ AI
            measurement = np.array([[np.float32(pts[i][0])], [np.float32(pts[i][1])]])
            corrected = kf.correct(measurement)
            
            # Lấy tọa độ x, y từ trạng thái đã hiệu chỉnh
            smoothed_pts.append([corrected[0][0], corrected[2][0]])
            
        return np.array(smoothed_pts)

    def reset(self, pts=None):
        """Reset lại bộ lọc khi mất dấu khuôn mặt hoặc bắt đầu khuôn mặt mới"""
        if pts is not None:
            for i in range(self.num_landmarks):
                self.filters[i].statePost = np.array([
                    [pts[i][0]], [0], [pts[i][1]], [0]
                ], np.float32)
        else:
            for kf in self.filters:
                kf.statePost = np.zeros((4, 1), np.float32)
                kf.statePre = np.zeros((4, 1), np.float32)

# ==============================================================================

print("--- ĐANG KHỞI TẠO HỆ THỐNG AI (VỚI KALMAN FILTER) ---")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Đang chạy trên thiết bị: {device}")

# --- B. KHỞI TẠO MODEL RESNET18 ---
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

# --- C. KHỞI TẠO MODEL YOLOv8 ---
print("⏳ Đang tải YOLOv8 Face Detection model...")
try:
    yolo_model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
    yolo_detector = YOLO(yolo_model_path)
    yolo_detector.to(device) 
    print("✅ Khởi động YOLOv8 Face Detector thành công!")
except Exception as e:
    print(f"❌ Lỗi khi tải YOLO: {e}")
    exit()

# --- D. KHỞI TẠO BỘ LỌC KALMAN ---
# process_noise nhỏ = mượt hơn nhưng phản ứng chậm hơn
# measurement_noise lớn = tin vào dự đoán hơn là tin vào AI (mượt hơn)
kf_smoother = KalmanLandmarkSmoother(num_landmarks=98, process_noise=0.005, measurement_noise=0.5)

def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, (256, 256))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
    return face_tensor.unsqueeze(0).to(device)

# --- E. CHẠY REAL-TIME ---
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
    
    if len(boxes) == 0:
        kf_smoother.reset()

    for i, box in enumerate(boxes):
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
        
        # --- BƯỚC QUAN TRỌNG: ÁP DỤNG KALMAN FILTER ---
        pts = kf_smoother.update(pts)
        
        # --- VẼ KẾT QUẢ ---
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Confidence: {conf_score:.2f} (Kalman: ON)"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        for (px, py) in pts:
            cv2.circle(frame, (int(px), int(py)), 2, (0, 0, 255), -1)
        
        break 

    cv2.imshow('YOLOv8 + ResNet18 WFLW (Kalman Filter)', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()