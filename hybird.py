import cv2
import torch
import numpy as np
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from src.models.components.resnet18 import ResNet18Landmark

# ==============================================================================
# 1. KALMAN FILTER - Lớp làm mượt cuối cùng
# ==============================================================================
class KalmanLandmarkSmoother:
    def __init__(self, num_landmarks=98, process_noise=0.01, measurement_noise=0.1):
        self.num_landmarks = num_landmarks
        self.filters = []
        for _ in range(num_landmarks):
            kf = cv2.KalmanFilter(4, 2) 
            kf.transitionMatrix = np.array([[1, 0.1, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0.1], [0, 0, 0, 1]], np.float32)
            kf.measurementMatrix = np.array([[1, 0, 0, 0], [0, 0, 1, 0]], np.float32)
            kf.processNoiseCov = np.eye(4, dtype=np.float32) * process_noise
            kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * measurement_noise
            self.filters.append(kf)

    def update(self, pts):
        smoothed_pts = []
        for i in range(self.num_landmarks):
            kf = self.filters[i]
            prediction = kf.predict()
            measurement = np.array([[np.float32(pts[i][0])], [np.float32(pts[i][1])]])
            corrected = kf.correct(measurement)
            smoothed_pts.append([corrected[0][0], corrected[2][0]])
        return np.array(smoothed_pts)

    def reset(self, pts):
        for i in range(self.num_landmarks):
            self.filters[i].statePost = np.array([[pts[i][0]], [0], [pts[i][1]], [0]], np.float32)

# ==============================================================================
# 2. KHỞI TẠO HỆ THỐNG AI
# ==============================================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet_weight_path = "checkpoints/resnet18_wflw_weights.pth" 
model_landmark = ResNet18Landmark(num_landmarks=98, pretrained=False)
model_landmark.load_state_dict(torch.load(resnet_weight_path, map_location=device))
model_landmark = model_landmark.to(device).eval()

yolo_model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
yolo_detector = YOLO(yolo_model_path).to(device)

# Khởi tạo bộ lọc và biến lưu trữ
kf_smoother = KalmanLandmarkSmoother(process_noise=0.01, measurement_noise=0.05)
prev_gray = None
prev_pts = None
frame_count = 0
AI_UPDATE_INTERVAL = 5 # Cứ 5 frame thì chạy AI một lần, các frame khác dùng Optical Flow

def preprocess_face(face_img):
    face_resized = cv2.resize(face_img, (256, 256))
    face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
    face_tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float() / 255.0
    return face_tensor.unsqueeze(0).to(device)

# ==============================================================================
# 3. VÒNG LẶP CHÍNH (INFERENCE PIPELINE)
# ==============================================================================
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    h_frame, w_frame, _ = frame.shape
    
    # QUYẾT ĐỊNH: Dùng AI hay dùng Optical Flow?
    use_ai = (frame_count % AI_UPDATE_INTERVAL == 0) or (prev_pts is None)
    
    current_pts = None

    if use_ai:
        # --- BƯỚC 1: AI DETECTION (YOLO + ResNet) ---
        yolo_results = yolo_detector(frame, verbose=False, conf=0.5)
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes) > 0:
            box = boxes[0] # Lấy mặt đầu tiên
            xmin, ymin, xmax, ymax = box
            pad_w, pad_h = int((xmax-xmin)*0.2), int((ymax-ymin)*0.2)
            x1, y1 = max(0, int(xmin-pad_w)), max(0, int(ymin-pad_h))
            x2, y2 = min(w_frame, int(xmax+pad_w)), min(h_frame, int(ymax+pad_h))
            
            face_crop = frame[y1:y2, x1:x2]
            if face_crop.size > 0:
                input_tensor = preprocess_face(face_crop)
                with torch.no_grad():
                    landmarks = model_landmark(input_tensor)
                
                pts = landmarks.view(-1, 2).cpu().numpy() / 256.0
                pts[:, 0] = pts[:, 0] * (x2 - x1) + x1
                pts[:, 1] = pts[:, 1] * (y2 - y1) + y1
                current_pts = pts
                
                # Reset Kalman khi AI chạy để tránh kéo vệt
                kf_smoother.reset(current_pts)
        
    else:
        # --- BƯỚC 2: FEATURE TRACKING (Optical Flow) ---
        if prev_pts is not None and prev_gray is not None:
            # Chuẩn bị điểm cho Lucas-Kanade (phải là float32, shape (N, 1, 2))
            p0 = np.array([prev_pts], dtype=np.float32)
            
            # Tính toán Optical Flow
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, 
                                                   winSize=(15, 15), maxLevel=2, 
                                                   criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
            
            if p1 is not None:
                # Lấy ra các điểm được track thành công
                tracked_pts = p1[0]
                # Những điểm track thất bại thì giữ nguyên vị trí cũ
                for i in range(len(prev_pts)):
                    if st[i] == 0:
                        tracked_pts[i] = prev_pts[i]
                current_pts = tracked_pts

    # --- BƯỚC 3: KALMAN FILTER (Final Fusion) ---
    if current_pts is not None:
        final_pts = kf_smoother.update(current_pts)
        
        # Vẽ kết quả
        for (px, py) in final_pts:
            cv2.circle(frame, (int(px), int(py)), 2, (0, 0, 255), -1)
        
        # Lưu lại cho frame sau
        prev_pts = final_pts
    else:
        prev_pts = None

    # Cập nhật trạng thái
    prev_gray = gray
    frame_count += 1
    
    # Hiển thị mode hiện tại
    mode_text = "Mode: AI" if use_ai else "Mode: Optical Flow"
    cv2.putText(frame, mode_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Hybrid Tracking Pipeline', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'): break

cap.release()
cv2.destroyAllWindows()