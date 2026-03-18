import cv2
import os
from tqdm import tqdm

def flip_muct_images(data_root):
    # Danh sách 5 thư mục ảnh dựa trên cấu trúc của bạn
    img_folders = [
        "muct-a-jpg-v1/jpg", # Thư mục này đặc biệt có thêm subfolder /jpg
        "muct-b-jpg-v1/jpg",
        "muct-c-jpg-v1/jpg",
        "muct-d-jpg-v1/jpg",
        "muct-e-jpg-v1/jpg"
    ]

    for folder in img_folders:
        folder_path = os.path.join(data_root, folder)
        if not os.path.exists(folder_path):
            print(f"Bỏ qua: {folder_path} (Không tồn tại)")
            continue

        print(f"Đang xử lý thư mục: {folder}")
        
        # Duyệt qua các file trong thư mục
        for filename in tqdm(os.listdir(folder_path)):
            # Chỉ xử lý ảnh gốc (bắt đầu bằng 'i' và không phải 'ir')
            if filename.startswith('i') and not filename.startswith('ir') and filename.endswith('.jpg'):
                img_path = os.path.join(folder_path, filename)
                
                # Đọc ảnh
                img = cv2.imread(img_path)
                if img is None:
                    continue
                
                # Lật ảnh theo trục dọc (trục tung - Horizontal Flip)
                # Trong OpenCV, flipCode = 1 là lật ngang
                flipped_img = cv2.flip(img, 1)
                
                # Tạo tên mới: i...jpg -> ir...jpg
                new_filename = 'ir' + filename[1:]
                new_path = os.path.join(folder_path, new_filename)
                
                # Lưu ảnh nếu chưa tồn tại
                if not os.path.exists(new_path):
                    cv2.imwrite(new_path, flipped_img)

if __name__ == "__main__":
    # Đường dẫn trỏ thẳng vào thư mục muct của bạn
    DATA_DIR = r"data\muct"
    flip_muct_images(DATA_DIR)
    print("Hoàn thành! Bây giờ bạn đã có đầy đủ ảnh để chạy train.")