import torch

def convert_ckpt_to_pth(ckpt_path, pth_path):
    # 1. Load file checkpoint
    print(f"Đang load file: {ckpt_path}...")
    checkpoint = torch.load(ckpt_path, map_location='cpu')

    # 2. Kiểm tra cấu trúc file ckpt
    # File .ckpt thường là một dictionary. Trọng số thường nằm trong key 'state_dict'
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
        print("Tìm thấy 'state_dict' trong checkpoint.")
    else:
        # Nếu không thấy 'state_dict', có thể file này vốn đã chỉ chứa trọng số
        state_dict = checkpoint
        print("Không tìm thấy 'state_dict', giả định file chứa trực tiếp trọng số.")

    # 3. Xử lý Prefix (Rất quan trọng)
    # PyTorch Lightning thường thêm tiền tố "model." hoặc "net." vào trước tên các layer
    # Ví dụ: "model.conv1.weight" -> "conv1.weight"
    # Bước này giúp bạn load vào mô hình PyTorch thuần dễ dàng hơn.
    
    new_state_dict = {}
    for key, value in state_dict.items():
        # Kiểm tra nếu key bắt đầu bằng 'model.' hoặc 'net.' thì xóa nó đi
        if key.startswith('model.'):
            new_key = key[6:] # Xóa 6 ký tự đầu 'model.'
        elif key.startswith('net.'):
            new_key = key[4:] # Xóa 4 ký tự đầu 'net.'
        else:
            new_key = key
        new_state_dict[new_key] = value

    # 4. Lưu thành file .pth
    torch.save(new_state_dict, pth_path)
    print(f"Đã chuyển đổi và lưu tại: {pth_path}")

# ==========================================
# SỬ DỤNG
# ==========================================
ckpt_file = "checkpoints/epoch_008.ckpt"  # Đường dẫn file của bạn
pth_file = "checkpoints/epoch_008.pth"    # Tên file muốn lưu

convert_ckpt_to_pth(ckpt_file, pth_file)