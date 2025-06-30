import torch
import pathlib
import sys
import os

# Thêm yolov5 vào PYTHONPATH nếu nó chứa models/yolo.py
yolov5_path = os.path.join(os.getcwd(), 'yolov5')  # đường dẫn đến thư mục yolov5
if yolov5_path not in sys.path:
    sys.path.append(yolov5_path)

# Đảm bảo tương thích đường dẫn PosixPath
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

# Load model
model = torch.load("weights/best_200.pt", map_location="cpu", weights_only=False)

# Save lại model để load dễ hơn lần sau
torch.save(model, "weights/best_200_fixed.pt")
