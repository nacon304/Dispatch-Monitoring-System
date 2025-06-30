# Dish Tracking System using YOLOv5, DeepSORT, and ResNet101

This project implements a real-time dish tracking and classification system using:

- **YOLOv5** for dish & tray detection
- **DeepSORT** for tracking dishes and trays across frames
- **ResNet101** (fine-tuned) for classifying dish states: `empty`, `kakigori`, `non_empty`

The system is containerized using **Docker Compose** for easy setup and deployment.

Here is an example of the system in action:
![Dish Tracking Screenshot](screenshot.png)

---

## 🚀 Features

- ✅ Dish detection using YOLOv5
- ✅ Multi-object tracking using DeepSORT
- ✅ Second-stage classification with a ResNet101 model
- ✅ Real-time inference with FPS, CPU, RAM, GPU usage display
- ✅ Drift handling: saves low-confidence frames
- ✅ Dockerized deployment with a single command

---

## 📦 Installation

### Prerequisites

- [Docker](https://docs.docker.com/get-docker/)
- [Docker Compose](https://docs.docker.com/compose/install/)
- (Optional) NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

### Setup

```bash
git clone https://github.com/nacon304/Dispatch-Monitoring-System.git
cd Dispatch-Monitoring-System
```

Download models from [Google Drive](https://drive.google.com/drive/folders/1vaO-CN56M2cj_AeNe5R-ZRK0PNIO289n?usp=sharing) and place it in the `weights/` folder.

### ✅ Place in `weights/` folder:

- `weights/yolov5s_best_200_fixed.pt` → YOLOv5 object detector
- `weights/resnet101_3.pt` → ResNet101 classifier

```
Dispatch-Monitoring-System/
├── weights/
    ├── yolov5s_best_200_fixed.pt
    └── resnet101_3.pt
```

### ✅ Place in DeepSORT checkpoint folder:

- `ckpt.t7` → place in:

```
deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7
```

> This file is required by DeepSORT for object re-identification.

---

## ▶️ Run the System

### 🐳 Option 1: Run with Docker

1. **Download the Docker image**  
   Download `dispatch-monitoring-system.tar` from [Google Drive](https://drive.google.com/drive/folders/1s9IpOeQ3Dfv9G7xYQPSWZJC9yLCzM7cP?usp=sharing)

2. **Load the Docker image**

   ```bash
   docker load -i dispatch-monitoring-system.tar
   ```

3. **Run the container**
   ```bash
   docker run --rm -p 8501:8501 dispatch-monitoring-system
   ```

### 💻 Option 2: Run Locally with Streamlit

Run the app with:

```bash
streamlit run app.py
```

---

## 🧪 Output

- Bounding boxes with detection and classification labels
- Tracking ID for each dish
- Realtime system stats (FPS, CPU, RAM, GPU)
- Optional output video saved to `runs/detect/`
- Low-confidence frames saved to `drift_frames/`

---
