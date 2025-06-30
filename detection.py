import time
import os
import sys
import cv2
import torch
import psutil
import subprocess
from pathlib import Path
from collections import Counter

sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5'))

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import LoadImages
from yolov5.utils.general import (
    check_img_size, increment_path, non_max_suppression, scale_boxes, set_logging
)
from yolov5.utils.plots import colors
from yolov5.utils.torch_utils import select_device, time_sync

from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort

from graphs import bbox_rel, draw_boxes, plot_one_box

from torchvision import models, transforms
import torch.nn as nn
from PIL import Image

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

def get_gpu_memory():
    try:
        result = subprocess.check_output([
            'nvidia-smi', '--query-gpu=memory.used',
            '--format=csv,nounits,noheader'
        ], encoding='utf-8')
        return int(result.strip().split('\n')[0])
    except:
        return 'NA'

@torch.no_grad()
def detect(
    weights=ROOT / 'weights/yolov5s_best_200_fixed.pt',
    source="",
    data=ROOT / 'dataset.yaml',
    stframe=None,
    kpi_fps_text="", kpi_object_text="", kpi_summary_text="",
    sys_ram_text="", sys_cpu_text="", sys_gpu_text="",
    imgsz=(640, 640),
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device='',
    nosave=False,
    classes=None,
    agnostic_nms=False,
    augment=False,
    visualize=False,
    project=ROOT / 'runs/detect',
    name='exp',
    exist_ok=False,
    line_thickness=1,
    half=False,
    dnn=False,
    display_labels=False,
    config_deepsort="deep_sort_pytorch/configs/deep_sort.yaml",
    drift_conf_thres=0.75,
    save_drift_frames=False,
    drift_obj_list_text="", drift_count_text="", min_fps_text="", max_fps_text="",
    fps_warn="", fps_drop_warn_threshold=8
):
    
    save_img = not nosave

    # Initialize Deep SORT
    cfg = get_config()
    cfg.merge_from_file(config_deepsort)
    deepsort = DeepSort(
        cfg.DEEPSORT.REID_CKPT,
        max_dist=cfg.DEEPSORT.MAX_DIST,
        min_confidence=cfg.DEEPSORT.MIN_CONFIDENCE,
        nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP,
        max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        max_age=cfg.DEEPSORT.MAX_AGE,
        n_init=cfg.DEEPSORT.N_INIT,
        nn_budget=cfg.DEEPSORT.NN_BUDGET,
        use_cuda=torch.cuda.is_available()
    )
    
    # Set up directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)
    if save_drift_frames:
        os.makedirs("drift_frames", exist_ok=True)

    set_logging()
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data)
    stride, class_names, pt, jit = model.stride, model.names, model.pt, model.jit
    imgsz = check_img_size(imgsz, s=stride)

    if pt or jit:
        model.model.half() if half and device.type != 'cpu' else model.model.float()

    # Second-stage classifier
    classifier_model = models.resnet101(pretrained=False)
    classifier_model.fc = nn.Linear(classifier_model.fc.in_features, 3)
    classifier_model.load_state_dict(torch.load(ROOT / 'weights/resnet101_3.pt', map_location=device)['model'])
    classifier_model.to(device).eval()

    cls_names = ['empty', 'kakigori', 'non_empty']

    cls_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    video_path, video_writer = [None], [None]
    
    timings = [0.0, 0.0, 0.0]
    previous_time = time.time()
    frame_index, drift_frame_count = -1, 0
    total_class_counter, frame_class_counter = dict(), dict()
    drift_objects, min_fps, max_fps = [], 10000, -1

    for path, img_tensor, original_frame, video_cap, _ in dataset:
        frame_index += 1

        # Preprocessing
        t1 = time_sync()
        img_tensor = torch.from_numpy(img_tensor).to(device)
        img_tensor = img_tensor.half() if half else img_tensor.float()
        img_tensor /= 255.0
        if len(img_tensor.shape) == 3:
            img_tensor = img_tensor[None]
        t2 = time_sync()
        timings[0] += t2 - t1

        # Inference
        visualize_path = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
        predictions = model(img_tensor, augment=augment, visualize=visualize_path)
        t3 = time_sync()
        timings[1] += t3 - t2

        # Apply NMS
        predictions = non_max_suppression(predictions, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        timings[2] += time_sync() - t3

        for detections in predictions:
            frame = original_frame.copy()

            if len(detections):
                detections[:, :4] = scale_boxes(img_tensor.shape[2:], detections[:, :4], frame.shape).round()

                current_class_names, current_counts = [], []
                for class_id in detections[:, -1].unique():
                    count = (detections[:, -1] == class_id).sum()
                    current_class_names.append(class_names[int(class_id)])
                    current_counts.append(int(count.cpu()))
                frame_class_counter.update(dict(zip(current_class_names, current_counts)))
                total_class_counter = Counter(total_class_counter) + Counter(frame_class_counter)

                # Prepare Deep SORT inputs
                bboxes_xywh, confidences = [], []
                for *xyxy, conf, cls in detections:
                    x_c, y_c, w, h = bbox_rel(*xyxy)
                    bboxes_xywh.append([x_c, y_c, w, h])
                    confidences.append([conf.item()])
                    if conf < drift_conf_thres and class_names[int(cls)] not in drift_objects:
                        drift_objects.append(class_names[int(cls)])
                        if save_drift_frames:
                            cv2.imwrite(f"drift_frames/frame_{frame_index}.png", frame)
                            drift_frame_count += 1
                
                # Update tracking
                outputs = deepsort.update(torch.Tensor(bboxes_xywh), torch.Tensor(confidences), frame)
                if len(outputs) > 0:
                    draw_boxes(frame, outputs[:, :4], outputs[:, -1])

                # Draw boxes with label
                for *xyxy, conf, cls in reversed(detections):
                    if save_img or display_labels:

                        x1, y1, x2, y2 = map(int, xyxy)
                        cropped_img = frame[y1:y2, x1:x2]
                        if cropped_img.size != 0:
                            # Convert to PIL & apply transform
                            img_pil = Image.fromarray(cv2.cvtColor(cropped_img, cv2.COLOR_BGR2RGB))
                            input_tensor = cls_transform(img_pil).unsqueeze(0).to(device)

                            # Predict class
                            with torch.no_grad():
                                output = classifier_model(input_tensor)
                                pred_label = cls_names[output.argmax(1).item()]
                        else:
                            pred_label = 'Unknown'

                        # Append label to YOLO label
                        label = f'{class_names[int(cls)]} {conf:.2f} ({pred_label})'
                        plot_one_box(xyxy, frame, label=label, color=colors(int(cls), True), line_thickness=line_thickness)
            else:
                deepsort.increment_ages()

            # Save video
            if save_img:
                if video_path[0] != path:
                    video_path[0] = path
                    if isinstance(video_writer[0], cv2.VideoWriter):
                        video_writer[0].release()
                    fps = video_cap.get(cv2.CAP_PROP_FPS)
                    w = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    video_writer[0] = cv2.VideoWriter(str(save_dir / Path(path).name), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                video_writer[0].write(frame)
            
             # FPS Calculation
            current_fps = round(1 / round(time.time() - previous_time, 3), 1)
            previous_time = time.time()
        
        # Update System Info & KPIs
        sys_ram_text.write(f"{psutil.virtual_memory()[2]}%")
        sys_cpu_text.write(f"{psutil.cpu_percent()}%")
        sys_gpu_text.write(f"{get_gpu_memory()} MB")

        kpi_fps_text.write(f"{current_fps} FPS")
        if current_fps < fps_drop_warn_threshold:
            fps_warn.warning(f"FPS dropped below {fps_drop_warn_threshold}")
        kpi_object_text.write(frame_class_counter)
        kpi_summary_text.write(total_class_counter)

        drift_obj_list_text.write(drift_objects)
        drift_count_text.write(drift_frame_count)
        if current_fps < min_fps:
            min_fps_text.write(current_fps)
            min_fps = current_fps
        if current_fps > max_fps:
            max_fps_text.write(current_fps)
            max_fps = current_fps

        stframe.image(frame, channels="BGR", use_column_width=True)

    if save_img:
        print(f"Results saved to {save_dir}")
    if video_cap:
        video_cap.release()