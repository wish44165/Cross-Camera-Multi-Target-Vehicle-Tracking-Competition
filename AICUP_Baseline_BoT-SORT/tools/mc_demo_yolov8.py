import argparse
import time
from pathlib import Path
import sys
from ultralytics import YOLO
import cv2
import torch
import torch.backends.cudnn as cudnn
import os
import glob
import numpy as np

from tqdm import tqdm
from numpy import random

sys.path.insert(0, './yolov8')
sys.path.append('.')

from yolov8.utils.general import set_logging, increment_path
from yolov8.utils.plots import plot_one_box
from yolov8.utils.torch_utils import select_device, time_synchronized

from tracker.mc_bot_sort import BoTSORT
from tracker.tracking_utils.timer import Timer

def detect(save_img=False):
    source, weights, view_img, save_txt, imgsz, trace = opt.source, opt.weights, opt.view_img, opt.save_txt, opt.img_size, opt.trace
    save_img = not opt.nosave and not source.endswith('.txt')  # save inference images
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Initialize
    set_logging()
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    print(opt.weights)
    # Load a model
    model = YOLO(weights)  # pretrained YOLOv8n model
    # data_dir = glob.glob(os.path.join(source, '*.jpg')) + glob.glob(os.path.join(source, '*.jpeg')) + glob.glob(os.path.join(source, '*.png'))
    data_dir = sorted(glob.glob(os.path.join(source, '*.*')))

    # Get names and colors
    names = model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(100)]

    # Create tracker
    tracker = BoTSORT(opt, frame_rate=30.0)

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz), device=device)  # run once
        
    t0 = time.time()
    
    # Process detections
    results = []
    frameID = 0

    for path in tqdm(data_dir, desc=f'tracking {opt.name}'):
        p = path
        frameID += 1
        img = cv2.imread(path)
        # Run batched inference on a list of images
        preds = model(img, device=device)  # return a list of Results objects

        # Run tracker
        detections = []
        if  preds and  preds[0].boxes:
            # print( preds)
            boxes =  preds[0].boxes.xyxy.cpu().numpy()  # Boxes object for bounding box outputs
            # print(boxes)
            conf =  preds[0].boxes.conf.cpu().numpy().reshape(-1, 1)
            # print(conf)
            zero_array = np.zeros((boxes.shape[0], 1))

            # 在每個陣列的右側合併 zero array
            detections = np.hstack((boxes, conf, zero_array))

        # print(detections)
        online_targets = tracker.update(detections, img, frameID)

        online_tlwhs = []
        online_ids = []
        online_scores = []
        online_cls = []
        for t in online_targets:
            tlwh = t.tlwh
            tlbr = t.tlbr
            tid = t.track_id
            tcls = t.cls
            if tlwh[2] * tlwh[3] > opt.min_box_area:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                online_cls.append(t.cls)

                if save_img or view_img:  # Add bbox to image
                    if opt.hide_labels_name:
                        label = f'{tid}, {int(tcls)}'
                    else:
                        label = f'{tid}, {names[int(tcls)]}'

                    if 'car' in label: # AICUP only have one cls: car
                        # save results
                        results.append(
                            f"{frameID},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                        )
                        plot_one_box(tlbr, img, label=label, color=colors[int(tid) % len(colors)], line_thickness=2)

                            
            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg

            # Print time (inference + NMS)
            # print(f'{s}Done. ({t2 - t1:.3f}s)')

            # Stream results
            if view_img:
                cv2.imshow('BoT-SORT', img)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                cv2.imwrite(save_path, img)

    if save_txt or save_img:
        with open(save_dir / f"{opt.name}.txt", 'w') as f:
            f.writelines(results)
            
        print(f"Results saved to {save_dir}")

    print(f'Done. ({time.time() - t0:.3f}s)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='yolov8.pt', help='model.pt path(s)')
    parser.add_argument('--source', type=str, default='inference/images', help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=1920, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.09, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.7, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--update', action='store_true', help='update all models')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    parser.add_argument('--trace', action='store_true', help='trace model')
    parser.add_argument('--hide-labels-name', default=False, action='store_true', help='hide labels')

    # tracking args
    parser.add_argument("--track_high_thresh", type=float, default=0.3, help="tracking confidence threshold")
    parser.add_argument("--track_low_thresh", default=0.05, type=float, help="lowest detection threshold")
    parser.add_argument("--new_track_thresh", default=0.4, type=float, help="new track thresh")
    parser.add_argument("--track_buffer", type=int, default=30, help="the frames for keep lost tracks")
    parser.add_argument("--match_thresh", type=float, default=0.7, help="matching threshold for tracking")
    parser.add_argument("--aspect_ratio_thresh", type=float, default=1.6,
                        help="threshold for filtering out boxes of which aspect ratio are above the given value.")
    parser.add_argument('--min_box_area', type=float, default=10, help='filter out tiny boxes')
    parser.add_argument("--fuse-score", dest="mot20", default=False, action="store_true",
                        help="fuse score and iou for association")

    # CMC
    parser.add_argument("--cmc-method", default="sparseOptFlow", type=str, help="cmc method: sparseOptFlow | files (Vidstab GMC) | orb | ecc")

    # ReID
    parser.add_argument("--with-reid", dest="with_reid", default=False, action="store_true", help="with ReID module.")
    parser.add_argument("--fast-reid-config", dest="fast_reid_config", default=r"fast_reid/configs/MOT17/sbs_S50.yml",
                        type=str, help="reid config file path")
    parser.add_argument("--fast-reid-weights", dest="fast_reid_weights", default=r"pretrained/mot17_sbs_S50.pth",
                        type=str, help="reid config file path")
    parser.add_argument('--proximity_thresh', type=float, default=0.5,
                        help='threshold for rejecting low overlap reid matches')
    parser.add_argument('--appearance_thresh', type=float, default=0.25,
                        help='threshold for rejecting low appearance similarity reid matches')

    opt = parser.parse_args()

    opt.jde = False
    opt.ablation = False

    print(opt)
    # check_requirements(exclude=('pycocotools', 'thop'))

    with torch.no_grad():
        detect()