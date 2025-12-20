import uuid

import numpy
import numpy as np
import time
from pathlib import Path
import os

import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random

from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_imshow, non_max_suppression, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging
from utils.torch_utils import select_device, time_synchronized
import config
import threading
from utils.datasets import letterbox

scale, maxScale = config.IMGSIZE[0], 2048

lock = threading.Lock()
converter_name = {'01': '增值税专用发票', '04': '增值税普通发票', '08': '增值税电子专用发票', '10': '增值税电子普通发票', '11': '增值税普通发票（卷式）',
                  '14': '增值税电子普通发票（通行费）', '': '未识别的发票'}
converter = {'增值税专用发票': '01', '增值税普通发票': '04', '增值税电子专用发票': '08', '增值税电子普通发票': '10', '增值税普通发票（卷式）': '11',
             '增值税电子普通发票（通行费）': '14'}
pub_weights = "models/title/best.pt"
pub_view_img = False
pub_save_txt = False
pub_img_size = 640
pub_nosave = False
pub_project = "images"
pub_augment = False
pub_conf_thres = 0.25
pub_iou_thres = 0.24
pub_classes = None
pub_agnostic_nms = False
pub_save_conf = False
from loguru import logger

# Initialize
set_logging()
if config.GPU:
    pub_device = "%s" % config.GPUID
else:
    pub_device = 'cpu'
device = select_device(pub_device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
if config.GPU:
    model = attempt_load(pub_weights, device=lambda storage, loc: storage.cuda(config.GPUID))
else:
    model = attempt_load(pub_weights, device=pub_device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(pub_img_size, s=stride)  # check img_size
if half:
    model.half()  # to FP16


def invoice_number_process(img):
    height, width, _ = img.shape
    for i in range(height):
        for j in range(width):
            dot = img[i, j]
            dot0 = dot[0]
            dot1 = dot[1]
            dot2 = dot[2]
            if dot2 < dot1 or dot2 < dot0:
                img[i, j] = [0, 0, 0]
                continue
    return img


def invoice_detection_image(img_numpy, invoice=None, context=None):
    title = ""
    names = model.module.names if hasattr(model, 'module') else model.names
    img = letterbox(img_numpy, pub_img_size, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    lock.acquire(timeout=3)
    # Inference
    pred = model(img, augment=pub_augment)[0]
    lock.release()

    # Apply NMS
    pred = non_max_suppression(pred, pub_conf_thres, pub_iou_thres, classes=pub_classes, agnostic=pub_agnostic_nms)
    # Process detections
    for i, det in enumerate(pred):  # detections per image
        im0 = img_numpy
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            # Write results
            for *xyxy, conf, cls in reversed(det):
                label = names[int(cls)]
                newimg = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]): int(xyxy[2])]
                if config.SaveImg:
                    invoice_fp = os.path.join('images', 'invoice')
                    path = os.path.join(invoice_fp, '%s.png' % label)
                    cv2.imwrite(path, newimg)
                if "title" == label:
                    title = context.chineseModel(newimg)
                if "go" == label:
                    title = "增值税电子普通发票（通行费）"
            invoice['type'] = '01'
            # id = uuid.uuid4()
            # invoice_fp = '/Users/fengzetao/PycharmProjects/yolov5/eggs'
            # img_path = os.path.join(invoice_fp, 'images', 'train', '%s.png' % id)
            # label_path = os.path.join(invoice_fp, 'labels', 'train', '%s.txt' % id)
            # result = context.paddleOCR.get_text_boxs(im0)
            # labels = {}
            # for res in result:
            #     text = res[1][0]
            #     box = res[0]
            #     if '大学附属华山' in text:
            #         labels[0] = box
            #     elif '材料入库' in text:
            #         labels[1] = box
            #     elif 'No' in text:
            #         labels[2] = box
            # cv2.imwrite(img_path, im0)
            # with open(label_path, 'w') as f:
            #     for key, value in labels.items():
            #         f.write('{0} {1} {2} {3} {4}\n'.format(key, value[0][0], value[0][1], value[2][0], value[2][1]))
    invoice_result = invoice['invoice']
    if "专用发票" in title:
        if "电子" in title:
            invoice_result['invoice_type'] = "08"
        else:
            invoice_result['invoice_type'] = "01"
    if "普通发票" in title:
        if "电子" in title:
            invoice_result['invoice_type'] = "10"
        else:
            invoice_result['invoice_type'] = "04"
    if "增值税电子普通发票（通行费）" in title:
        invoice_result['invoice_type'] = "14"
    invoice_result['invoice_type_name'] = converter_name[invoice_result['invoice_type']]
    invoice_result['title'] = title
    return invoice


def invoice_detection(file_name=None, invoice=None, context=None):
    if isinstance(file_name, numpy.ndarray):
        return invoice_detection_image(file_name, invoice, context)
    title = ""
    pub_source = file_name
    source, weights, view_img, save_txt, imgsz = pub_source, pub_weights, pub_view_img, pub_save_txt, pub_img_size
    save_img = True
    webcam = source.isnumeric() or source.endswith('.txt') or source.lower().startswith(
        ('rtsp://', 'rtmp://', 'http://', 'https://'))

    # Directories
    save_dir = Path(pub_project)  # increment run

    # Set Dataloader
    vid_path, vid_writer = None, None
    if webcam:
        view_img = check_imshow()
        cudnn.benchmark = True  # set True to speed up constant image size inference
        dataset = LoadStreams(source, img_size=imgsz, stride=stride)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    # Run inference
    if device.type != 'cpu':
        model(torch.zeros(1, 3, imgsz, imgsz).to(device).type_as(next(model.parameters())))  # run once
    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        lock.acquire(timeout=3)
        # Inference
        pred = model(img, augment=pub_augment)[0]
        lock.release()

        # Apply NMS
        pred = non_max_suppression(pred, pub_conf_thres, pub_iou_thres, classes=pub_classes, agnostic=pub_agnostic_nms)
        t2 = time_synchronized()

        # Process detections
        for i, det in enumerate(pred):  # detections per image
            if webcam:  # batch_size >= 1
                p, s, im0, frame = path[i], '%g: ' % i, im0s[i].copy(), dataset.count
            else:
                p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            save_path = str(save_dir / p.name)  # img.jpg
            txt_path = str(save_dir / 'labels' / p.stem) + ('' if dataset.mode == 'image' else f'_{frame}')  # img.txt
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf) if pub_save_conf else (cls, *xywh)  # label format
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img or view_img:  # Add bbox to image
                        label = names[int(cls)]
                        newimg = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]): int(xyxy[2])]
                        invoice_fp = os.path.join('images', 'invoice')
                        path = os.path.join(invoice_fp, '%s.png' % label)
                        cv2.imwrite(path, newimg)
                        if "title" == label:
                            title = context.chineseModel(newimg)
                        if "go" == label:
                            # go = context.chineseModel(newimg)
                            title = "增值税电子普通发票（通行费）"
            else:
                height = im0.shape[0]
                width = im0.shape[1]
                xyxy = [int(width / 2 - 120), 0, int(width / 2 + 150), int(height / 7)]
                newimg = im0[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]): int(xyxy[2])]
                if config.SaveImg:
                    qrcodes_fp = os.path.join('images', 'stock')
                    path = os.path.join(qrcodes_fp, '%s.png' % "入库单Title")
                    cv2.imwrite(path, newimg)
                invoice['is_stock'] = True
                result = context.paddleOCR.get_text_boxs(newimg)
                if result:
                    invoice['is_stock'] = True
                    for text in [x[1][0] for x in result]:
                        if '入库' in text:
                            invoice['stock']['type'] = text
                        else:
                            invoice['stock']['hospital'] = text
    invoice_result = invoice['invoice']
    if "专用发票" in title:
        if "电子" in title:
            invoice_result['invoice_type'] = "08"
        else:
            invoice_result['invoice_type'] = "01"
    if "普通发票" in title:
        if "电子" in title:
            invoice_result['invoice_type'] = "10"
        else:
            invoice_result['invoice_type'] = "04"
    if "增值税电子普通发票（通行费）" in title:
        invoice_result['invoice_type'] = "14"
    if invoice['is_stock']:
        return invoice

    invoice_result['invoice_type_name'] = converter_name[invoice_result['invoice_type']]
    invoice_result['title'] = title
    return invoice
