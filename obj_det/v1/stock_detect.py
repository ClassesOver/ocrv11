# -*- coding: utf-8 -*-
# @Time    : 2022/10/19 23:42
# @Author  : FengZeTao zetao.feng@hand-china.com
# @File    : stock_detect.py
# 如遇到看不懂且无备注的部分，请联系相关人
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
from utils.datasets import LoadStreams, LoadImages, letterbox
from utils.general import check_img_size, check_imshow, non_max_suppression, \
    scale_coords, xyxy2xywh, set_logging
from utils.torch_utils import select_device, time_synchronized
import threading
import config
from util.tool import *

lock = threading.Lock()

converter = {'code': 'code', 'amount': 'total_amount', 'invoice_number': 'invoice_number', 'page': 'page'}

pub_weights = "models/stock/best.pt"
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


def calculate_coordinates(xyxy, fault_tolerant):
    result = []
    for index, x in enumerate(xyxy):
        i = int(x)
        if index < 2:
            if i >= fault_tolerant:
                i -= fault_tolerant
        else:
            i += fault_tolerant
        result.append(i)
    return result


def stock_detection_image(img_numpy, stock=None, context=None):
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
            fault_tolerant = 10
            labels = {}
            for *xyxy, conf, cls in reversed(det):
                label = names[int(cls)]
                xyxy = calculate_coordinates(xyxy, fault_tolerant)
                newimg = im0[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                if config.SaveImg:
                    qrcodes_fp = os.path.join('images', 'stock')
                    path = os.path.join(qrcodes_fp, '%s.png' % label)
                    cv2.imwrite(path, newimg)
                labels[label] = newimg
            has_qrcode = False
            if labels.__contains__("qrcode"):
                has_qrcode = qrcode_pyzbar(labels.get('qrcode'), stock, True)
            if not has_qrcode:
                invoice_list = []
                for label, newimg in labels.items():
                    if label == 'qrcode':
                        continue
                    text = context.paddleOCR.get_text(newimg, 0, 1).strip()
                    if label == 'invoice_number' and len(text) == 8:
                        invoice_list.append(text)
                    elif label == 'amount':
                        stock[converter[label]] = get_float(text)
                    elif label == 'page':
                        stock[converter[label]] = get_page(text)
                    else:
                        stock[converter[label]] = get_num(text)
                if invoice_list:
                    stock['invoice_number'] = '、'.join(set(invoice_list))
            for val in converter.values():
                if stock.get(val) is None:
                    if val == 'page':
                        value = "-1/-1"
                    else:
                        value = ""
                    stock[val] = value
    return stock
