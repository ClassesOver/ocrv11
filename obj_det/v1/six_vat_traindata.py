import base64
import io
import os

from PIL import Image
from paddleocr_ import img_decode
from util.tool import convert2yolov5

invoice_dir = '/Users/fengzetao/PycharmProjects/yolov5/six_invoice/train/images'
labels_dir = '/Users/fengzetao/PycharmProjects/yolov5/six_invoice/train/labels'

import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device
import threading
import numpy as np

lock = threading.Lock()

converter = {'invoice_code': 'invoice_code',
             'invoice_code2': 'invoice_code2',
             'invoice_number': 'invoice_number',
             'invoice_number2': 'invoice_number2',
             'bill_date': 'billing_date',
             'check_code': 'check_code',
             'check_code2': 'check_code2',
             'title': 'title',
             'total': 'total_amount',
             'tax': 'tax',
             'amount_with_tax': 'amount_with_tax',
             'invoice_type': 'invoice_type',
             'buy_title': 'buy_title',
             'buy_tax': 'buy_tax',
             'buy_addr': 'buy_addr',
             'buy_bank': 'buy_bank',
             'sale_title': 'sale_title',
             'sale_tax': 'sale_tax',
             'sale_addr': 'sale_addr',
             'sale_bank': 'sale_bank'
             }

pub_weights = '/Users/fengzetao/PycharmProjects/ocr/code/secondary_train/six_invoice/best_complex.pt'
pub_img_size = 640
pub_augment = False
pub_conf_thres = 0.25
pub_iou_thres = 0.24
pub_classes = None
pub_agnostic_nms = False

# Initialize
set_logging()
pub_device = 'cpu'
device = select_device(pub_device)
half = device.type != 'cpu'  # half precision only supported on CUDA
model = attempt_load(pub_weights, device='cpu')  # load FP32 model
stride = int(model.stride.max())  # model stride


def invoice_detection(img_numpy):
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
    result = []
    for i, det in enumerate(pred):  # detections per image
        im0 = img_numpy
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            for *xyxy, conf, cls in reversed(det):
                l = list(convert2yolov5((im0.shape[1], im0.shape[0]), xyxy))
                l.insert(0, int(cls))
                result.append(l)
    result.sort(key=lambda x: x[0])
    return "\n".join([" ".join(str(y) for y in x) for x in result])

def image_to_base64(file_path):
    image = Image.open(file_path)
    byte_stream = io.BytesIO()
    image.save(byte_stream, format="PNG")  # 可根据需要修改保存格式
    return byte_stream.getvalue()

def train(*args, **kwargs):
    for file in os.listdir(invoice_dir):
        file_path = os.path.join(invoice_dir, file)
        if not os.path.isdir(file_path):
            print(f'处理{file}')
            text = invoice_detection(img_decode(image_to_base64(file_path)))
            with open(os.path.join(labels_dir, file.replace(".jpg", ".txt")), "w") as f:
                f.write(text)
