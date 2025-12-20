from ultralytics import YOLO
from loguru import logger
import cv2
import os
from PIL import Image
import config
from util.tool import get_amount, get_date, get_num, get_title, get_qrcode_data

# 类别与输出字段映射（财务票据）
converter = {
    'qrcode': 'qrcode',
    'invoice_code': 'invoice_code',
    'invoice_number': 'invoice_number',
    'title': 'title',
    'bill_date': 'billing_date',
    'total': 'total_amount',
    'check_code': 'check_code',
    'amount_with_tax': 'amount_with_tax',
    'buy_title': 'buy_title',
    'sale_title': 'sale_title',
    'seal_1': 'seal_1',
    'seal_2': 'seal_2',
}

# 模型路径和推理尺寸，支持通过配置覆盖
pub_weights = getattr(config, "BILL_MODEL_PATH", "models/bill/11n/best.onnx")
pub_img_size = getattr(config, "BILL_MODEL_IMGSZ", 640)

# 选择设备
device = config.GPUID if getattr(config, "GPU", False) else "cpu"

# 跳过 OCR 的标签
SKIP_OCR_LABELS = {'qrcode', 'seal_1', 'seal_2'}

# 置信度阈值（可配置，默认 0.618）
CONFIDENCE_THRESHOLD = getattr(config, "BILL_CONFIDENCE_THRESHOLD", 0.618)

# 初始化 YOLO 模型
model = YOLO(pub_weights, task='detect')


def _process_label_text(label: str, text: str) -> str:
    """根据标签类型做后处理。"""
    if label == 'invoice_number':
        return get_num(text)
    if label == 'invoice_code':
        return get_num(text)[-12:]
    if label == 'bill_date':
        return get_date(text)
    if label in ('total', 'amount_with_tax'):
        return get_amount(text)
    if label in ('buy_title', 'sale_title'):
        return get_title(text)
    if label.startswith('seal_'):
        return "detected"
    return text.strip()


def bill_detection(img_numpy, invoice=None, context=None, saveImage=False):
    """
    财务票据检测与识别（参考 vat_detect 的流程，精简字段）。

    Args:
        img_numpy: 输入图像 (numpy array，BGR)
        invoice: 结果字典（可复用外部对象）
        context: OCR 上下文（需提供 ocr 与 batch_ocr）
        saveImage: 是否保存各字段裁剪图
    """
    if invoice is None:
        invoice = {}
    if context is None:
        return invoice
    # 获取类别名称和检测结果
    names = model.names
    results = model.predict(
        source=img_numpy,
        imgsz=pub_img_size,
        device=device,
        verbose=False,
        half=False,
    )

    im0 = img_numpy
    batch_ocr = context.batch_ocr
    detected = False

    if results and len(results) > 0:
        result = results[0]
        boxes = result.boxes

        if boxes is not None and len(boxes) > 0:
            labels = {}
            label_confidences = {}  # 保存每个标签的置信度
            converter_keys = set(converter.keys())

            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())  # 获取置信度
                label = names[cls]
                if label not in converter_keys:
                    continue

                x1, y1, x2, y2 = [int(v) for v in xyxy]

                if saveImage:
                    stock_fp = os.path.join('images', 'bill')
                    os.makedirs(stock_fp, exist_ok=True)
                    path = os.path.join(stock_fp, f'{label}.png')
                    cv2.imwrite(path, im0[y1:y2, x1:x2])

                labels[label] = [y1, y2, x1, x2]
                label_confidences[label] = conf  # 保存置信度

            if labels:
                # 裁剪区域
                label_images = {key: im0[v[0]:v[1], v[2]:v[3]] for key, v in labels.items()}

                # 批量 OCR，跳过无需 OCR 的标签
                ocr_label_keys, ocr_images = [], []
                for label, img_region in label_images.items():
                    if label in SKIP_OCR_LABELS:
                        continue
                    ocr_label_keys.append(label)
                    ocr_images.append(img_region)

                ocr_results = {}
                if ocr_images:
                    for label, text in zip(ocr_label_keys, batch_ocr(ocr_images)):
                        ocr_results[label] = text

                # 处理二维码
                if 'qrcode' in label_images:
                    try:
                        qr_text = get_qrcode_data(Image.fromarray(label_images['qrcode']))
                        if qr_text:
                            invoice['qrcode'] = qr_text
                            invoice['qrcode_conf'] = label_confidences.get('qrcode', 0.0)
                            detected = True
                    except Exception:
                        pass

                # 处理 OCR 结果
                for label, text in ocr_results.items():
                    invoice[converter[label]] = _process_label_text(label, text)
                    detected = True

                # SKIP_OCR_LABELS 中的其他标签：保存置信度
                # 只有当置信度 >= 阈值且不需要识别文本的标签，才设置成 'detected'
                # 但置信度字段始终设置
                for label in SKIP_OCR_LABELS:
                    if label in label_images:
                        if label == 'qrcode':
                            # qrcode 已在上面处理
                            continue
                        conf = label_confidences.get(label, 0.0)
                        # 置信度始终设置
                        invoice[f'{converter[label]}_conf'] = conf
                        # 只有置信度 >= 阈值时才设置 'detected'
                        if conf >= CONFIDENCE_THRESHOLD:
                            invoice[converter[label]] = 'detected'
                            detected = True

    if detected:
        # 补齐缺省字段，金额默认 0
        for key in converter.values():
            if key in ('total_amount', 'amount_with_tax'):
                invoice.setdefault(key, '¥ 0.00')
            else:
                invoice.setdefault(key, "")
        title = invoice.get('title') or ''
        if '据' in title:
            invoice['_bill_detected'] = True
        else:
            invoice['_bill_detected'] = False


    return invoice

