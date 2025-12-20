from ultralytics import YOLO
import numpy as np
import cv2
import os
import re
from PIL import Image
from loguru import logger

import config
from util.tool import get_amount, get_chinese_amount, get_date, get_num, get_page, get_qrcode_data

# 入库单类别映射（保持与模型标签一致）
converter = {
    'title': 'title',
    'qrcode': 'qrcode',
    'supplier': 'supplier',
    'total': 'total',
    'idate': 'idate',
    'doc_number': 'doc_number',
    'fs': 'fs',
    'rk_way': 'rk_way',
    'hs_categ': 'hs_categ',
    'total2': 'total2',
    'verified_by': 'verified_by',
    'handled_by': 'handled_by',
    'accountant': 'accountant',
    'seal': 'seal',
    'line': 'line',
    'cnt': 'cnt',
    'page': 'page',
    'total3': 'total3',
    'note': 'note',
}

pub_weights = os.getenv("STOCK_V1_WEIGHTS", "models/stock_1/11m/best.onnx")
pub_img_size = getattr(config, "STOCK_V1_IMGSZ", 640)

# 初始化 YOLO 模型
device = getattr(config, "GPUID", 0) if getattr(config, "GPU", False) else "cpu"
model = YOLO(pub_weights, task='detect')

# 置信度阈值（可配置，默认 0.618）
CONFIDENCE_THRESHOLD = getattr(config, "STOCK_V1_CONFIDENCE_THRESHOLD", 0.618)

# 跳过 OCR 的标签（仅检测，不识别文本）
SKIP_OCR_LABELS = {'qrcode', 'line', 'seal', 'handled_by', 'verified_by', 'accountant'}


def _process_label_text(label: str, text: str) -> str:
    """根据标签类型做后处理。"""
    if label == 'total3':
        # total3 是金额中文大写
        return get_chinese_amount(text)
    if label in ('total', 'total2'):
        return get_amount(text)
    if label == 'idate':
        # 处理入库时间标签，提取日期部分
        # 例如："入库时间：2025-09-3014:15:22" -> "2025-09-30"
        text = text.strip()
        # 查找冒号位置
        colon_pos = text.find('：')  # 中文冒号
        if colon_pos == -1:
            colon_pos = text.find(':')  # 英文冒号
        if colon_pos != -1:
            # 提取冒号后的内容
            text = text[colon_pos + 1:].strip()
        
        # 去除常见前缀
        prefixes = ['入库时间', '入库日期', '日期', '时间']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip('：:').strip()
                break
        
        # 提取日期部分（格式如：2025-09-30 或 2025-09-3014:15:22）
        # 尝试匹配日期格式 YYYY-MM-DD
        date_match = re.search(r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})', text)
        if date_match:
            date_str = date_match.group(1)
            # 统一格式为 YYYY-MM-DD
            date_str = date_str.replace('年', '-').replace('月', '-').replace('日', '').replace('/', '-')
            # 处理月份和日期可能只有一位数的情况
            parts = date_str.split('-')
            if len(parts) == 3:
                year, month, day = parts
                date_str = f"{year}-{month.zfill(2)}-{day.zfill(2)}"
            text = date_str
        
        return get_date(text)
    if label in ('doc_number', 'cnt'):
        return get_num(text)
    if label == 'page':
        return get_page(text)
    if label == 'supplier':
        # 处理供应商标签，提取供应商名称
        # 例如："对于供应商：浙江致创广告装饰有限公司" -> "浙江致创广告装饰有限公司"
        text = text.strip()
        # 查找冒号位置
        colon_pos = text.find('：')  # 中文冒号
        if colon_pos == -1:
            colon_pos = text.find(':')  # 英文冒号
        if colon_pos != -1:
            # 提取冒号后的内容
            text = text[colon_pos + 1:].strip()
        # 去除常见前缀
        prefixes = ['对于供应商', '供应商', '供方', '供货单位', '供应单位']
        for prefix in prefixes:
            if text.startswith(prefix):
                # 去除前缀（可能带冒号）
                text = text[len(prefix):].lstrip('：:').strip()
                break
        return text
    return text.strip()


def stock_detection(img_numpy, stock=None, context=None, saveImage=False):
    """
    入库单检测与识别（材料/总务入库单）
    参考 vat_detect 的 ultralytics 推理流程，批量 OCR 提升吞吐。
    """
    if stock is None:
        stock = {}
    if context is None:
        return stock

    results = model.predict(
        source=img_numpy,
        imgsz=pub_img_size,
        device=device,
        verbose=False,
        half=False,
    )

    names = model.names
    im0 = img_numpy
    ocr = context.ocr
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
                    stock_fp = os.path.join('images', 'stock_v1')
                    os.makedirs(stock_fp, exist_ok=True)
                    path = os.path.join(stock_fp, f'{label}.png')
                    cv2.imwrite(path, im0[y1:y2, x1:x2])

                labels[label] = [y1, y2, x1, x2]
                label_confidences[label] = conf  # 保存置信度

            if labels:
                labels = {key: im0[v[0]:v[1], v[2]:v[3]] for key, v in labels.items()}

                # 批量 OCR，跳过无需 OCR 的标签
                ocr_label_keys = []
                ocr_images = []
                for label, img_region in labels.items():
                    if label in SKIP_OCR_LABELS:
                        continue
                    ocr_label_keys.append(label)
                    ocr_images.append(img_region)

                ocr_results_dict = {}
                if ocr_images:
                    batch_results = batch_ocr(ocr_images)
                    for label, text in zip(ocr_label_keys, batch_results):
                        ocr_results_dict[label] = text

                # 处理二维码
                if 'qrcode' in labels:
                    try:
                        qr_text = get_qrcode_data(Image.fromarray(labels['qrcode']))
                        if qr_text:
                            stock['qrcode'] = qr_text
                            stock['qrcode_conf'] = label_confidences.get('qrcode', 0.0)
                            detected = True
                    except Exception:
                        pass

                # 处理 OCR 结果
                for label, text in ocr_results_dict.items():
                    processed = _process_label_text(label, text)
                    stock[converter[label]] = processed
                    detected = True

                # SKIP_OCR_LABELS 中的标签：保存置信度
                # 只有当置信度 >= 阈值且不需要识别文本的标签，才设置成 'detected'
                # 但置信度字段始终设置
                for label in SKIP_OCR_LABELS:
                    if label in labels:
                        if label in ('qrcode', 'line'):
                            continue
                        conf = label_confidences.get(label, 0.0)
                        # 置信度始终设置
                        stock[f'{converter[label]}_conf'] = conf
                        # 只有置信度 >= 阈值时才设置 'detected'
                        if conf >= CONFIDENCE_THRESHOLD:
                            stock[converter[label]] = 'detected'
                            detected = True
 
                # 处理 line 标签：进行表格识别
                if 'line' in labels:
                    try:
                        # 对line区域进行表格识别和OCR
                        line_img = labels['line']
                        if label_confidences.get('line', 0.0) > CONFIDENCE_THRESHOLD:
                            rows = context.ocr_table_cells(line_img)
                        else:
                            rows = []
                        stock['line_conf'] = label_confidences.get('line', 0.0)
                        stock['line'] = rows
                        detected = True
                    except Exception as e:
                        logger.error(f"表格识别错误: {e}")
                        # 如果表格识别失败，且置信度 >= 阈值，回退到简单标记
                        # 但置信度始终设置
                        line_conf = label_confidences.get('line', 0.0)
                        stock['line_conf'] = line_conf
                        if line_conf >= CONFIDENCE_THRESHOLD:
                            stock[converter['line']] = 'detected'
                            detected = True

    if detected:
        # 补齐关键字段的默认值，方便上层消费
        for key in converter.values():
            if key in ('total', 'total2', 'total3'):
                stock.setdefault(key, '¥ 0.00')
            else:
                stock.setdefault(key, "")
        stock.setdefault('total_amount', stock.get('total') or stock.get('total2') or stock.get('total3') or '¥ 0.00')
        stock.setdefault('page', '-1/-1')
        stock['_stock_detected'] = True
        title = stock.get('title') or ''
        if '总务' in title or '结算' in title:
            stock['_stock_v1_detected'] = True
        else:
            stock['_stock_v1_detected'] = False
    return stock

