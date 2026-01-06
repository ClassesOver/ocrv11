from ultralytics import YOLO
import cv2
import os
import re
from PIL import Image
from loguru import logger

import config
from util.tool import get_amount, get_date, get_num, get_page, get_qrcode_data_v2 as get_qrcode_data
from obj_det.model_loader import load_yolo_model

# 药品入库单类别映射（与模型标签保持一致）
converter = {
    'title': 'title',
    'qrcode': 'qrcode',
    'supplier': 'supplier',
    'total': 'total',
    'idate': 'idate',
    'doc_number': 'doc_number',
    'total2': 'total2',
    'verified_by': 'verified_by',
    'accountant': 'accountant',
    'page': 'page',
    'line': 'line',
    'director': 'director',
    'purchaser': 'purchaser',
    'note': 'note',
    'rk_way': 'rk_way',
}

# 模型目录与输入尺寸
model_dir = os.getenv("STOCK_V2_MODEL_DIR", "models/stock_2")
model_format = getattr(config, "STOCK_V2_MODEL_FORMAT", None)  # None 表示自动选择
pub_img_size = getattr(config, "STOCK_V2_IMGSZ", 640)

# 初始化 YOLOv11 模型（根据配置自动选择 best.pt、best.onnx 或 best_openvino_model）
device = config.GPUID if getattr(config, "GPU", False) else "cpu"
try:
    model = load_yolo_model(model_dir, model_name='best', model_format=model_format, task='detect')
except Exception as e:
    logger.error(f"加载 YOLOv11 模型失败: {e}")
    # 回退到直接指定路径的方式（兼容旧配置）
    pub_weights = os.getenv("STOCK_V2_WEIGHTS", "models/stock_2/best.pt")
    logger.warning(f"使用回退方式加载模型: {pub_weights}")
    model = YOLO(pub_weights, task='detect')

# 置信度阈值（可配置，默认 0.618）
CONFIDENCE_THRESHOLD = getattr(config, "STOCK_V2_CONFIDENCE_THRESHOLD", 0.618)

# 仅检测不做 OCR 的标签
SKIP_OCR_LABELS = {'qrcode', 'line', 'director', 'purchaser', 'verified_by', 'accountant'}


def _process_label_text(label: str, text: str) -> str:
    """根据标签类型做简单后处理。"""
    if label in ('total', 'total2'):
        return get_amount(text, precision=3)
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
    if label == 'doc_number':
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


def stock_detection_v2(img_numpy, stock=None, context=None, saveImage=False):
    """
    入库单检测与识别（西药/中药版本）。
    参考 vat_detect 的推理流程，使用 batch_ocr 提升吞吐。
    """
    logger.debug("开始入库单检测（stock_v2）")
    if stock is None:
        stock = {}
    if context is None:
        logger.warning("OCR context 未提供，返回空结果")
        return stock

    results = model.predict(
        source=img_numpy,
        imgsz=pub_img_size,
        device=device,
        verbose=False,
        half=False,
    )

    names = model.names
    # logger.debug(f"模型类别映射 (model.names): {names}")
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

            logger.debug(f"检测到 {len(boxes)} 个检测框")
            for box in boxes:
                xyxy = box.xyxy[0].cpu().numpy()
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())  # 获取置信度
                label = names[cls]
                if label not in converter_keys:
                    logger.debug(f"跳过未知标签: {label}")
                    continue

                x1, y1, x2, y2 = [int(v) for v in xyxy]
                logger.debug(f"检测到标签: {label}, 置信度: {conf:.3f}, 位置: ({x1}, {y1}, {x2}, {y2})")
                # if label == 'line':
                #     # line标签边界容错处理
                #     img_h, img_w = im0.shape[:2]
                #
                #     # x方向：扩展到图像边界，但添加容错边距
                #     margin_x = 5  # 左右边距容错（像素）
                #     x1 = max(0, x1 - margin_x)  # 左边界，确保不小于0
                #     x2 = min(img_w, x2 + margin_x)  # 右边界，确保不超过图像宽度
                #
                #     # 如果检测框宽度太小，扩展到全宽（容错处理）
                #     box_width = x2 - x1
                #     min_width_ratio = 0.3  # 最小宽度比例阈值
                #     if box_width < img_w * min_width_ratio:
                #         # 检测框太小，扩展到全宽
                #         x1 = 0
                #         x2 = img_w
                #
                #     # y方向：添加容错边距，但确保不超出图像范围
                #     margin_y = 5  # 上下边距容错（像素）
                #     y1 = max(0, y1 - margin_y)  # 上边界
                #     y2 = min(img_h, y2 + margin_y)  # 下边界
                #
                #     # 确保边界值有效
                #     x1 = max(0, min(x1, img_w - 1))
                #     x2 = max(x1 + 1, min(x2, img_w))
                #     y1 = max(0, min(y1, img_h - 1))
                #     y2 = max(y1 + 1, min(y2, img_h))

                if saveImage:
                    stock_fp = os.path.join('images', 'stock_v2')
                    os.makedirs(stock_fp, exist_ok=True)
                    path = os.path.join(stock_fp, f'{label}.png')
                    cv2.imwrite(path, im0[y1:y2, x1:x2])

                labels[label] = [y1, y2, x1, x2]
                label_confidences[label] = conf  # 保存置信度

            if labels:
                logger.info(f"检测到 {len(labels)} 个有效标签: {list(labels.keys())}")
                labels = {key: im0[v[0]:v[1], v[2]:v[3]] for key, v in labels.items()}

                # 批量 OCR，跳过无需 OCR 的标签
                ocr_label_keys = []
                ocr_images = []
                for label, img_region in labels.items():
                    if label in SKIP_OCR_LABELS:
                        logger.debug(f"跳过 OCR 标签: {label}")
                        continue
                    ocr_label_keys.append(label)
                    ocr_images.append(img_region)

                ocr_results_dict = {}
                if ocr_images:
                    logger.debug(f"开始批量 OCR，共 {len(ocr_images)} 个区域")
                    batch_results = batch_ocr(ocr_images)
                    for label, text in zip(ocr_label_keys, batch_results):
                        ocr_results_dict[label] = text
                        logger.debug(f"OCR 结果 [{label}]: {text[:50] if text else ''}")

                # 处理二维码
                if 'qrcode' in labels:
                    try:
                        logger.debug("开始处理二维码")
                        qr_text = get_qrcode_data(Image.fromarray(labels['qrcode']))
                        if qr_text:
                            stock['qrcode'] = qr_text
                            stock['qrcode_conf'] = label_confidences.get('qrcode', 0.0)
                            logger.info(f"二维码识别成功: {qr_text}")
                            detected = True
                    except Exception as e:
                        logger.warning(f"二维码识别失败: {e}")
                title = ocr_results_dict.get('title')
                if not title or '药' not in title:
                    logger.info(f"标题不匹配，跳过处理: {title}")
                    return stock
                # 处理 OCR 结果
                for label, text in ocr_results_dict.items():
                    processed = _process_label_text(label, text)
                    stock[converter[label]] = processed
                    logger.debug(f"处理标签 [{label}]: {text} -> {processed}")
                    detected = True

                # SKIP_OCR_LABELS 中的标签：保存置信度
                # 只有当置信度 >= 阈值且不需要识别文本的标签，才设置成 'detected'
                # 但置信度字段始终设置
                for label in SKIP_OCR_LABELS:
                    if label in labels:
                        if label == 'qrcode':
                            # qrcode 已在上面处理
                            continue
                        conf = label_confidences.get(label, 0.0)
                        # 置信度始终设置
                        stock[f'{converter[label]}_conf'] = conf
                        # 只有置信度 >= 阈值时才设置 'detected'
                        if conf >= CONFIDENCE_THRESHOLD:
                            stock[converter[label]] = 'detected'
                            logger.debug(f"标签 [{label}] 置信度 {conf:.3f} >= 阈值 {CONFIDENCE_THRESHOLD}，设置为 detected")
                            detected = True
                        else:
                            logger.debug(f"标签 [{label}] 置信度 {conf:.3f} < 阈值 {CONFIDENCE_THRESHOLD}，仅保存置信度")
                force_ocr_table = not stock.get('qrcode') or config.force_ocr_table
                # 处理 line 标签：进行表格识别
                if 'line' in labels and force_ocr_table:
                    try:
                        logger.info("开始处理 line 标签（表格识别）")
                        # 对line区域进行表格识别和OCR
                        line_img = labels['line']
                        line_conf = label_confidences.get('line', 0.0)
                        if line_conf > CONFIDENCE_THRESHOLD:
                            logger.debug(f"line 置信度 {line_conf:.3f} > 阈值 {CONFIDENCE_THRESHOLD}，开始表格识别和OCR")
                            rows = context.ocr_table_cells(line_img, selected_columns=[0, 7], saveImage=saveImage)
                            logger.info(f"表格识别成功，共 {len(rows)} 行")
                        else:
                            logger.debug(f"line 置信度 {line_conf:.3f} <= 阈值 {CONFIDENCE_THRESHOLD}，跳过表格识别")
                            rows = []
                        stock['line_conf'] = line_conf
                        stock['line'] = rows
                        detected = True
                    except Exception as e:
                        logger.error(f"表格识别错误: {e}", exc_info=True)
                        # 如果表格识别失败，且置信度 >= 阈值，回退到简单标记
                        # 但置信度始终设置
                        line_conf = label_confidences.get('line', 0.0)
                        stock['line_conf'] = line_conf
                        if line_conf >= CONFIDENCE_THRESHOLD:
                            stock['line'] = 'detected'
                            logger.warning(f"表格识别失败，回退到 detected 标记，置信度: {line_conf:.3f}")
                            detected = True
        else:
            logger.debug("未检测到任何检测框")
    else:
        logger.debug("模型预测结果为空")

    if detected:
        # 补齐关键字段，方便上层消费
        for key in converter.values():
            if key in ('total', 'total2'):
                stock.setdefault(key, '¥ 0.00')
            else:
                stock.setdefault(key, "")
        stock.setdefault('total_amount', stock.get('total') or stock.get('total2') or '¥ 0.00')
        stock.setdefault('page', '-1/-1')
        stock['_stock_detected'] = True
        title = stock.get('title') or ''
        if '药' in title:
            stock['_stock_v2_detected'] = True
            logger.info(f"识别为药品入库单，标题: {title}")
        else:
            stock['_stock_v2_detected'] = False
    else:
        logger.warning("未检测到有效的入库单信息")
    return stock
