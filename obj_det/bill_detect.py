from ultralytics import YOLO
from loguru import logger
import cv2
import os
import re
from PIL import Image
import config
from util.tool import get_amount, get_date, get_num, get_title, get_qrcode_data, normalize_invoice_type

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
        # 处理开票日期标签，提取日期部分
        # 例如："开票日期：20251010" -> "20251010"
        text = text.strip()
        # 查找冒号位置
        colon_pos = text.find('：')  # 中文冒号
        if colon_pos == -1:
            colon_pos = text.find(':')  # 英文冒号
        if colon_pos != -1:
            # 提取冒号后的内容
            text = text[colon_pos + 1:].strip()
        
        # 去除常见前缀
        prefixes = ['开票日期', '开票', '日期', '时间', '开票时间']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip('：:').strip()
                break
        
        # 提取日期部分（可能是 YYYYMMDD 格式或其他格式）
        # 先尝试提取连续8位数字（YYYYMMDD格式）
        date_match = re.search(r'(\d{8})', text)
        if date_match:
            date_str = date_match.group(1)
            # 格式化为 YYYY-MM-DD 格式，然后调用 get_date
            formatted_date = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
            return get_date(formatted_date)
        
        # 如果没有找到8位数字，尝试其他日期格式
        # 尝试匹配 YYYY-MM-DD 或 YYYY/MM/DD 格式
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
            return get_date(date_str)
        
        # 如果都没有匹配到，直接调用 get_date 处理
        return get_date(text)
    if label in ('total', 'amount_with_tax'):
        return get_amount(text)
    if label in ('buy_title', 'sale_title'):
        return get_title(text)
    if label == 'check_code':
        # 处理校验码标签，提取数字部分
        # 例如："校验码：679695" -> "679695"
        text = text.strip()
        # 查找冒号位置
        colon_pos = text.find('：')  # 中文冒号
        if colon_pos == -1:
            colon_pos = text.find(':')  # 英文冒号
        if colon_pos != -1:
            # 提取冒号后的内容
            text = text[colon_pos + 1:].strip()
        
        # 去除常见前缀
        prefixes = ['校验码', '校验', '验证码', '验证']
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip('：:').strip()
                break
        
        # 提取所有数字
        numbers = re.findall(r'\d+', text)
        if numbers:
            # 返回所有数字连接的结果（通常校验码是连续的数字）
            return ''.join(numbers)
        return text
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
    logger.debug("开始财务票据检测")
    if invoice is None:
        invoice = {}
    if context is None:
        logger.warning("OCR context 未提供，返回空结果")
        return invoice
    # 获取类别名称和检测结果
    names = model.names
    # logger.debug(f"模型类别映射 (model.names): {names}")
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
    invoice['invoice_type'] = ''
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

                if saveImage:
                    stock_fp = os.path.join('images', 'bill')
                    os.makedirs(stock_fp, exist_ok=True)
                    path = os.path.join(stock_fp, f'{label}.png')
                    cv2.imwrite(path, im0[y1:y2, x1:x2])

                labels[label] = [y1, y2, x1, x2]
                label_confidences[label] = conf  # 保存置信度

            if labels:
                logger.info(f"检测到 {len(labels)} 个有效标签: {list(labels.keys())}")
                # 裁剪区域
                label_images = {key: im0[v[0]:v[1], v[2]:v[3]] for key, v in labels.items()}

                # 批量 OCR，跳过无需 OCR 的标签
                ocr_label_keys, ocr_images = [], []
                for label, img_region in label_images.items():
                    if label in SKIP_OCR_LABELS:
                        logger.debug(f"跳过 OCR 标签: {label}")
                        continue
                    ocr_label_keys.append(label)
                    ocr_images.append(img_region)

                ocr_results = {}
                if ocr_images:
                    logger.debug(f"开始批量 OCR，共 {len(ocr_images)} 个区域")
                    for label, text in zip(ocr_label_keys, batch_ocr(ocr_images)):
                        ocr_results[label] = text
                        logger.debug(f"OCR 结果 [{label}]: {text[:50] if text else ''}")

                # 处理二维码
                if 'qrcode' in label_images:
                    try:
                        logger.debug("开始处理二维码")
                        qr_text = get_qrcode_data(Image.fromarray(label_images['qrcode']))
                        if qr_text:
                            # 提取发票类型（取第一个逗号前的部分）
                            invoice_type_raw = qr_text.split(',')[0].strip() if qr_text else ''
                            # 标准化发票类型：CZ-EI-33 -> CZEI033
                            invoice['invoice_type'] = normalize_invoice_type(invoice_type_raw)
                            invoice['qrcode'] = qr_text
                            invoice['qrcode_conf'] = label_confidences.get('qrcode', 0.0)
                            logger.info(f"二维码识别成功: {qr_text}, 发票类型: {invoice['invoice_type']}")
                            detected = True
                    except Exception as e:
                        logger.warning(f"二维码识别失败: {e}")

                title = ocr_results.get('title')
                if not title or'据' not in title:
                    logger.info(f"标题不匹配，跳过处理: {title}")
                    return invoice
                # 处理 OCR 结果
                for label, text in ocr_results.items():
                    invoice[converter[label]] = processed = _process_label_text(label, text)
                    logger.debug(f"处理标签 [{label}]: {text} -> {processed}")
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
                            logger.debug(
                                f"标签 [{label}] 置信度 {conf:.3f} >= 阈值 {CONFIDENCE_THRESHOLD}，设置为 detected")
                            detected = True
                        else:
                            logger.debug(f"标签 [{label}] 置信度 {conf:.3f} < 阈值 {CONFIDENCE_THRESHOLD}，仅保存置信度")
        else:
            logger.debug("未检测到任何检测框")
    else:
        logger.debug("模型预测结果为空")

    if detected:
        logger.info("财务票据检测成功")
        # 补齐缺省字段，金额默认 0
        for key in converter.values():
            if key in ('total_amount', 'amount_with_tax'):
                invoice.setdefault(key, '¥ 0.00')
            else:
                invoice.setdefault(key, "")
        title = invoice.get('title') or ''
        if '据' in title:
            if not invoice['invoice_type']:
                invoice['invoice_type'] = 'CZEI011'
            invoice['invoice_type_name'] = '电子财政票据'
            invoice['_bill_detected'] = True
            logger.info(f"识别为财务票据，标题: {title}")
        else:
            invoice['_bill_detected'] = False
            logger.info(f"识别为非财务票据，标题: {title}")
    else:
        logger.warning("未检测到有效的财务票据信息")
    return invoice
