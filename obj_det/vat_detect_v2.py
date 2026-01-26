from ultralytics import YOLO
import numpy as np
import re
import cv2
import os
import config
from PIL import Image
from datetime import datetime
from util.tool import *
from util.tool import _vat_qrcode,_vat_qrcode_v2
from functools import lru_cache
from obj_det.model_loader import load_yolo_model
from loguru import logger

# 预编译正则表达式以提高性能
RE_ADDR_PREFIX = re.compile(r'^\s*(地址|单位地址|购方地址|销方地址|地址、电话)[:：]?\s*')
RE_ADDR_SPLIT = re.compile(r'(电话|开户行|账号|银行|Bank)')
RE_ADDR_CLEAN = re.compile(r'[★☆※*•·●⊙◎¤■◆◇▪▎▏▍▌▋▊▉|｜~`^_=+<>《》〈〉【】\[\]{}（）()]')
RE_BANK_PREFIX = re.compile(r'^\s*(开户行及账号|开户行|账号|银行)[:：]?\s*')
RE_BANK_CLEAN = re.compile(r'[★☆※*•·●⊙◎¤■◆◇▪▎▏▍▌▋▊▉|｜~`^_=+<>《》〈〉【】\[\]{}（）()]')
RE_DIGITS = re.compile(r'\d')
RE_AMOUNT = re.compile(r'-?[0-9]\d*\.*')
# 预编译用于地址和银行信息处理的正则表达式
RE_COMMA_NORMALIZE = re.compile(r'[，,;；]+')
RE_SPACE_NORMALIZE = re.compile(r'\s+')

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
             'total2': 'total_amount2',
             'tax2': 'tax2',
             'qrcode': 'qrcode',
             'page': 'page',
             'amount_with_tax': 'amount_with_tax',
             'invoice_type': 'invoice_type'}

if config.ocrRange == "complex":
    converter.update({
        'buy_title': 'buy_title',
        # 'buy_tax': 'buy_tax',
        # 'buy_addr': 'buy_addr',
        # 'buy_bank': 'buy_bank',
        'sale_title': 'sale_title',
        # 'sale_tax': 'sale_tax',
        # 'sale_addr': 'sale_addr',
        # 'sale_bank': 'sale_bank',
        'seal_1': 'seal_1',
        'seal_2': 'seal_2',
    })

type_converter_name = {'01': '增值税专用发票', '04': '增值税普通发票',
                       '08': '增值税电子专用发票', '10': '增值税电子普通发票',
                       '31': '电子发票（增值税专用发票）', '32': '电子发票（增值税普通发票）',
                       '': '未识别的发票'}
type_converter = {'增值税专用发票': '01', '增值税普通发票': '04',
                  '增值税电子专用发票': '08', '增值税电子普通发票': '10',
                  '电子发票（增值税专用发票）': '31', '电子发票（增值税普通发票）': '32'}

# 模型目录和推理尺寸
model_dir = getattr(config, "VAT_MODEL_DIR", "models/vat_2")
model_format = getattr(config, "VAT_MODEL_FORMAT", None)  # None 表示自动选择
pub_img_size = getattr(config, "VAT_MODEL_IMGSZ", 640)

# 初始化 YOLOv11 模型（根据配置自动选择 best.pt、best.onnx 或 best_openvino_model）
# 根据 config.GPU 配置选择设备
if config.GPU:
    device = config.GPUID
else:
    device = 'cpu'

try:
    model = load_yolo_model(model_dir, model_name='best', model_format=model_format, task='detect')
except Exception as e:
    logger.error(f"加载 YOLOv11 模型失败: {e}")
    # 回退到直接指定路径的方式（兼容旧配置）
    pub_weights = f"models/vat/best.onnx"
    logger.warning(f"使用回退方式加载模型: {pub_weights}")
    model = YOLO(pub_weights, task='detect')

# 置信度阈值（可配置，默认 0.618）
CONFIDENCE_THRESHOLD = getattr(config, "VAT_V2_CONFIDENCE_THRESHOLD", 0.618)

# 跳过 OCR 的标签（仅检测，不识别文本）
SKIP_OCR_LABELS = {'qrcode', 'seal_1', 'seal_2'}


def get_check_code(code1, code2):
    if not code2:
        return get_num(code1)
    if code1 and '验码' in code1:
        return get_num(code1)
    if code2 and '验码' in code2:
        return get_num(code2)
    return max(get_num(code1), get_num(code2))


def judge_invoice_type(title, invoice):
    invoice_type = None
    if not title:
        return False
    if title.startswith('电子发票'):
        if "普" in title or '通' in title:
            invoice_type = "32"
        else:
            invoice_type = "31"
    else:
        if "专" in title or '用':
            if '电子' in title:
                invoice_type = '08'
            else:
                invoice_type = '01'
            invoice['invoice_type'] = "01"
        if "普" in title or '通' in title:
            if '电子' in title:
                invoice_type = '10'
            else:
                invoice_type = '04'
    if not invoice_type:
        if invoice.get('check_code'):
            invoice_type = "04"
        else:
            invoice_type = "01"
    invoice['invoice_type'] = invoice_type


def judge_invoice_repeat_data(invoice):
    # 对发票代码和发票号进行处理
    invoice_code = invoice.get('invoice_code', '')
    invoice_code2 = invoice.get('invoice_code2', '')
    if invoice_code != invoice_code2:
        if (len(invoice_code) != 12 and len(invoice_code2) == 12) or len(invoice_code) < len(invoice_code2):
            invoice['invoice_code'] = invoice_code2
    invoice_number = invoice.get('invoice_number', '')
    invoice_number2 = invoice.get('invoice_number2', '')
    if invoice_number != invoice_number2:
        in1 = len(invoice_number)
        in2 = len(invoice_number2)
        if in1 == 8:
            invoice_number_really = invoice_number
        elif in2 == 8:
            invoice_number_really = invoice_number2
        elif in2 > in1:
            invoice_number_really = invoice_number2
        else:
            invoice_number_really = invoice_number
        invoice['invoice_number'] = invoice_number_really
    invoice['check_code'] = get_check_code(invoice.get('check_code'), invoice.get('check_code2'))
    
    # 使用 in 替代 __contains__ 以提高性能
    if 'check_code2' in invoice:
        del invoice['check_code2']
    if 'invoice_code2' in invoice:
        del invoice['invoice_code2']
    if 'invoice_number2' in invoice:
        del invoice['invoice_number2']



def extract_addr(text: str) -> str:
    """
    提取地址信息（优化版，使用预编译正则）
    
    Args:
        text: 原始文本（已归一化）
        
    Returns:
        提取的地址
    """
    s = RE_ADDR_PREFIX.sub('', text)
    s = RE_ADDR_SPLIT.split(s, maxsplit=1)[0]
    s = RE_ADDR_CLEAN.sub('', s)
    s = RE_COMMA_NORMALIZE.sub('，', s)  # 统一分隔符
    s = RE_SPACE_NORMALIZE.sub(' ', s)   # 压缩多余空格
    return s.strip(' ，;；')


def extract_bank(text: str) -> str:
    """
    提取银行信息（优化版，使用预编译正则）
    
    Args:
        text: 原始文本（已归一化）
        
    Returns:
        提取的银行信息
    """
    s = RE_BANK_PREFIX.sub('', text)
    s = RE_BANK_CLEAN.sub('', s)
    s = RE_COMMA_NORMALIZE.sub('，', s)  # 统一分隔符
    s = RE_SPACE_NORMALIZE.sub(' ', s).strip(' ，;；')
    # 提取账号数字（允许空格/逗号分隔）
    account = ''.join(RE_DIGITS.findall(s))
    # 去掉账号部分得到银行名称
    name_part = RE_DIGITS.split(s, maxsplit=1)[0].strip(' ,;')
    if account and name_part:
        return f'{name_part} {account}'
    if account:
        return account
    return s.strip()


def process_buy_sale_field(label: str, text: str) -> str:
    """
    统一处理购买方/销售方字段（消除重复代码）
    
    Args:
        label: 字段标签
        text: OCR识别的原始文本
        
    Returns:
        处理后的文本
    """
    text = text.strip()
    
    # 税号处理
    if label in ('buy_tax', 'sale_tax'):
        return get_tax(text)
    
    # 名称处理
    if label in ('buy_title', 'sale_title'):
        return get_title(text)
    
    # 地址和银行需要归一化
    if label in ('buy_addr', 'sale_addr', 'buy_bank', 'sale_bank'):
        normalized = text.replace('：', ':').replace('，', ',').replace('；', ';')
        
        if label in ('buy_addr', 'sale_addr'):
            return extract_addr(normalized)
        else:  # buy_bank, sale_bank
            return extract_bank(normalized)
    
    # 其他字段直接返回
    return text


def ocr_buy_sale(ocr, label, img):
    """
    识别购买方/销售方信息

    Args:
        ocr: OCR 识别方法
        label: 字段标签
        img: 图像区域

    Returns:
        识别的文本
    """
    raw = ocr(img) or ""
    return process_buy_sale_field(label, raw)


def process_ocr_text(label: str, text: str) -> str:
    """
    根据标签类型处理OCR识别的文本
    
    Args:
        label: 字段标签
        text: OCR识别的原始文本
        
    Returns:
        处理后的文本
    """
    if label in ('check_code', 'check_code2'):
        return text
    elif label in ('invoice_number', 'invoice_number2'):
        return get_num(text)
    elif label in ('invoice_code2', 'invoice_code'):
        num_text = get_num(text)
        return num_text[-12:] if num_text else ''
    elif label == 'bill_date':
        return get_date(text)
    elif label in ('total', 'amount_with_tax', 'tax', 'total2', 'tax2'):
        return get_amount(text)
    elif label.startswith(('buy_', 'sale_')):
        return process_buy_sale_field(label, text)
    elif label == 'title':
        return text
    elif label == 'page':
        return get_page(text)
    else:
        return text


def update_invoice_from_ocr(ocr_results_dict: dict, invoice: dict):
    """
    根据OCR识别结果更新invoice
    
    Args:
        ocr_results_dict: OCR识别结果字典
        invoice: 发票信息字典
    """
    for label, text in ocr_results_dict.items():
        if label == 'qrcode':
            continue
        
        processed_text = process_ocr_text(label, text)
        
        # 更新 invoice
        if label in converter:
            invoice[converter[label]] = processed_text
    
    # 处理 title 和 invoice_type（根据 title 判断）
    if invoice.get('title') and not invoice.get('invoice_type'):
        judge_invoice_type(invoice.get('title'), invoice)


def process_qrcode(labels: dict, label_confidences: dict, ocr_results_dict: dict, invoice: dict) -> bool:
    """
    处理二维码，更新invoice
    
    Args:
        labels: 标签图像区域字典
        label_confidences: 标签置信度字典
        ocr_results_dict: OCR识别结果字典
        invoice: 发票信息字典
        
    Returns:
        bool: 二维码是否解析成功
    """
    if 'qrcode' not in labels:
        return False
    
    qrcode_parsed = False
    qr_text = None
    
    try:
        logger.debug("开始处理二维码")
        qr_text = get_qrcode_data_v2(Image.fromarray(labels['qrcode']))
        if qr_text:
            invoice['qrcode'] = qr_text
            invoice['qrcode_conf'] = label_confidences.get('qrcode', 0.0)
            logger.info(f"二维码识别成功: {qr_text}")
            
            # 解析二维码数据并更新 invoice（会覆盖 OCR 结果中的相关字段）
            try:
                _vat_qrcode_v2(qr_text, invoice)
                # 检查关键字段是否设置成功，判断二维码解析是否成功
                if invoice.get('invoice_type') and invoice.get('invoice_number'):
                    qrcode_parsed = True
                    logger.debug(f"二维码解析成功，发票类型: {invoice.get('invoice_type')}, 发票号码: {invoice.get('invoice_number')}")
                else:
                    logger.warning("二维码解析后关键字段缺失")
            except Exception as e:
                logger.warning(f"二维码解析失败: {e}", exc_info=True)
    except Exception as e:
        logger.warning(f"二维码识别失败: {e}")
    
    # 如果二维码识别成功，处理 title 和金额字段
    if qr_text:
        if qrcode_parsed:
            _supplement_amount_fields(invoice, ocr_results_dict)
    
    return qrcode_parsed


def _update_title_from_qrcode(invoice: dict, ocr_results_dict: dict):
    """
    根据二维码解析结果更新title
    
    Args:
        invoice: 发票信息字典
        ocr_results_dict: OCR识别结果字典
    """
    invoice_type = invoice.get('invoice_type')
    if invoice_type == '32':
        invoice['title'] = '电子发票（普通发票）'
    elif invoice_type == '31':
        invoice['title'] = '电子发票（专用发票）'
    elif not invoice.get('title'):
        # 如果没有 title，使用 OCR 结果
        invoice['title'] = ocr_results_dict.get('title', '')


def _supplement_amount_fields(invoice: dict, ocr_results_dict: dict):
    """
    二维码解析成功后，补充缺失的金额字段
    
    Args:
        invoice: 发票信息字典
        ocr_results_dict: OCR识别结果字典
    """
    invoice_type = invoice.get('invoice_type')
    if invoice_type in ['01', '04']:
        # 普通发票：二维码已设置 total_amount，尝试从 OCR 获取 amount_with_tax 和 tax
        if not invoice.get('amount_with_tax') or invoice.get('amount_with_tax') == '¥ 0.00':
            invoice['amount_with_tax'] = get_amount(ocr_results_dict.get('amount_with_tax', ''))
        if not invoice.get('tax') or invoice.get('tax') == '¥ 0.00':
            invoice['tax'] = get_amount(ocr_results_dict.get('tax', ''))
    elif invoice_type in ['31', '32']:
        # 电子发票：二维码已设置 amount_with_tax，尝试从 OCR 获取 total_amount 和 tax
        if not invoice.get('total_amount') or invoice.get('total_amount') == '¥ 0.00':
            invoice['total_amount'] = get_amount(ocr_results_dict.get('total', ''))
        if not invoice.get('tax') or invoice.get('tax') == '¥ 0.00':
            invoice['tax'] = get_amount(ocr_results_dict.get('tax', ''))


def process_seals(label_coords: dict, label_confidences: dict, invoice: dict):
    """
    处理印章（seal_*）字段
    
    Args:
        label_coords: 标签坐标字典
        label_confidences: 标签置信度字典
        invoice: 发票信息字典
    """
    for label in label_coords.keys():
        if label.startswith('seal_'):
            conf = label_confidences.get(label, 0.0)
            invoice[f'{converter[label]}_conf'] = conf
            # 只有置信度 >= 阈值时才设置 'detected'
            if conf >= CONFIDENCE_THRESHOLD:
                invoice[converter[label]] = "detected"
                logger.debug(f"标签 [{label}] 置信度 {conf:.3f} >= 阈值 {CONFIDENCE_THRESHOLD}，设置为 detected")
            else:
                logger.debug(f"标签 [{label}] 置信度 {conf:.3f} < 阈值 {CONFIDENCE_THRESHOLD}，仅保存置信度")


def post_process_invoice(invoice: dict):
    """
    后处理invoice：税额计算、负数处理、填充缺失字段等
    
    Args:
        invoice: 发票信息字典
    """
    # 税额计算 - 使用预编译的正则表达式
    # 只有当税额为 0.00 且两个金额字段都有有效值时才计算
    if invoice.get('tax') == '¥ 0.00':
        total_amount = float(''.join(RE_AMOUNT.findall(invoice.get('total_amount', '0'))))
        amount_with_tax = float(''.join(RE_AMOUNT.findall(invoice.get('amount_with_tax', '0'))))
        # 只有当两个金额都不为 0 时才计算税额（避免电子发票二维码解析后 total_amount 为 0 的情况）
        if total_amount != 0.0 and amount_with_tax != 0.0:
            invoice['tax'] = '¥ {}'.format(round(total_amount - amount_with_tax, 2))
    
    # 处理负数税额
    tax = invoice.get('tax', '')
    if '-' not in tax and ('-' in invoice.get('total_amount', '') or '-' in invoice.get('amount_with_tax', '')):
        invoice['tax'] = tax.replace('¥ ', '¥ -')
    
    # 填充缺失字段
    for val in converter.values():
        if invoice.get(val) is None:
            if val in ('total_amount', 'tax', 'amount_with_tax'):
                invoice[val] = '¥ 0.00'
            else:
                invoice[val] = ""
    invoice.setdefault('page', '1/1')
    # 设置发票类型名称和处理重复数据
    invoice_type = invoice.get('invoice_type', '')
    invoice['invoice_type_name'] = type_converter_name.get(invoice_type, '未识别的发票')
    judge_invoice_repeat_data(invoice)



def invoice_detection(img_numpy, invoice=None, context=None, saveImage=False):
    """
    使用 ultralytics YOLO 进行发票检测和识别
    
    Args:
        img_numpy: 输入图像 (numpy array)
        invoice: 发票信息字典
        context: OCR上下文对象
        saveImage: 是否保存检测区域图像，默认为 False
        
    Returns:
        invoice: 处理后的发票信息字典
        
    注意:
        YOLO/ONNX Runtime 推理是线程安全的，无需额外的锁保护
    """
    # 初始化 invoice
    if invoice is None:
        invoice = {}
    
    # 检查 context 是否为 None
    if context is None:
        logger.warning("OCR context 未提供，返回空结果")
        return invoice
    
    # 使用 ultralytics YOLO 进行推理，添加性能优化参数
    # YOLO 推理是线程安全的，可以在多线程环境下并发调用
    results = model.predict(
        source=img_numpy,
        imgsz=pub_img_size,
        device=device,
        verbose=False,      # 禁用详细输出以提升性能
        half=False          # 根据需要启用半精度（GPU时可设为True）
    )

    # 获取类别名称和检测结果
    names = model.names
    im0 = img_numpy
    ocr = context.ocr
    batch_ocr = context.batch_ocr

    # 处理检测结果 (ultralytics 返回的是 Results 对象列表)
    if results and len(results) > 0:
        result = results[0]  # 取第一个结果
        boxes = result.boxes  # Boxes 对象

        if boxes is not None and len(boxes) > 0:
            labels = {}
            label_confidences = {}  # 保存每个标签的置信度
            converter_keys = set(converter.keys())  # 转换为集合以提高查找速度

            # 遍历所有检测框
            for box in boxes:
                # 获取坐标、置信度和类别
                xyxy = box.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2]
                conf = box.conf[0].item()
                cls = int(box.cls[0].item())

                label = names[cls]
                if label not in converter_keys:
                    continue

                # 前两个控制竖向坐标，后两个控制横向
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                img_w = im0.shape[1]
                newimg_list = [
                    y1,                                     # y1 不处理
                    y2,                                     # y2 不处理
                    max(0, x1 - 3),                        # x1 向左扩展3像素，但不小于0
                    min(img_w, x2 + 3)                     # x2 向右扩展3像素，但不大于图像宽度
                ]

                # 处理重复的 check_code
                if label == 'check_code' and label in labels:
                    label = 'check_code2'

                # 特殊处理 buy_bank
                if label == 'buy_bank':
                    ocr_result = ocr(im0[y1 - 5:y2 + 5, x1 - 100:x1])
                    if '电话' in ocr_result:
                        labels["buy_addr"] = newimg_list
                        label_confidences["buy_addr"] = conf
                        continue

                # 保存检测区域图像（如果配置了）
                if saveImage:
                    invoice_fp = os.path.join('images', 'invoice_2')
                    os.makedirs(invoice_fp, exist_ok=True)
                    path = os.path.join(invoice_fp, '%s.png' % label)
                    cv2.imwrite(path, im0[newimg_list[0]:newimg_list[1], newimg_list[2]:newimg_list[3]])

                labels[label] = newimg_list
                label_confidences[label] = conf  # 保存置信度

            # 将标签坐标转换为实际图像区域
            label_coords = labels.copy()  # 保存坐标列表
            labels = {key: im0[coords[0]:coords[1], coords[2]:coords[3]]
                      for key, coords in label_coords.items()}

            # 使用batch_ocr批量识别所有labels
            # 批量 OCR，跳过无需 OCR 的标签
            ocr_label_keys = []
            ocr_images = []
            for label, img_region in labels.items():
                # 跳过不需要OCR的标签：集合中的标签或 seal_ 开头的标签
                if label in SKIP_OCR_LABELS or label.startswith('seal_'):
                    logger.debug(f"跳过 OCR 标签: {label}")
                    continue
                ocr_label_keys.append(label)
                ocr_images.append(img_region)
            
            # 批量OCR识别
            ocr_results_dict = {}
            if ocr_images:
                logger.debug(f"开始批量 OCR，共 {len(ocr_images)} 个区域")
                batch_results = batch_ocr(ocr_images)
                # 将结果映射回对应的标签
                for label, text in zip(ocr_label_keys, batch_results):
                    ocr_results_dict[label] = text
                    logger.debug(f"OCR 结果 [{label}]: {text[:50] if text else ''}")

            # ========== 第一步：先根据 OCR 识别结果更新 invoice ==========
            update_invoice_from_ocr(ocr_results_dict, invoice)

            # ========== 第二步：然后解析二维码更新 invoice ==========
            process_qrcode(labels, label_confidences, ocr_results_dict, invoice)

            # ========== 第三步：处理印章（seal_*） ==========
            process_seals(label_coords, label_confidences, invoice)

            # ========== 第四步：后处理（税额计算、负数处理、填充缺失字段等） ==========
            post_process_invoice(invoice)

    return invoice

