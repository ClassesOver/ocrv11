from ultralytics import YOLO
import numpy as np
import re
import cv2
import os
import config
from util.tool import *
from functools import lru_cache

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
             'amount_with_tax': 'amount_with_tax',
             'invoice_type': 'invoice_type'}

if config.ocrRange == "complex":
    converter.update({
        'buy_title': 'buy_title',
        'buy_tax': 'buy_tax',
        'buy_addr': 'buy_addr',
        'buy_bank': 'buy_bank',
        'sale_title': 'sale_title',
        'sale_tax': 'sale_tax',
        'sale_addr': 'sale_addr',
        'sale_bank': 'sale_bank',
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

pub_weights = f"models/vat/best.onnx"
pub_img_size = 640

# 初始化 YOLO 模型
# 根据 config.GPU 配置选择设备
if config.GPU:
    device = config.GPUID
else:
    device = 'cpu'

# 使用 ultralytics YOLO 类加载模型
model = YOLO(pub_weights, task='detect')


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
        if "普通" in title:
            invoice_type = "32"
        else:
            invoice_type = "31"
    else:
        if "专用" in title:
            if '电子' in title:
                invoice_type = '08'
            else:
                invoice_type = '01'
            invoice['invoice_type'] = "01"
        if "普通" in title:
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
                newimg_list = [y1 - 5, y2 + 5, x1 - 12, x2 + 12]

                # 处理重复的 check_code
                if label == 'check_code' and label in labels:
                    label = 'check_code2'

                # 特殊处理 buy_bank
                if label == 'buy_bank':
                    type = ocr(im0[y1 - 5:y2 + 5, x1 - 100:x1])
                    if '电话' in type:
                        labels["buy_addr"] = newimg_list
                        continue

                # 保存检测区域图像（如果配置了）
                if saveImage:
                    invoice_fp = os.path.join('images', 'invoice')
                    os.makedirs(invoice_fp, exist_ok=True)
                    path = os.path.join(invoice_fp, '%s.png' % label)
                    cv2.imwrite(path, im0[newimg_list[0]:newimg_list[1], newimg_list[2]:newimg_list[3]])

                labels[label] = newimg_list
            # 扩展发票号码区域
            if 'invoice_number' in labels and ('invoice_code' not in labels or 'invoice_number2' not in labels):
                labels['invoice_number'][3] += 48


            # 将标签坐标转换为实际图像区域
            labels = {key: im0[newimg_list[0]:newimg_list[1], newimg_list[2]:newimg_list[3]]
                      for key, newimg_list in labels.items()}

            # 使用batch_ocr批量识别所有labels
            # 收集所有需要OCR的图像区域
            ocr_label_keys = []
            ocr_images = []
            
            for label, img_region in labels.items():
                # 跳过不需要OCR的标签
                if label in ('qrcode',) or label.startswith('seal_'):
                    continue
                ocr_label_keys.append(label)
                ocr_images.append(img_region)
            
            # 批量OCR识别
            ocr_results_dict = {}
            if ocr_images:
                batch_results = batch_ocr(ocr_images)
                # 将结果映射回对应的标签
                for label, text in zip(ocr_label_keys, batch_results):
                    ocr_results_dict[label] = text

            has_qrcode = False
            # 处理二维码
            if 'qrcode' in labels:
                has_qrcode = qrcode_pyzbar(labels.get('qrcode'), invoice)
                if has_qrcode:
                    if invoice.get('invoice_type') == '32':
                        title = '电子发票（普通发票）'
                    elif invoice.get('invoice_type') == '31':
                        title = '电子发票（专用发票）'
                    else:
                        title = ocr_results_dict.get('title', '')
                    invoice['title'] = title

                    if invoice.get('invoice_type') in ['01', '04']:
                        invoice['amount_with_tax'] = get_amount(ocr_results_dict.get('amount_with_tax', ''))
                        invoice['tax'] = get_amount(ocr_results_dict.get('tax', ''))

                    if invoice.get('invoice_type') in ['31', '32']:
                        invoice['total_amount'] = get_amount(ocr_results_dict.get('total', ''))
                        invoice['tax'] = get_amount(ocr_results_dict.get('tax', ''))

                    if config.ocrRange == 'complex':
                        for label in labels.keys():
                            if label.startswith(('buy_', 'sale_')):
                                text = ocr_results_dict.get(label, '')
                                # 使用统一的处理函数（消除重复代码）
                                invoice[converter[label]] = process_buy_sale_field(label, text)
                            elif label.startswith('seal_'):
                                # 印章识别（通常无需OCR，只需检测）
                                invoice[converter[label]] = "detected"

            # 没有二维码时的处理
            if not has_qrcode:
                for label in labels.keys():
                    if label == 'qrcode':
                        continue
                    
                    # 从批量识别结果中获取文本
                    text = ocr_results_dict.get(label, '')

                    # 根据标签类型进行后处理
                    if label in ('check_code', 'check_code2'):
                        processed_text = text
                    elif label in ('invoice_number', 'invoice_number2'):
                        processed_text = get_num(text)
                    elif label in ('invoice_code2', 'invoice_code'):
                        processed_text = get_num(text)[-12:]
                    elif label == 'bill_date':
                        processed_text = get_date(text)
                    elif label in ('total', 'amount_with_tax', 'tax'):
                        processed_text = get_amount(text)
                    elif label.startswith(('buy_', 'sale_')):
                        # 使用统一的处理函数（消除重复代码）
                        processed_text = process_buy_sale_field(label, text)
                    elif label.startswith('seal_'):
                        # 印章识别（通常无需OCR，只需检测）
                        processed_text = "detected"
                    else:
                        processed_text = text.strip()

                    invoice[converter[label]] = processed_text

                # 判断发票类型
                title = invoice.get('title')
                if not title and 'title' in labels:
                    if invoice.get('check_code'):
                        invoice['title'] = title = '增值税普通发票'
                    else:
                        invoice['title'] = title = '增值税专用发票'
                judge_invoice_type(title, invoice)

            # 税额计算 - 使用预编译的正则表达式
            if "¥ 0.00" == invoice.get('tax'):
                total_amount = float(''.join(RE_AMOUNT.findall(invoice.get('total_amount', '0'))))
                amount_with_tax = float(''.join(RE_AMOUNT.findall(invoice.get('amount_with_tax', '0'))))
                invoice['tax'] = '¥ {}'.format(round(total_amount - amount_with_tax, 2))

            # 处理负数税额
            if '-' not in invoice.get('tax', '') and (
                    '-' in invoice.get('total_amount', '') or '-' in invoice.get('amount_with_tax', '')):
                invoice['tax'] = invoice['tax'].replace('¥ ', '¥ -')

            # 填充缺失字段
            for val in converter.values():
                if invoice.get(val) is None:
                    if val in ('total_amount', 'tax', 'amount_with_tax'):
                        invoice[val] = '¥ 0.00'
                    else:
                        invoice[val] = ""

            invoice['invoice_type_name'] = type_converter_name[invoice['invoice_type']]
            judge_invoice_repeat_data(invoice)

    return invoice

