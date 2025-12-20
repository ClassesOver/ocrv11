# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 23:25
# @Author  : FengZeTao zetao.feng@hand-china.com
# @File    : tool.py
# 如遇到看不懂且无备注的部分，请联系相关人
import os
import re
from random import random

import cv2
import numpy as np
from datetime import date, datetime
from PIL import Image, ImageEnhance
import pyzbar.pyzbar as pyzbar

# ============================================
# 预编译正则表达式以提高性能
# ============================================
RE_NUM = re.compile(r'-?[0-9]\d*')
RE_TAX = re.compile(r'-?[0-9]\d*[a-zA-Z]*')
RE_TITLE = re.compile(r'-?[^:：]*')
RE_ADDR_BANK = re.compile(r'[0-9\-]*$')
RE_FLOAT = re.compile(r'-?[0-9]\d*\.*')
RE_PAGE = re.compile(r'第(.*)页/共(.*)页')

# 金额提取相关的预编译正则
RE_AMOUNT_CURRENCY = re.compile(r'(?:¥|RMB|CNY)\s*([-+]?\d[\d,]*(?:\.\d+)?)', flags=re.IGNORECASE)
RE_AMOUNT_SUFFIX = re.compile(r'([-+]?\d[\d,]*(?:\.\d+)?)(?=\s*(?:¥|RMB|CNY))', flags=re.IGNORECASE)
RE_AMOUNT_GENERIC = re.compile(r'([-+]?\d[\d,]*(?:\.\d+)?)', flags=re.IGNORECASE)
RE_AMOUNT_CLEAN = re.compile(r'[★☆※*•·●⊙◎¤■◆◇▪▎▏▍▌▋▊▉|｜~`^_=+<>《》〈〉【】\[\]{}（）()]')
RE_AMOUNT_TRAILING_MINUS = re.compile(r'-\s*$')

# 日期提取相关的预编译正则
RE_DATE_CLEAN = re.compile(r'[★☆※*•·●⊙◎¤■◆◇▪▎▏▍▌▋▊▉|｜~`^_=+<>《》〈〉【】\[\]{}（）()]')

# 全角转半角映射表（预构建以提高性能）
FULLWIDTH_TO_HALFWIDTH = str.maketrans({
    '０': '0', '１': '1', '２': '2', '３': '3', '４': '4',
    '５': '5', '６': '6', '７': '7', '８': '8', '９': '9',
    '，': ',', '．': '.', '－': '-', '﹣': '-', '–': '-', '—': '-',
    '／': '/', '。': '.',
    '￥': '¥', '元': '¥', '圆': '¥', ' ': ''
})

# OCR误识别字符映射表
OCR_CORRECTION_MAP = str.maketrans({
    'O': '0', 'o': '0', 'D': '0',
    'S': '5', 'B': '8', 'l': '1', 'I': '1', 'i': '1',
    'Y': '¥'
})


def convert2yolov5(size, box):
    x_center = (box[0] + box[2]) / 2.0
    y_center = (box[1] + box[3]) / 2.0
    x = x_center / size[0]
    y = y_center / size[1]

    w = (box[2] - box[0]) / size[0]
    h = (box[3] - box[1]) / size[1]

    return round(x.item(), 6), round(y.item(), 6), round(w.item(), 6), round(h.item(), 6)


def get_num(string):
    """提取数字（优化版，使用预编译正则）"""
    string = string.replace('l', '1').replace('I', '1').replace('i', '1')
    return ''.join(RE_NUM.findall(string))


def get_tax(string):
    """提取税号（优化版，使用预编译正则）"""
    return ''.join(RE_TAX.findall(string))


def get_title(string):
    """提取标题（优化版，使用预编译正则）"""
    return ''.join(RE_TITLE.findall(string))


def get_addr_bank(string):
    """提取地址和银行信息（优化版，使用预编译正则）"""
    normalized = string.replace(':', '').replace('：', '')
    pre = RE_ADDR_BANK.split(normalized)[0]
    return f'{pre} {string[len(pre):]}' if pre else string


def get_float(string):
    """提取浮点数金额（优化版，修复语法错误）"""
    if not string:
        return '¥ 0.00'
    try:
        str_list = list(''.join(RE_FLOAT.findall(string)))
        if not str_list:
            return '¥ 0.00'
        
        # 修正常见OCR错误
        if str_list[0] == '-' and len(str_list) > 2 and str_list[1] == '0':
            str_list[1] = '8'
        elif str_list[0] == '0' and len(str_list) > 1:
            str_list[0] = '8'
        
        num_str = ''.join(str_list)
        value = float(num_str) if '.' in num_str else float(num_str)
        return f'¥ {value:.2f}'
    except Exception:
        return '¥ 0.00'

def get_amount(string):
    """
    提取金额并最大化容错（优化版，使用预编译正则和映射表）：
    - 统一全角/半角字符、货币符号与中文"元/圆"
    - 修正常见 OCR 误识别（O→0、S→5 等）
    - 支持货币符号在前/后、括号表示负数、末尾减号
    返回格式统一为 `¥ xx.xx`，找不到有效数字时返回 `¥ 0.00`。
    """
    if not string:
        return '¥ 0.00'
    try:
        raw = str(string).strip()
        
        # 使用预构建的映射表进行字符转换（性能优化）
        raw = raw.translate(FULLWIDTH_TO_HALFWIDTH)
        raw = raw.translate(OCR_CORRECTION_MAP)
        
        # 去掉明显无关的符号（使用预编译正则）
        raw = RE_AMOUNT_CLEAN.sub('', raw)
        raw = re.sub(r'\s+', '', raw)  # 金额中去除所有空白
        
        # 负数标记：括号或末尾减号
        is_bracket_negative = '(' in string and ')' in string  # 使用原始字符串检查
        has_trailing_minus = bool(RE_AMOUNT_TRAILING_MINUS.search(raw))
        
        # 使用预编译的正则表达式提取候选值
        def parse_candidates(pattern):
            vals = []
            for m in pattern.finditer(raw):
                num = m.group(1).replace(',', '')
                try:
                    vals.append(float(num))
                except ValueError:
                    continue
            return vals
        
        # 优先：带货币符号的匹配
        currency_vals = parse_candidates(RE_AMOUNT_CURRENCY)
        # 次优：数字后跟货币符号
        suffix_vals = parse_candidates(RE_AMOUNT_SUFFIX)
        # 兜底：任意数字串
        generic_vals = parse_candidates(RE_AMOUNT_GENERIC)
        
        candidates = currency_vals or suffix_vals or generic_vals
        if not candidates:
            return '¥ 0.00'
        
        # 选择最有可能的金额：优先最后出现的金额，其次绝对值最大
        value = candidates[-1]
        if len(candidates) > 1:
            max_abs_val = max(candidates, key=abs)
            if abs(max_abs_val) != abs(value):
                value = max_abs_val
        
        # 处理负数
        if (is_bracket_negative or has_trailing_minus) and value > 0:
            value = -value
        
        return f'¥ {value:.2f}'
    except Exception:
        return '¥ 0.00'


def get_chinese_amount(string):
    """
    提取并转换中文大写金额为阿拉伯数字格式
    例如：壹万贰仟叁佰肆拾伍元陆角柒分 -> ¥ 12345.67
    """
    if not string:
        return '¥ 0.00'
    
    try:
        # 中文数字映射
        cn_num = {
            '零': 0, '壹': 1, '贰': 2, '叁': 3, '肆': 4,
            '伍': 5, '陆': 6, '柒': 7, '捌': 8, '玖': 9,
            '〇': 0, '一': 1, '二': 2, '三': 3, '四': 4,
            '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
        }
        
        # 单位映射
        cn_unit = {
            '拾': 10, '十': 10,
            '佰': 100, '百': 100,
            '仟': 1000, '千': 1000,
            '万': 10000, '萬': 10000,
            '亿': 100000000, '億': 100000000
        }
        
        # 小数单位
        decimal_unit = {
            '角': 0.1, '毛': 0.1,
            '分': 0.01
        }
        
        raw = str(string).strip()
        
        # 按元分割整数和小数部分
        parts = re.split(r'[元圆]', raw)
        integer_part = parts[0] if len(parts) > 0 else ''
        decimal_part = parts[1] if len(parts) > 1 else ''
        
        # 解析整数部分
        total = 0
        temp_num = 0
        temp_unit = 1
        
        for char in integer_part:
            if char in cn_num:
                temp_num = cn_num[char]
            elif char in cn_unit:
                unit_value = cn_unit[char]
                if unit_value >= 10000:  # 万、亿等大单位
                    temp_num = (temp_num if temp_num > 0 else 1) * unit_value
                    total += temp_num
                    temp_num = 0
                else:  # 十、百、千
                    temp_num = (temp_num if temp_num > 0 else 1) * unit_value
                    total += temp_num
                    temp_num = 0
        
        # 加上剩余的数字
        total += temp_num
        
        # 解析小数部分
        decimal_value = 0
        temp_decimal = 0
        for char in decimal_part:
            if char in cn_num:
                temp_decimal = cn_num[char]
            elif char in decimal_unit:
                decimal_value += temp_decimal * decimal_unit[char]
                temp_decimal = 0
            elif char in ('整', '正'):
                break
        
        # 合并整数和小数
        result = total + decimal_value
        
        return f'¥ {result:.2f}'
        
    except Exception:
        # 如果解析失败，尝试用get_amount兜底
        return get_amount(string)


def get_page(string):
    """提取页码信息（优化版，使用预编译正则）"""
    try:
        string = string.replace('|', '1').replace('I', '1').replace('l', '1')
        match = RE_PAGE.search(string)
        if match:
            return f"{match.group(1) or 1}/{match.group(2) or 1}"
        # 回退方案：提取所有数字
        nums = RE_NUM.findall(string)
        if len(nums) >= 2:
            return f"{nums[0]}/{nums[1]}"
        return '-1/-1'
    except Exception:
        return '-1/-1'


def get_date(string):
    """提取日期（优化版，使用预编译正则和映射表）"""
    try:
        raw = str(string).strip()
        
        # 使用预构建的映射表（扩展版，包含日期相关字符）
        date_trans_map = FULLWIDTH_TO_HALFWIDTH.copy()
        date_trans_map.update({
            '年': '', '月': '', '日': '', '号': '',
            ' ': '', '\t': '', '\n': ''
        })
        
        raw = raw.translate(str.maketrans(date_trans_map))
        raw = raw.translate(OCR_CORRECTION_MAP)
        
        # 去掉无关符号（使用预编译正则）
        raw = RE_DATE_CLEAN.sub('', raw)
        raw = re.sub(r'\s+', '', raw)
        
        # 提取数字
        date_str = get_num(raw)
        date_len = len(date_str)
        
        # 补全年份
        if date_len < 8:
            curr_date = date.today().strftime('%Y%m%d')
            date_str = curr_date[:8 - date_len] + date_str
        
        return datetime.strptime(date_str, '%Y%m%d').strftime('%Y年%m月%d日')
    except Exception:
        return string


def remove_red_seal(image):
    """
    去除红色印章
    """
    # 获得红色通道
    blue_c, green_c, red_c = cv2.split(image)
    # 多传入一个参数cv2.THRESH_OTSU，并且把阈值thresh设为0，算法会找到最优阈值
    thresh, ret = cv2.threshold(red_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # 实测调整为95%效果好一些
    filter_condition = int(thresh * 0.95)
    _, red_thresh = cv2.threshold(red_c, filter_condition, 255, cv2.THRESH_BINARY)
    # 把图片转回 3 通道
    result_img = np.expand_dims(red_thresh, axis=2)
    result_img = np.concatenate((blue_c, green_c, result_img), axis=-1)
    return result_img


def _vat_qrcode(barcodeData, invoice):
    barcode_list = barcodeData.split(',')
    invoice['invoice_type'] = barcode_list[1].strip()
    invoice['invoice_code'] = barcode_list[2].strip()
    invoice['invoice_number'] = barcode_list[3].strip()
    if barcode_list[1] in ('31', '32'):
        invoice['amount_with_tax'] = get_float(barcode_list[4])
        invoice['total_amount'] = '¥ 0.00'
        invoice['tax'] = '¥ 0.00'
    else:
        invoice['total_amount'] = get_float(barcode_list[4])
        invoice['tax'] = '¥ 0.00'
        invoice['amount_with_tax'] = '¥ 0.00'
    invoice['billing_date'] = datetime.strptime(barcode_list[5], '%Y%m%d').strftime('%Y年%m月%d日')
    if invoice['invoice_type'] in ('04', '10'):
        invoice['check_code'] = get_num(barcode_list[6])


def _stock_qrcode(barcodeData, stock):
    barcode_list = [x.split(':')[1] for x in barcodeData.split(' ')]
    if len(barcode_list[0]) in (11, 12):
        code = get_num(barcode_list[0])
        invoice_number = barcode_list[2].replace('/', '、')
    else:
        code = get_num(barcode_list[2])
        invoice_number = barcode_list[0].replace('/', '、')
    stock['code'] = code
    stock['total_amount'] = get_float(barcode_list[1])
    stock['invoice_number'] = invoice_number
    stock['page'] = get_page(barcode_list[4])


def get_qrcode_data(img, index=0):
    if index > 3:
        return ''
    optimization = ["img = ImageEnhance.Brightness(img).enhance(2.0)",
                    "img = ImageEnhance.Sharpness(img).enhance(1.5)",
                    "img = ImageEnhance.Contrast(img).enhance(2.0)"]
    for x in optimization[0:index]:
        dic = {"img": img, "ImageEnhance": ImageEnhance}
        exec(x, dic)
        img = dic.get('img')
    barcodes = pyzbar.decode(img.convert('L'))
    if barcodes:
        return barcodes[0].data.decode("utf-8")
    else:
        return get_qrcode_data(img, index + 1)


def qrcode_pyzbar(image, val, is_stock=False):
    try:
        img = Image.fromarray(image)
        #
        # #锐利化
        # #增加对比度
        img = img.convert('L')  # 灰度化
        barcodeData = get_qrcode_data(img)
        if barcodeData:
            print(barcodeData)
            if is_stock:
                _stock_qrcode(barcodeData, val)
            else:
                _vat_qrcode(barcodeData, val)
            val['qrcode'] = True
            return True
        else:
            return False
    except Exception as e:
        return False


if __name__ == '__main__':
    invoice = {}
    path = '/Users/fengzetao/PycharmProjects/ocr/code/images/invoice/qrcode.png'
    image = cv2.imread(path)
    qrcode_pyzbar(image, invoice, False)
    print(invoice)
