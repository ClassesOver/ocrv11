import cv2
import time
import fitz
import threading
import traceback
import numpy
from obj_det.ocr_context import context

import os
from loguru import logger
import datetime
import config

lock = threading.Lock()

allowed_extension = ['jpg', 'png', 'JPG', 'pdf', 'ofd']
image_extension = ['jpg', 'png', 'JPG']
pdf_extension = ['pdf']
ofd_extension = ['ofd']

vat_names = ['01', '04']
e_vat_names = ['08', '10', '14']
tra_names = ['88']
taxi_names = ['92']
roll_names = ['11']
no_tax_names = ['81']


# 检查文件扩展名
def allowed_file(filename, type_extension):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in type_extension


def time_synchronized():
    return time.time()


def rotate(fp, new_fp=False):
    """
    方向矫正
    """
    is_file = False
    if isinstance(fp, str):
        is_file = True
        img = cv2.imread(fp)
    elif isinstance(fp, numpy.ndarray):
        img = fp

    angle = context.angleModel(img)
    if angle != 0:
        index = 3 - angle / 90
        img = cv2.rotate(img, int(index))
        if is_file:
            if not new_fp:
                cv2.imwrite(fp, img)
            else:
                cv2.imwrite(new_fp, img)
                fp = new_fp
    if is_file:
        return fp
    else:
        return img


def process_image(fp):
    result = []
    rotate_fp = rotate(fp)
    list_invoice = context.det(rotate_fp)
    for invoice in list_invoice:
        invoice_rotate_fp = rotate(invoice['file_path'])
        invoice_type = invoice['invoice_type']
        logger.debug('the invoice type is %s.' % invoice_type)

        if str(invoice_type) in vat_names or str(invoice_type) in e_vat_names:
            context.title(file_name=invoice_rotate_fp, invoice=invoice, context=context)
            invoice_type = invoice['invoice_type']
        if str(invoice_type) in tra_names:
            context.tra(file_name=invoice['file_path'], invoice=invoice)
            result.append(invoice)
        if str(invoice_type) in vat_names:
            context.vat(file_name=invoice['file_path'], invoice=invoice, context=context)
            result.append(invoice)
        if str(invoice_type) in e_vat_names:
            context.evat(file_name=invoice['file_path'], invoice=invoice, context=context)
            result.append(invoice)
        if str(invoice_type) in taxi_names:
            context.taxi(file_name=invoice['file_path'], invoice=invoice)
            result.append(invoice)
        if str(invoice_type) in roll_names:
            context.roll(file_name=invoice['file_path'], invoice=invoice)
            result.append(invoice)
    return result


def process_pdf(fp):
    doc = fitz.open(fp)
    base_fname, _ = os.path.splitext(fp)
    result = []
    for pg in range(doc.pageCount):
        file_name = "%s_%s" % (base_fname, pg)
        page = doc[pg]
        rotate = int(0)
        zoom_x = 2.0
        zoom_y = 2.0
        trans = fitz.Matrix(zoom_x, zoom_y).preRotate(rotate)
        pm = page.getPixmap(matrix=trans, alpha=False)
        img_fp = '%s.png' % file_name
        pm.writePNG(img_fp)
        invoices = process_image(img_fp)
        result.extend(invoices)
    return result


def detection(fp):
    if allowed_file(fp, pdf_extension):
        return process_pdf(fp)
    else:
        return process_image(fp)


def detection_img_invoice_vat(fp):
    s = datetime.datetime.now()
    r = process_vat_invoice_image(fp)
    e = datetime.datetime.now()
    logger.debug('the  %s of detection takes in %s s.' % (e - s, fp))
    return r


def paddle_ocr(img):
    # 使用新版 PaddleOCR，不再使用已弃用的 get_text 方法
    return context.paddleOCR(cv2.resize(img, None, fx=0.9, fy=0.9))


def text_ocr(img):
    return context.chineseModel(img)


def is_stock_v1(stock):
    if stock.get('_stock_v1_detected'):
        return True


def is_stock_v2(stock):
    if stock.get('_stock_v2_detected'):
        return True


def is_bill(bill):
    return bill.get('_bill_detected', False)


def detection_img(img, saveImage=False):
    # img = rotate(img)
    invoice = {'invoice_type': ''}
    stock = {}
    result = {'type': '03', 'invoice': invoice, 'stock': stock}

    # 入库单优先检测
    try:
        result['stock'] = stock = {}
        context.stock_v1(img, stock, context, saveImage=saveImage)
        if is_stock_v1(stock):
            result['type'] = '02'
            result['invoice'] = {}
            return result
    except Exception as e:
        logger.error(traceback.format_exc())
    try:
        result['stock'] =  stock = {}
        context.stock_v2(img, stock, context, saveImage=saveImage)
        if is_stock_v2(stock):
            result['type'] = '02'
            result['invoice'] = {}
            return result
    except Exception as e:
        logger.error(traceback.format_exc())

    # 财务票据检测（无入库单时尝试）
    try:
        result['invoice'] = invoice = {'invoice_type': ''}
        context.bill(img, invoice, context, saveImage=saveImage)
        if is_bill(invoice) and invoice['invoice_type']:
            result['stock'] = {}
            result['type'] = '01'
            return result
    except Exception as e:
        logger.error(traceback.format_exc())


    # 增值税发票检测
    try:
        result['invoice'] = invoice = {'invoice_type': ''}
        context.vat(img, invoice, context, saveImage=saveImage)
    except Exception as e:
        logger.error(traceback.format_exc())
    if invoice['invoice_type']:
        result['stock'] = {}
        result['type'] = '01'
    else:
        if stock:
            result['invoice'] = {}
            result['type'] = '02'

    return result


def process_vat_invoice_image(fp):
    rotate_fp = rotate(fp)
    invoice = {}
    result = {'is_stock': False, 'invoice': invoice, 'stock': {}}
    s = datetime.datetime.now()
    context.title(file_name=rotate_fp, invoice=result, context=context)
    if result['is_stock']:
        return result
    e = datetime.datetime.now()
    if invoice:
        invoice_type = invoice['invoice_type']
        logger.debug('the title detection takes in  %s.' % (e - s,))
        logger.debug('the invoice type is %s.' % invoice_type)
        s = datetime.datetime.now()
        if str(invoice_type) in vat_names:
            context.vat(file_name=rotate_fp, invoice=invoice, context=context)
            e = datetime.datetime.now()
            logger.debug('the vat detection takes in  %s.' % (e - s,))
        if str(invoice_type) in e_vat_names:
            context.evat(file_name=rotate_fp, invoice=invoice, context=context)
            e = datetime.datetime.now()
            logger.debug('the e_vat detection takes in  %s.' % (e - s,))
    return result
