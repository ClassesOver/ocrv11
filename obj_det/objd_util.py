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

# YOLOv 分类模型相关配置
_classification_model = None
_classification_model_path = None

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


def _load_classification_model():
    """
    加载 YOLOv11 分类模型（懒加载）
    支持 YOLOv8/YOLOv11 分类模型
    
    Returns:
        YOLO 分类模型对象，如果模型不存在则返回 None
    """
    global _classification_model, _classification_model_path
    
    # 从 config 获取分类模型路径，如果没有配置则返回 None
    model_path = getattr(config, 'CLASSIFICATION_MODEL_PATH', "models/classification/best.onnx")
    if not model_path:
        return None
    
    # 如果模型路径改变或模型未加载，重新加载
    if _classification_model is None or _classification_model_path != model_path:
        try:
            from ultralytics import YOLO
            
            if not os.path.exists(model_path):
                logger.warning(f"分类模型文件不存在: {model_path}，跳过分类检测")
                return None
            
            # 根据设备配置选择设备
            if config.GPU:
                device = config.GPUID
            else:
                device = 'cpu'
            
            # 加载 YOLOv11 分类模型（task='classify' 自动识别为分类任务）
            _classification_model = YOLO(model_path, task='classify')
            _classification_model_path = model_path
            
            # 获取模型信息
            model_info = f"YOLOv11 分类模型"
            if hasattr(_classification_model, 'model'):
                if hasattr(_classification_model.model, 'yaml'):
                    yaml_info = _classification_model.model.yaml
                    if isinstance(yaml_info, dict):
                        version = yaml_info.get('version', 'unknown')
                        model_info = f"YOLOv{version} 分类模型"
            
            logger.info(f"{model_info}加载成功: {model_path}, device={device}")
        except Exception as e:
            logger.error(f"加载分类模型失败: {e}")
            _classification_model = None
            return None
    
    return _classification_model


def classify_image(img, confidence_threshold=0.618):
    """
    使用 YOLOv11 分类模型对图像进行分类检测（优化版）
    支持 YOLOv8/YOLOv11 分类模型
    
    Args:
        img: 输入图像 (numpy array)
        confidence_threshold: 置信度阈值，低于此值的结果将被忽略（默认 0.5）
        
    Returns:
        dict: 包含分类结果的字典
            - 'predicted_class': 预测的类别名称
            - 'confidence': 置信度
            - 'class_id': 类别ID
            如果分类失败、模型不存在或置信度低于阈值，返回 None
    """
    # 快速检查：图像有效性验证
    if img is None or not isinstance(img, numpy.ndarray) or img.size == 0:
        logger.debug("分类检测：输入图像无效")
        return None
    
    # 检查图像尺寸，避免过大图像影响性能
    h, w = img.shape[:2]
    max_size = getattr(config, 'CLASSIFICATION_MAX_IMG_SIZE', 1920)
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(f"分类检测：图像尺寸过大 ({h}x{w})，已缩放至 ({new_h}x{new_w})")
    
    try:
        model = _load_classification_model()
        if model is None:
            return None
        
        # 获取设备配置
        device = config.GPUID if config.GPU else 'cpu'
        
        # 获取图像尺寸配置（可从 config 覆盖）
        imgsz = getattr(config, 'CLASSIFICATION_IMG_SIZE', 640)
        
        # 执行分类预测（优化参数）
        results = model.predict(
            source=img,
            device=device,
            verbose=False,
            half=False,  # 分类任务通常不需要半精度
            augment=False  # 推理时不使用数据增强
        )
        
        if not results or len(results) == 0:
            logger.debug("分类检测：未获得预测结果")
            return None
        
        result = results[0]
        
        # 只处理分类模型结果（YOLOv8/YOLOv11 分类模型）
        if not hasattr(result, 'probs'):
            logger.warning("分类检测：模型不是分类模型，缺少 probs 属性（请使用 YOLOv8/YOLOv11 分类模型）")
            return None
        
        # 获取分类结果（YOLOv11 使用 probs 属性）
        try:
            class_id = result.probs.top1
            confidence = float(result.probs.top1conf)
        except AttributeError as e:
            logger.warning(f"分类检测：无法获取分类结果，probs 属性异常: {e}")
            return None
        
        if not hasattr(result, 'names') or class_id is None:
            logger.warning("分类检测：无法获取类别名称")
            return None
        
        try:
            class_name = result.names[class_id]
        except (KeyError, IndexError) as e:
            logger.warning(f"分类检测：类别ID {class_id} 无效: {e}")
            return None
        
        # 验证结果有效性
        if confidence < confidence_threshold:
            logger.debug(f"分类检测：置信度 {confidence:.2f} 低于阈值 {confidence_threshold:.2f}")
            return None
        
        logger.debug(f"分类检测结果: {class_name}, 置信度: {confidence:.2f}")
        
        return {
            'predicted_class': class_name,
            'confidence': confidence,
            'class_id': class_id
        }
        
    except Exception as e:
        logger.error(f"分类检测失败: {e}")
        logger.debug(traceback.format_exc())
        return None


def detection_img(img, saveImage=False):
    # img = rotate(img)
    invoice = {'invoice_type': ''}
    stock = {}
    result = {'type': '03', 'invoice': invoice, 'stock': stock}

    # 先使用 YOLOv 分类模型进行目标分类检测
    confidence_threshold = getattr(config, 'CLASSIFICATION_CONFIDENCE_THRESHOLD', 0.618)
    classification_result = classify_image(img, confidence_threshold=confidence_threshold)
    if classification_result:
        predicted_class = classification_result.get('predicted_class', '').lower()
        confidence = classification_result.get('confidence', 0.0)
        
        logger.debug(f"分类检测: {predicted_class}, 置信度: {confidence:.2f}")
        
        # 根据分类结果选择对应的检测流程
        # 类别名称映射：invoice, stock1, stock2, bill
        if predicted_class == 'stock1':
            # 入库单 v1
            try:
                result['stock'] = stock = {}
                context.stock_v1(img, stock, context, saveImage=saveImage)
                if is_stock_v1(stock):
                    result['type'] = '02'
                    result['invoice'] = {}
                    logger.info(result)
                    logger.debug(f"通过分类检测直接识别为入库单v1 (stock1)")
                    return result
            except Exception as e:
                logger.error(traceback.format_exc())
        elif predicted_class == 'stock2':
            # 入库单 v2
            try:
                result['stock'] = stock = {}
                context.stock_v2(img, stock, context, saveImage=saveImage)
                if is_stock_v2(stock):
                    result['type'] = '02'
                    result['invoice'] = {}
                    logger.info(result)
                    logger.debug(f"通过分类检测直接识别为入库单v2 (stock2)")
                    return result
            except Exception as e:
                logger.error(traceback.format_exc())
        elif predicted_class == 'bill':
            # 财务票据
            try:
                result['invoice'] = invoice = {'invoice_type': ''}
                context.bill(img, invoice, context, saveImage=saveImage)
                if is_bill(invoice) and invoice['invoice_type']:
                    result['stock'] = {}
                    result['type'] = '01'
                    logger.info(result)
                    logger.debug(f"通过分类检测直接识别为财务票据 (bill)")
                    return result
            except Exception as e:
                logger.error(traceback.format_exc())
        elif predicted_class == 'invoice1':
            # 增值税发票
            try:
                result['invoice'] = invoice = {'invoice_type': ''}
                context.vat(img, invoice, context, saveImage=saveImage)
                if invoice['invoice_type']:
                    result['stock'] = {}
                    result['type'] = '01'
                    logger.info(result)
                    logger.debug(f"通过分类检测直接识别为发票 (invoice1)")
                    return result
            except Exception as e:
                logger.error(traceback.format_exc())
        elif predicted_class == 'invoice2':
            # 增值税发票
            try:
                result['invoice'] = invoice = {'invoice_type': ''}
                context.vat_v2(img, invoice, context, saveImage=saveImage)
                if invoice['invoice_type']:
                    result['stock'] = {}
                    result['type'] = '01'
                    logger.info(result)
                    logger.debug(f"通过分类检测直接识别为发票 (invoice2)")
                    return result
            except Exception as e:
                logger.error(traceback.format_exc())
        elif 'invoice' in predicted_class:
            # 增值税发票
            try:
                result['invoice'] = invoice = {'invoice_type': ''}
                context.vat(img, invoice, context, saveImage=saveImage)
                if invoice['invoice_type']:
                    result['stock'] = {}
                    result['type'] = '01'
                    logger.info(result)
                    logger.debug(f"通过分类检测直接识别为发票 (invoice)")
                    return result
            except Exception as e:
                logger.error(traceback.format_exc())
        else:
            # 分类结果不明确或置信度较低，继续使用原有检测流程
            logger.debug(f"分类结果不明确 ({predicted_class})，使用原有检测流程")

    # 如果分类检测未成功或未启用，使用原有的检测流程
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
    logger.info(result)
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
