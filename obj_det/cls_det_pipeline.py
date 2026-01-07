# -*- coding: utf-8 -*-
"""
分类-检测流水线模块
整合图片分类和目标检测功能：先分类，再根据分类结果选择对应的检测模型进行预测
"""
import os
import sys
import cv2
import numpy as np
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from loguru import logger
import traceback

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import config
from obj_det.model_loader import load_yolo_model

# 支持的图像格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# 分类模型相关（懒加载）
_classification_model = None
_classification_model_path = None

# 检测模型缓存 {class_name: model}
_detection_models = {}

# 分类结果到检测模型路径的映射
CLASS_TO_MODEL_MAP = {
    'stock1': {
        'dir': 'models/stock_1',
        'format': getattr(config, 'STOCK_V1_MODEL_FORMAT', 'onnx'),
        'imgsz': getattr(config, 'STOCK_V1_IMGSZ', 640),
        'conf': getattr(config, 'STOCK_V1_CONFIDENCE_THRESHOLD', 0.618)
    },
    'stock2': {
        'dir': 'models/stock_2',
        'format': getattr(config, 'STOCK_V2_MODEL_FORMAT', 'onnx'),
        'imgsz': getattr(config, 'STOCK_V2_IMGSZ', 640),
        'conf': getattr(config, 'STOCK_V2_CONFIDENCE_THRESHOLD', 0.618)
    },
    'bill': {
        'dir': 'models/bill',
        'format': getattr(config, 'BILL_MODEL_FORMAT', 'onnx'),
        'imgsz': getattr(config, 'BILL_MODEL_IMGSZ', 640),
        'conf': getattr(config, 'BILL_CONFIDENCE_THRESHOLD', 0.618)
    },
    'invoice1': {
        'dir': 'models/vat',
        'format': getattr(config, 'VAT_MODEL_FORMAT', 'onnx'),
        'imgsz': getattr(config, 'VAT_IMGSZ', 640),
        'conf': getattr(config, 'VAT_CONFIDENCE_THRESHOLD', 0.618)
    },
    'invoice2': {
        'dir': 'models/vat_2',
        'format': getattr(config, 'VAT_2_MODEL_FORMAT', 'onnx'),
        'imgsz': getattr(config, 'VAT_2_IMGSZ', 640),
        'conf': getattr(config, 'VAT_2_CONFIDENCE_THRESHOLD', 0.618)
    }
}


def _load_classification_model() -> Optional[object]:
    """
    加载分类模型（懒加载）
    
    Returns:
        YOLO 分类模型对象，如果模型不存在则返回 None
    """
    global _classification_model, _classification_model_path
    
    model_path = getattr(config, 'CLASSIFICATION_MODEL_PATH', None)
    if not model_path:
        model_path = os.path.join(project_root, "models", "classification", "best.onnx")
    
    if not model_path:
        return None
    
    if _classification_model is None or _classification_model_path != model_path:
        try:
            from ultralytics import YOLO
            
            if not os.path.exists(model_path):
                logger.warning(f"分类模型文件不存在: {model_path}")
                return None
            
            device = config.GPUID if config.GPU else 'cpu'
            _classification_model = YOLO(model_path, task='classify')
            _classification_model_path = model_path
            
            logger.info(f"分类模型加载成功: {model_path}, device={device}")
        except Exception as e:
            logger.error(f"加载分类模型失败: {e}")
            _classification_model = None
            return None
    
    return _classification_model


def _load_detection_model(class_name: str) -> Tuple[Optional[object], Optional[Dict]]:
    """
    根据分类结果加载对应的检测模型（懒加载）
    
    Args:
        class_name: 分类结果类别名称（如 'stock1', 'stock2', 'bill' 等）
        
    Returns:
        (模型对象, 模型配置字典)，如果模型不存在则返回 (None, None)
    """
    global _detection_models
    
    class_name_lower = class_name.lower()
    
    # 如果模型已加载，直接返回
    if class_name_lower in _detection_models:
        model_info = _detection_models[class_name_lower]
        return model_info['model'], model_info['config']
    
    # 检查类别是否在映射表中
    if class_name_lower not in CLASS_TO_MODEL_MAP:
        logger.warning(f"未知的类别: {class_name_lower}，无法加载检测模型")
        return None, None
    
    model_config = CLASS_TO_MODEL_MAP[class_name_lower]
    model_dir = os.path.join(project_root, model_config['dir'])
    model_format = model_config.get('format', 'onnx')
    
    try:
        # 使用 model_loader 加载模型
        # 如果model_format为None，默认使用onnx
        actual_format = model_format if model_format else 'onnx'
        model = load_yolo_model(
            model_dir=model_dir,
            model_name='best',
            model_format=actual_format,
            task='detect'
        )
        
        # 缓存模型
        _detection_models[class_name_lower] = {
            'model': model,
            'config': model_config
        }
        
        logger.info(f"检测模型加载成功: {class_name_lower} -> {model_dir}")
        return model, model_config
        
    except Exception as e:
        logger.error(f"加载检测模型失败 ({class_name_lower}): {e}")
        logger.debug(traceback.format_exc())
        return None, None


def classify_image(img: np.ndarray, confidence_threshold: float = 0.618) -> Optional[Dict]:
    """
    对图像进行分类
    
    Args:
        img: 输入图像 (numpy array)
        confidence_threshold: 置信度阈值
        
    Returns:
        dict: 包含分类结果的字典
            - 'predicted_class': 预测的类别名称
            - 'confidence': 置信度
            - 'class_id': 类别ID
            如果分类失败则返回 None
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        logger.debug("分类检测：输入图像无效")
        return None
    
    # 检查图像尺寸
    h, w = img.shape[:2]
    max_size = getattr(config, 'CLASSIFICATION_MAX_IMG_SIZE', 1920)
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_h, new_w = int(h * scale), int(w * scale)
        img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        logger.debug(f"分类检测：图像尺寸过大，已缩放至 ({new_h}x{new_w})")
    
    try:
        model = _load_classification_model()
        if model is None:
            return None
        
        device = config.GPUID if config.GPU else 'cpu'
        
        results = model.predict(
            source=img,
            device=device,
            verbose=False,
            half=False,
            augment=False
        )
        
        if not results or len(results) == 0:
            logger.debug("分类检测：未获得预测结果")
            return None
        
        result = results[0]
        
        if not hasattr(result, 'probs'):
            logger.warning("分类检测：模型不是分类模型，缺少 probs 属性")
            return None
        
        try:
            class_id = result.probs.top1
            confidence = float(result.probs.top1conf)
        except AttributeError as e:
            logger.warning(f"分类检测：无法获取分类结果: {e}")
            return None
        
        if not hasattr(result, 'names') or class_id is None:
            logger.warning("分类检测：无法获取类别名称")
            return None
        
        try:
            class_name = result.names[class_id]
        except (KeyError, IndexError) as e:
            logger.warning(f"分类检测：类别ID {class_id} 无效: {e}")
            return None
        
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


def xyxy2xywh(x1, y1, x2, y2, img_width, img_height):
    """
    将 xyxy 格式转换为归一化的 xywh 格式
    
    Args:
        x1, y1, x2, y2: 边界框坐标（像素）
        img_width, img_height: 图像宽度和高度
    
    Returns:
        (x_center, y_center, width, height) 归一化坐标（0-1之间）
    """
    x_center = (x1 + x2) / 2.0 / img_width
    y_center = (y1 + y2) / 2.0 / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height
    
    # 确保坐标在 [0, 1] 范围内
    x_center = max(0.0, min(1.0, x_center))
    y_center = max(0.0, min(1.0, y_center))
    width = max(0.0, min(1.0, width))
    height = max(0.0, min(1.0, height))
    
    return x_center, y_center, width, height


def detect_image(
    img: np.ndarray,
    class_name: str,
    confidence_threshold: Optional[float] = None,
    imgsz: Optional[int] = None
) -> Optional[List[Dict]]:
    """
    根据分类结果对图像进行目标检测
    
    Args:
        img: 输入图像 (numpy array)
        class_name: 分类结果类别名称
        confidence_threshold: 置信度阈值（如果为None则使用模型配置的默认值）
        imgsz: 输入图像尺寸（如果为None则使用模型配置的默认值）
        
    Returns:
        list: 检测结果列表，每个元素包含：
            - 'class_id': 类别ID
            - 'class_name': 类别名称
            - 'confidence': 置信度
            - 'bbox': [x1, y1, x2, y2] 边界框坐标（像素）
            - 'bbox_normalized': [x_center, y_center, width, height] 归一化坐标
        如果检测失败，返回 None
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        logger.debug("检测：输入图像无效")
        return None
    
    try:
        # 加载对应的检测模型
        model, model_config = _load_detection_model(class_name)
        if model is None:
            logger.warning(f"无法加载类别 {class_name} 的检测模型")
            return None
        
        # 使用模型配置的默认值
        if confidence_threshold is None:
            confidence_threshold = model_config.get('conf', 0.25)
        if imgsz is None:
            imgsz = model_config.get('imgsz', 640)
        
        device = config.GPUID if config.GPU else 'cpu'
        
        # 执行检测预测
        results = model.predict(
            source=img,
            imgsz=imgsz,
            device=device,
            verbose=False,
            conf=confidence_threshold,
            half=False,
            augment=False
        )
        
        if not results or len(results) == 0:
            logger.debug("检测：未获得预测结果")
            return None
        
        result = results[0]
        
        if not hasattr(result, 'boxes'):
            logger.warning("检测：模型不是检测模型，缺少 boxes 属性")
            return None
        
        boxes = result.boxes
        names = result.names
        
        if boxes is None or len(boxes) == 0:
            logger.debug("检测：未检测到任何目标")
            return []
        
        # 获取图像尺寸
        img_height, img_width = img.shape[:2]
        
        # 解析检测结果
        detections = []
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            cls = int(box.cls[0].item())
            conf = float(box.conf[0].item())
            
            x1, y1, x2, y2 = [int(v) for v in xyxy]
            
            # 转换为归一化坐标
            x_center, y_center, width, height = xyxy2xywh(
                x1, y1, x2, y2, img_width, img_height
            )
            
            class_name_det = names.get(cls, f"class_{cls}")
            
            detections.append({
                'class_id': cls,
                'class_name': class_name_det,
                'confidence': conf,
                'bbox': [x1, y1, x2, y2],
                'bbox_normalized': [x_center, y_center, width, height]
            })
        
        logger.debug(f"检测到 {len(detections)} 个目标")
        return detections
        
    except Exception as e:
        logger.error(f"检测失败: {e}")
        logger.debug(traceback.format_exc())
        return None


def process_image_pipeline(
    img: np.ndarray,
    classification_conf: float = 0.618,
    detection_conf: Optional[float] = None
) -> Dict:
    """
    完整的分类-检测流水线处理单张图片
    
    Args:
        img: 输入图像 (numpy array)
        classification_conf: 分类置信度阈值
        detection_conf: 检测置信度阈值（如果为None则使用模型配置的默认值）
        
    Returns:
        dict: 处理结果
            - 'success': bool, 是否成功
            - 'classification': dict, 分类结果（如果成功）
            - 'detection': list, 检测结果列表（如果成功）
            - 'error': str, 错误信息（如果失败）
    """
    result = {
        'success': False,
        'classification': None,
        'detection': None,
        'error': None
    }
    
    try:
        # 步骤1：分类
        classification_result = classify_image(img, confidence_threshold=classification_conf)
        
        if classification_result is None:
            result['error'] = '分类失败或置信度不足'
            return result
        
        result['classification'] = classification_result
        predicted_class = classification_result['predicted_class']
        
        # 步骤2：根据分类结果进行检测
        detection_results = detect_image(
            img,
            predicted_class,
            confidence_threshold=detection_conf
        )
        
        if detection_results is None:
            result['error'] = f'检测失败（类别: {predicted_class}）'
            return result
        
        result['detection'] = detection_results
        result['success'] = True
        
        return result
        
    except Exception as e:
        logger.error(f"流水线处理失败: {e}")
        logger.debug(traceback.format_exc())
        result['error'] = str(e)
        return result


def save_yolo_label(label_path: str, detections: List[Dict], save_conf: bool = True) -> None:
    """
    保存 YOLO 格式的标签文件
    
    Args:
        label_path: 标签文件路径
        detections: 检测结果列表
        save_conf: 是否保存置信度
    """
    try:
        sorted_detections = sorted(detections, key=lambda x: x['class_id'])
        
        os.makedirs(os.path.dirname(label_path), exist_ok=True)
        
        with open(label_path, 'w', encoding='utf-8') as f:
            for det in sorted_detections:
                class_id = det['class_id']
                x_center, y_center, width, height = det['bbox_normalized']
                conf = det['confidence']
                
                if save_conf:
                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n"
                else:
                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                
                f.write(line)
    except Exception as e:
        logger.error(f"保存标签文件失败 {label_path}: {e}")
        raise

