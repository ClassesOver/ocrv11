"""
YOLOv11 模型加载工具模块
独立模块，避免循环导入问题
"""
import os
from loguru import logger


def load_yolo_model(model_dir, model_name='best', model_format='pt', task='detect'):
    """
    加载 YOLOv11 模型，根据指定格式加载模型
    
    支持的模型格式：
    1. 'pt' (PyTorch) - 默认
    2. 'onnx' (ONNX Runtime)
    3. 'openvino' (OpenVINO)
    
    Args:
        model_dir: 模型目录路径（相对或绝对路径）
        model_name: 模型名称前缀，默认为 'best'
        model_format: 指定模型格式 ('pt', 'onnx', 'openvino')，默认为 'pt'
        task: 任务类型，默认为 'detect'（检测任务）
        
    Returns:
        YOLO 模型对象
        
    Raises:
        ValueError: 如果指定的格式不支持
        FileNotFoundError: 如果找不到指定的模型文件
    """
    from ultralytics import YOLO
    
    # 根据指定格式构建模型路径
    if model_format == 'pt':
        model_path = os.path.join(model_dir, f'{model_name}.pt')
    elif model_format == 'onnx':
        model_path = os.path.join(model_dir, f'{model_name}.onnx')
    elif model_format == 'openvino':
        model_path = os.path.join(model_dir, f'{model_name}_openvino_model')
    else:
        raise ValueError(f"不支持的模型格式: {model_format}，支持: 'pt', 'onnx', 'openvino'")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"指定的模型文件不存在: {model_path}")
    model_path = model_path = os.path.normpath(model_path)
    # 规范化路径（转换为绝对路径）
    model_path = os.path.abspath(model_path)
    logger.info(f"加载 YOLOv11 模型: {model_path} (格式: {model_format})")
    return YOLO(model_path, task=task)

