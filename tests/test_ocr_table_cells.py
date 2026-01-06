# -*- coding: utf-8 -*-
"""
测试表格单元格 OCR 识别功能
先使用 TextDetection 进行文本检测和裁剪，然后使用 TextRecognition 进行 OCR 识别
"""
import cv2
import os
import sys
import re
import numpy as np
from loguru import logger

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paddleocr import TextRecognition, TextDetection
import config

# ========== 配置参数 ==========
# table_cells 目录路径
table_cells_dir = os.path.join(config.base_dir, 'images', 'table_cells')

# PaddleOCR 配置
use_gpu = getattr(config, 'GPU', False)
model_name = getattr(config, 'PADDLE_REC_MODEL_NAME', 'PP-OCRv5_server_rec')

# 是否使用文本检测进行裁剪
use_text_detection = True

# 支持的图像格式
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# 初始化 TextDetection
text_detector = None
if use_text_detection:
    logger.info("正在初始化 TextDetection...")
    try:
        text_detector = TextDetection(device='gpu' if use_gpu else 'cpu')
        logger.info(f"TextDetection 初始化成功: device={'gpu' if use_gpu else 'cpu'}")
    except Exception as e:
        logger.error(f"TextDetection 初始化失败: {e}", exc_info=True)
        logger.warning("将回退到不使用文本检测的模式")
        use_text_detection = False
        text_detector = None

# 初始化 TextRecognition
logger.info("正在初始化 TextRecognition...")
try:
    paddle_ocr = TextRecognition(
        model_name=model_name,
        device='gpu' if use_gpu else 'cpu',
    )
    logger.info(f"TextRecognition 初始化成功: model={model_name}, device={'gpu' if use_gpu else 'cpu'}")
except Exception as e:
    logger.error(f"TextRecognition 初始化失败: {e}", exc_info=True)
    sys.exit(1)


def extract_position_from_filename(filename):
    """
    从文件名中提取行列位置
    例如: row_0_col_1.png -> (0, 1)
    
    Args:
        filename: 文件名
        
    Returns:
        (row_idx, col_idx) 元组，如果无法解析则返回 (None, None)
    """
    match = re.search(r'row_(\d+)_col_(\d+)', filename)
    if match:
        return (int(match.group(1)), int(match.group(2)))
    return (None, None)


def crop_text_regions(img, dt_boxes, padding=5):
    """
    根据检测到的文本边界框裁剪文本区域
    
    Args:
        img: 输入图像
        dt_boxes: 文本检测边界框列表，每个框是4个点的坐标 [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
        padding: 边界框扩展像素数，默认5
        
    Returns:
        裁剪后的图像（包含所有文本区域的最小外接矩形）
    """
    if not dt_boxes or len(dt_boxes) == 0:
        return img
    
    h, w = img.shape[:2]
    
    # 计算所有边界框的最小外接矩形
    all_x = []
    all_y = []
    
    for box in dt_boxes:
        # box 是 4 个点的坐标
        for point in box:
            x, y = int(point[0]), int(point[1])
            all_x.append(max(0, min(x, w)))
            all_y.append(max(0, min(y, h)))
    
    if not all_x or not all_y:
        return img
    
    # 计算边界框（添加 padding）
    x_min = max(0, min(all_x) - padding)
    y_min = max(0, min(all_y) - padding)
    x_max = min(w, max(all_x) + padding)
    y_max = min(h, max(all_y) + padding)
    
    # 裁剪图像
    cropped = img[y_min:y_max, x_min:x_max]
    
    return cropped


def detect_and_crop_text(img, text_detector):
    """
    使用 TextDetection 检测文本并裁剪文本区域
    
    Args:
        img: 输入图像
        text_detector: TextDetection 实例
        
    Returns:
        裁剪后的图像
    """
    if text_detector is None:
        return img
    
    try:
        # 使用 TextDetection 检测文本
        result = text_detector.predict(img)
        
        # 提取边界框
        dt_boxes = []
        if result is not None:
            # 处理不同的返回格式
            if isinstance(result, list):
                # 如果是列表，遍历每个元素
                for item in result:
                    if isinstance(item, dict):
                        # 字典格式，尝试不同的键
                        for key in ['dt_boxes', 'dt_polys', 'points', 'polys']:
                            if key in item:
                                boxes = item[key]
                                if isinstance(boxes, (list, np.ndarray)):
                                    if len(boxes) > 0:
                                        # 检查是否是嵌套列表
                                        if isinstance(boxes[0], (list, np.ndarray)):
                                            dt_boxes.extend(boxes)
                                        else:
                                            dt_boxes.append(boxes)
                                break
                    elif isinstance(item, (list, np.ndarray)):
                        # 直接是边界框
                        if len(item) > 0:
                            dt_boxes.append(item)
            elif isinstance(result, dict):
                # 字典格式
                for key in ['dt_boxes', 'dt_polys', 'points', 'polys']:
                    if key in result:
                        boxes = result[key]
                        if isinstance(boxes, (list, np.ndarray)):
                            if len(boxes) > 0:
                                if isinstance(boxes[0], (list, np.ndarray)):
                                    dt_boxes.extend(boxes)
                                else:
                                    dt_boxes = boxes
                        break
            elif isinstance(result, np.ndarray):
                # numpy 数组格式
                if len(result.shape) >= 2:
                    dt_boxes = result.tolist() if len(result.shape) == 3 else [result.tolist()]
        
        if dt_boxes and len(dt_boxes) > 0:
            logger.debug(f"  检测到 {len(dt_boxes)} 个文本区域")
            # 裁剪文本区域
            cropped_img = crop_text_regions(img, dt_boxes, padding=5)
            logger.debug(f"  已裁剪文本区域")
            return cropped_img
        else:
            logger.debug(f"  未检测到文本区域，使用原图")
            return img
            
    except Exception as e:
        logger.warning(f"  文本检测失败: {e}，使用原图")
        import traceback
        logger.debug(traceback.format_exc())
        return img


def ocr_single_cell(image_path, paddle_ocr_instance, use_text_detection=False, text_detector=None):
    """
    对单张单元格图片进行 OCR 识别
    先使用 TextDetection 检测和裁剪文本区域，然后使用 TextRecognition 进行识别
    
    Args:
        image_path: 图像路径
        paddle_ocr_instance: TextRecognition 实例
        use_text_detection: 是否使用文本检测进行裁剪
        text_detector: TextDetection 实例
        
    Returns:
        识别的文本字符串
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"无法读取图像: {image_path}")
            return None
        
        # 先使用文本检测裁剪文本区域（如果启用）
        if use_text_detection and text_detector is not None:
            img = detect_and_crop_text(img, text_detector)
        
        # 使用 TextRecognition 进行识别
        result = paddle_ocr_instance.predict(img)
        
        # 提取文本内容
        if result and len(result) > 0 and 'rec_text' in result[0]:
            return result[0]['rec_text']
        
        return ""
        
    except Exception as e:
        logger.error(f"OCR 识别失败 {image_path}: {e}", exc_info=True)
        return None


def process_table_cells_directory(cells_dir, paddle_ocr_instance, use_text_detection=False, text_detector=None):
    """
    遍历 table_cells 目录，对每张图片进行 OCR 识别
    
    Args:
        cells_dir: table_cells 目录路径
        paddle_ocr_instance: TextRecognition 实例
        use_text_detection: 是否使用文本检测进行裁剪
        text_detector: TextDetection 实例
    """
    if not os.path.exists(cells_dir):
        logger.error(f"目录不存在: {cells_dir}")
        logger.info(f"请先运行表格识别生成单元格图片，或检查路径是否正确")
        return
    
    # 获取所有图像文件
    image_files = []
    for file in os.listdir(cells_dir):
        file_path = os.path.join(cells_dir, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(file_path)
    
    if not image_files:
        logger.warning(f"目录中没有找到图像文件: {cells_dir}")
        return
    
    # 按文件名排序（按行列位置排序）
    image_files.sort(key=lambda x: extract_position_from_filename(os.path.basename(x)))
    
    logger.info(f"找到 {len(image_files)} 张单元格图片")
    logger.info("=" * 60)
    
    # 统计信息
    success_count = 0
    fail_count = 0
    empty_count = 0  # 识别结果为空的数量
    
    # 存储识别结果（按行列组织）
    results_dict = {}  # {(row, col): text}
    
    # 处理每张图像
    for idx, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        row_idx, col_idx = extract_position_from_filename(filename)
        
        logger.info(f"\n[{idx}/{len(image_files)}] 处理: {filename}")
        if row_idx is not None and col_idx is not None:
            logger.info(f"  位置: 第 {row_idx} 行, 第 {col_idx} 列")
        
        # 进行 OCR 识别
        result = ocr_single_cell(image_path, paddle_ocr_instance, use_text_detection=use_text_detection, text_detector=text_detector)
        
        if result is None:
            fail_count += 1
            logger.warning(f"  ✗ 识别失败")
        elif not result.strip():
            empty_count += 1
            logger.info(f"  ○ 识别结果为空")
            if row_idx is not None and col_idx is not None:
                results_dict[(row_idx, col_idx)] = ""
        else:
            success_count += 1
            logger.info(f"  ✓ 识别结果: {result}")
            if row_idx is not None and col_idx is not None:
                results_dict[(row_idx, col_idx)] = result
    
    # 打印统计信息
    logger.info("\n" + "=" * 60)
    logger.info("处理完成！")
    logger.info("=" * 60)
    logger.info(f"成功识别: {success_count} 张")
    logger.info(f"识别为空: {empty_count} 张")
    logger.info(f"识别失败: {fail_count} 张")
    logger.info(f"总计: {len(image_files)} 张")
    
    # 如果有结果，按行列组织显示
    if results_dict:
        logger.info("\n" + "=" * 60)
        logger.info("识别结果汇总（按行列组织）:")
        logger.info("=" * 60)
        
        # 找到最大行列号
        if results_dict:
            max_row = max(pos[0] for pos in results_dict.keys() if pos[0] is not None)
            max_col = max(pos[1] for pos in results_dict.keys() if pos[1] is not None)
            
            # 按行列显示
            for row_idx in range(max_row + 1):
                logger.info(f"\n第 {row_idx} 行:")
                for col_idx in range(max_col + 1):
                    text = results_dict.get((row_idx, col_idx), "")
                    if text:
                        logger.info(f"  列 {col_idx}: {text}")
                    else:
                        logger.info(f"  列 {col_idx}: (空)")
        
        logger.info("=" * 60)


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("表格单元格 OCR 识别测试")
    logger.info("先使用 TextDetection 进行文本检测和裁剪，然后使用 TextRecognition 进行 OCR")
    logger.info("=" * 60)
    logger.info(f"table_cells 目录: {table_cells_dir}")
    logger.info(f"模型: {model_name}")
    logger.info(f"设备: {'GPU' if use_gpu else 'CPU'}")
    logger.info(f"使用文本检测裁剪: {'是' if use_text_detection else '否'}")
    logger.info("=" * 60)
    
    # 处理 table_cells 目录
    process_table_cells_directory(table_cells_dir, paddle_ocr, use_text_detection=use_text_detection, text_detector=text_detector)


if __name__ == "__main__":
    main()

