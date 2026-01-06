# -*- coding: utf-8 -*-
"""
使用 TextDetection 进行文本检测和裁剪
批量处理单元格图片，裁剪出文本区域
"""
import cv2
import os
import sys
import numpy as np
from loguru import logger
from paddleocr import TextDetection

# 添加项目根目录到路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config

# ========== 配置参数 ==========
# 输入目录：包含单元格图片的目录
input_dir = os.path.join(config.base_dir, 'images', 'table_cells')

# 输出目录：保存裁剪后的图片
output_dir = os.path.join(config.base_dir, 'images', 'table_cells_cropped')

# 是否使用 TextDetection 进行文本检测和裁剪
use_text_detection = True

# TextDetection 配置
use_gpu = getattr(config, 'GPU', False)

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


def process_single_image(image_path, output_path, use_text_detection=False, text_detector=None):
    """
    处理单张图片，使用文本检测裁剪文本区域并保存
    
    Args:
        image_path: 输入图片路径
        output_path: 输出图片路径
        use_text_detection: 是否使用文本检测进行裁剪
        text_detector: TextDetection 实例
        
    Returns:
        bool: 是否处理成功
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"无法读取图像: {image_path}")
            return False
        
        # 如果使用文本检测，检测文本区域并裁剪
        if use_text_detection and text_detector is not None:
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
                    img = crop_text_regions(img, dt_boxes, padding=5)
                    logger.debug(f"  已裁剪文本区域")
                else:
                    logger.debug(f"  未检测到文本区域，使用原图")
                    
            except Exception as e:
                logger.warning(f"  文本检测失败: {e}，使用原图")
                import traceback
                logger.debug(traceback.format_exc())
        
        # 保存处理后的图片
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        success = cv2.imwrite(output_path, img)
        
        if success:
            logger.info(f"✓ 处理成功: {os.path.basename(image_path)} -> {os.path.basename(output_path)}")
        else:
            logger.error(f"✗ 保存失败: {output_path}")
        
        return success
        
    except Exception as e:
        logger.error(f"处理图片失败 {image_path}: {e}", exc_info=True)
        return False


def process_directory(input_dir, output_dir, use_text_detection=False, text_detector=None):
    """
    批量处理目录中的所有图片
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        use_text_detection: 是否使用文本检测
        text_detector: TextDetection 实例
    """
    if not os.path.exists(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        return
    
    # 获取所有图像文件
    image_files = []
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append((file_path, file))
    
    if not image_files:
        logger.warning(f"目录中没有找到图像文件: {input_dir}")
        return
    
    logger.info(f"找到 {len(image_files)} 张图片")
    logger.info("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 统计信息
    success_count = 0
    fail_count = 0
    
    # 处理每张图片
    for idx, (image_path, filename) in enumerate(image_files, 1):
        logger.info(f"\n[{idx}/{len(image_files)}] 处理: {filename}")
        
        # 生成输出路径（保持原文件名）
        output_path = os.path.join(output_dir, filename)
        
        # 处理图片
        if process_single_image(image_path, output_path, use_text_detection, text_detector):
            success_count += 1
        else:
            fail_count += 1
    
    # 打印统计信息
    logger.info("\n" + "=" * 60)
    logger.info("处理完成！")
    logger.info("=" * 60)
    logger.info(f"成功: {success_count} 张")
    logger.info(f"失败: {fail_count} 张")
    logger.info(f"总计: {len(image_files)} 张")
    logger.info(f"\n输出目录: {output_dir}")
    logger.info("=" * 60)


def compare_images(original_path, cropped_path):
    """
    对比原图和处理后的图片（用于调试）
    
    Args:
        original_path: 原图路径
        cropped_path: 裁剪后的图片路径
    """
    try:
        original = cv2.imread(original_path)
        cropped = cv2.imread(cropped_path)
        
        if original is None or cropped is None:
            logger.error("无法读取对比图片")
            return
        
        # 创建对比图（并排显示）
        h1, w1 = original.shape[:2]
        h2, w2 = cropped.shape[:2]
        
        max_h = max(h1, h2)
        total_w = w1 + w2 + 20  # 20像素间距
        
        # 创建画布
        if len(original.shape) == 3:
            comparison = np.ones((max_h, total_w, 3), dtype=np.uint8) * 255
        else:
            comparison = np.ones((max_h, total_w), dtype=np.uint8) * 255
        
        # 放置原图
        comparison[:h1, :w1] = original
        
        # 放置处理后的图
        comparison[:h2, w1+20:w1+20+w2] = cropped
        
        return comparison
        
    except Exception as e:
        logger.error(f"对比图片失败: {e}")
        return None


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("单元格图片文本检测和裁剪测试")
    logger.info("=" * 60)
    logger.info(f"输入目录: {input_dir}")
    logger.info(f"输出目录: {output_dir}")
    logger.info(f"使用文本检测裁剪: {'是' if use_text_detection else '否'}")
    logger.info("=" * 60)
    
    # 批量处理
    process_directory(input_dir, output_dir, use_text_detection, text_detector)


if __name__ == "__main__":
    main()

