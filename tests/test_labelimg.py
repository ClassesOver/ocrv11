# -*- coding: utf-8 -*-
"""
YOLOv11 目标检测预测脚本
遍历指定目录，使用 YOLOv11 检测模型对图片进行预测，并保存标签文件（YOLO 格式）
"""
import cv2
import os
import sys
import argparse
import traceback
import time
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy as np

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    # 简单的进度条替代
    def tqdm(iterable, **kwargs):
        return iterable

from loguru import logger

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

import config
from obj_det.model_loader import load_yolo_model

# 支持的图像格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# YOLOv11 检测模型相关配置
_detection_model = None
_detection_model_path = None


def _load_detection_model(model_path: str) -> Optional[object]:
    """
    加载 YOLOv11 检测模型（懒加载）
    
    Args:
        model_path: 模型文件路径（支持 .pt, .onnx 等格式）
    
    Returns:
        YOLO 检测模型对象，如果模型不存在则返回 None
    """
    global _detection_model, _detection_model_path
    
    # 如果模型路径改变或模型未加载，重新加载
    if _detection_model is None or _detection_model_path != model_path:
        try:
            from ultralytics import YOLO
            
            if not os.path.exists(model_path):
                logger.error(f"检测模型文件不存在: {model_path}")
                return None
            
            # 根据设备配置选择设备
            device = config.GPUID if config.GPU else 'cpu'
            
            # 加载 YOLOv11 检测模型
            _detection_model = YOLO(model_path, task='detect')
            _detection_model_path = model_path
            
            logger.info(f"YOLOv11 检测模型加载成功: {model_path}, device={device}")
        except Exception as e:
            logger.error(f"加载检测模型失败: {e}")
            logger.debug(traceback.format_exc())
            _detection_model = None
            return None
    
    return _detection_model


def xyxy2xywh(x1, y1, x2, y2, img_width, img_height):
    """
    将 xyxy 格式转换为归一化的 xywh 格式
    
    Args:
        x1, y1, x2, y2: 边界框坐标（像素）
        img_width, img_height: 图像宽度和高度
    
    Returns:
        (x_center, y_center, width, height) 归一化坐标（0-1之间）
    """
    # 计算中心点和宽高
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
    model_path: str,
    confidence_threshold: float = 0.25,
    imgsz: int = 640
) -> Optional[List[Dict]]:
    """
    使用 YOLOv11 检测模型对图像进行目标检测
    
    Args:
        img: 输入图像 (numpy array)
        model_path: 模型文件路径
        confidence_threshold: 置信度阈值，低于此值的结果将被忽略（默认 0.25）
        imgsz: 输入图像尺寸（默认 640）
        
    Returns:
        list: 检测结果列表，每个元素包含：
            - 'class_id': 类别ID
            - 'class_name': 类别名称
            - 'confidence': 置信度
            - 'bbox': [x1, y1, x2, y2] 边界框坐标（像素）
            - 'bbox_normalized': [x_center, y_center, width, height] 归一化坐标
        如果检测失败或模型不存在，返回 None
    """
    # 快速检查：图像有效性验证
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        logger.debug("检测：输入图像无效")
        return None
    
    try:
        model = _load_detection_model(model_path)
        if model is None:
            return None
        
        # 获取设备配置
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
        
        # 检查是否是检测模型结果
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
            
            class_name = names.get(cls, f"class_{cls}")
            
            detections.append({
                'class_id': cls,
                'class_name': class_name,
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


def collect_image_files(directory: str, recursive: bool = False) -> List[str]:
    """
    收集目录中的所有图像文件
    
    Args:
        directory: 目录路径
        recursive: 是否递归搜索子目录
        
    Returns:
        图像文件路径列表
    """
    image_files = []
    directory = os.path.abspath(directory)
    
    if recursive:
        # 递归搜索
        for root, dirs, files in os.walk(directory):
            for file in files:
                ext = os.path.splitext(file)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    image_files.append(os.path.join(root, file))
    else:
        # 只搜索当前目录
        for file in os.listdir(directory):
            file_path = os.path.join(directory, file)
            if os.path.isfile(file_path):
                ext = os.path.splitext(file)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    image_files.append(file_path)
    
    return sorted(image_files)


def save_yolo_label(
    label_path: str,
    detections: List[Dict],
    save_conf: bool = True
) -> None:
    """
    保存 YOLO 格式的标签文件
    
    Args:
        label_path: 标签文件路径
        detections: 检测结果列表
        save_conf: 是否保存置信度（默认 True）
    """
    try:
        # 按照 class_id 排序
        sorted_detections = sorted(detections, key=lambda x: x['class_id'])
        
        with open(label_path, 'w', encoding='utf-8') as f:
            for det in sorted_detections:
                class_id = det['class_id']
                x_center, y_center, width, height = det['bbox_normalized']
                conf = det['confidence']
                
                if save_conf:
                    # 格式：class_id x_center y_center width height confidence
                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f} {conf:.6f}\n"
                else:
                    # 格式：class_id x_center y_center width height
                    line = f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}\n"
                
                f.write(line)
    except Exception as e:
        logger.error(f"保存标签文件失败 {label_path}: {e}")


def save_annotated_image(
    img: np.ndarray,
    detections: List[Dict],
    output_path: str,
    names: Dict[int, str]
) -> None:
    """
    保存带标注框的图像
    
    Args:
        img: 原始图像
        detections: 检测结果列表
        output_path: 输出图像路径
        names: 类别名称映射
    """
    try:
        annotated_img = img.copy()
        
        # 按照 class_id 排序
        sorted_detections = sorted(detections, key=lambda x: x['class_id'])
        
        # 绘制检测框和标签
        for det in sorted_detections:
            x1, y1, x2, y2 = det['bbox']
            class_name = det['class_name']
            conf = det['confidence']
            
            # 绘制边界框
            cv2.rectangle(annotated_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签文本
            label = f"{class_name} {conf:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            
            # 标签背景
            cv2.rectangle(
                annotated_img,
                (x1, y1 - text_height - baseline - 5),
                (x1 + text_width, y1),
                (0, 255, 0),
                -1
            )
            
            # 标签文字
            cv2.putText(
                annotated_img,
                label,
                (x1, y1 - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1
            )
        
        cv2.imwrite(output_path, annotated_img)
    except Exception as e:
        logger.error(f"保存标注图像失败 {output_path}: {e}")


def process_single_image(
    image_path: str,
    model_path: str,
    output_dir: str,
    confidence_threshold: float = 0.25,
    imgsz: int = 640,
    save_labels: bool = True,
    save_annotated: bool = False,
    save_images: bool = True,
    save_conf: bool = True,
    verbose: bool = True
) -> Optional[Dict]:
    """
    处理单张图片，进行检测并保存标签
    
    Args:
        image_path: 输入图片路径
        model_path: 模型文件路径
        output_dir: 输出目录
        confidence_threshold: 置信度阈值
        imgsz: 输入图像尺寸
        save_labels: 是否保存标签文件
        save_annotated: 是否保存带标注的图像
        save_images: 是否保存原始图片
        save_conf: 是否在标签文件中保存置信度
        verbose: 是否输出详细信息
        
    Returns:
        dict: 检测结果统计，包含 'count', 'classes' 等
              如果检测失败则返回 None
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"无法读取图像: {image_path}")
            return None
        
        # 进行检测
        detections = detect_image(
            img, model_path, confidence_threshold, imgsz
        )
        
        if detections is None:
            if verbose:
                logger.warning(f"检测失败: {os.path.basename(image_path)}")
            return None
        
        if len(detections) == 0:
            if verbose:
                logger.debug(f"未检测到目标: {os.path.basename(image_path)}")
            return {'count': 0, 'classes': {}}
        
        # 获取类别统计
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # 保存原始图片
        if save_images:
            image_name = os.path.basename(image_path)
            image_output_path = os.path.join(output_dir, 'images', image_name)
            os.makedirs(os.path.dirname(image_output_path), exist_ok=True)
            cv2.imwrite(image_output_path, img)
        
        # 保存标签文件
        if save_labels:
            image_name = os.path.splitext(os.path.basename(image_path))[0]
            label_path = os.path.join(output_dir, 'labels', f"{image_name}.txt")
            os.makedirs(os.path.dirname(label_path), exist_ok=True)
            save_yolo_label(label_path, detections, save_conf=save_conf)
        
        # 保存带标注的图像
        if save_annotated:
            image_name = os.path.basename(image_path)
            annotated_path = os.path.join(output_dir, 'annotated', image_name)
            os.makedirs(os.path.dirname(annotated_path), exist_ok=True)
            
            # 获取类别名称映射
            model = _load_detection_model(model_path)
            names = model.names if model else {}
            
            save_annotated_image(img, detections, annotated_path, names)
        
        if verbose:
            logger.info(
                f"✓ {os.path.basename(image_path)}: "
                f"检测到 {len(detections)} 个目标 "
                f"({', '.join([f'{k}:{v}' for k, v in class_counts.items()])})"
            )
        
        return {
            'count': len(detections),
            'classes': class_counts
        }
        
    except Exception as e:
        logger.error(f"处理图片失败 {image_path}: {e}", exc_info=True)
        return None


def process_directory(
    input_dir: str,
    model_path: str,
    output_dir: str,
    confidence_threshold: float = 0.25,
    imgsz: int = 640,
    recursive: bool = False,
    save_labels: bool = True,
    save_annotated: bool = False,
    save_images: bool = True,
    save_conf: bool = True,
    save_stats: bool = True,
    verbose: bool = True
) -> Tuple[int, int, Dict]:
    """
    批量处理目录中的所有图片
    
    Args:
        input_dir: 输入目录
        model_path: 模型文件路径
        output_dir: 输出目录
        confidence_threshold: 置信度阈值
        imgsz: 输入图像尺寸
        recursive: 是否递归搜索子目录
        save_labels: 是否保存标签文件
        save_annotated: 是否保存带标注的图像
        save_images: 是否保存原始图片
        save_conf: 是否在标签文件中保存置信度
        save_stats: 是否保存统计信息
        verbose: 是否输出详细信息
        
    Returns:
        (success_count, fail_count, class_statistics)
    """
    if not os.path.exists(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
        return 0, 0, {}
    
    if not os.path.exists(model_path):
        logger.error(f"模型文件不存在: {model_path}")
        return 0, 0, {}
    
    # 收集所有图像文件
    logger.info(f"正在扫描目录: {input_dir} (递归: {recursive})")
    image_files = collect_image_files(input_dir, recursive=recursive)
    
    if not image_files:
        logger.warning(f"目录中没有找到图像文件: {input_dir}")
        return 0, 0, {}
    
    logger.info(f"找到 {len(image_files)} 张图片")
    logger.info("=" * 60)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    if save_images:
        os.makedirs(os.path.join(output_dir, 'images'), exist_ok=True)
    if save_labels:
        os.makedirs(os.path.join(output_dir, 'labels'), exist_ok=True)
    if save_annotated:
        os.makedirs(os.path.join(output_dir, 'annotated'), exist_ok=True)
    
    # 统计信息
    success_count = 0
    fail_count = 0
    class_statistics = {}  # {class_name: count}
    total_detections = 0
    
    # 开始计时
    start_time = time.time()
    
    # 处理每张图片（使用进度条）
    progress_bar = tqdm(
        enumerate(image_files, 1),
        total=len(image_files),
        desc="检测处理",
        unit="张",
        disable=not verbose
    )
    
    for idx, image_path in progress_bar:
        filename = os.path.basename(image_path)
        
        # 更新进度条描述
        if HAS_TQDM:
            progress_bar.set_description(f"处理: {filename[:30]}")
        
        # 进行检测
        result = process_single_image(
            image_path,
            model_path,
            output_dir,
            confidence_threshold,
            imgsz,
            save_labels,
            save_annotated,
            save_images,
            save_conf,
            verbose=verbose
        )
        
        if result:
            success_count += 1
            total_detections += result.get('count', 0)
            for class_name, count in result.get('classes', {}).items():
                class_statistics[class_name] = class_statistics.get(class_name, 0) + count
        else:
            fail_count += 1
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    
    # 打印统计信息
    logger.info("\n" + "=" * 60)
    logger.info("处理完成！")
    logger.info("=" * 60)
    logger.info(f"成功处理: {success_count} 张")
    logger.info(f"处理失败: {fail_count} 张")
    logger.info(f"总计: {len(image_files)} 张")
    logger.info(f"检测到目标总数: {total_detections} 个")
    logger.info(f"耗时: {elapsed_time:.2f} 秒 (平均 {elapsed_time/len(image_files):.3f} 秒/张)")
    
    # 打印类别统计
    if class_statistics:
        logger.info("\n类别统计:")
        logger.info("-" * 60)
        for class_name, count in sorted(class_statistics.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_detections * 100) if total_detections > 0 else 0
            logger.info(f"  {class_name}: {count} 个 ({percentage:.1f}%)")
        logger.info("-" * 60)
    
    logger.info(f"\n输出目录: {output_dir}")
    if save_images:
        logger.info(f"原始图片目录: {os.path.join(output_dir, 'images')}")
    if save_labels:
        logger.info(f"标签文件目录: {os.path.join(output_dir, 'labels')}")
    if save_annotated:
        logger.info(f"标注图像目录: {os.path.join(output_dir, 'annotated')}")
    
    # 保存统计信息
    if save_stats:
        stats_file = os.path.join(output_dir, 'detection_statistics.json')
        stats_data = {
            'summary': {
                'total_images': len(image_files),
                'success': success_count,
                'failed': fail_count,
                'total_detections': total_detections,
                'elapsed_time_seconds': round(elapsed_time, 2),
                'average_time_per_image': round(elapsed_time / len(image_files), 3) if len(image_files) > 0 else 0
            },
            'class_distribution': class_statistics
        }
        
        try:
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats_data, f, ensure_ascii=False, indent=2)
            logger.info(f"统计信息已保存到: {stats_file}")
        except Exception as e:
            logger.warning(f"保存统计信息失败: {e}")
    
    logger.info("=" * 60)
    
    return success_count, fail_count, class_statistics


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='YOLOv11 目标检测预测脚本 - 对目录中的图片进行检测并保存标签',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法（使用默认模型路径）
  python tests/test_labelimg.py -i images/test -o output
  
  # 指定模型路径
  python tests/test_labelimg.py -i images/test -o output -m models/stock_1/best.pt
  
  # 指定置信度阈值
  python tests/test_labelimg.py -i images/test -o output -m models/stock_1/best.pt --conf 0.5
  
  # 保存带标注的图像
  python tests/test_labelimg.py -i images/test -o output -m models/stock_1/best.pt --annotated
  
  # 递归搜索子目录
  python tests/test_labelimg.py -i images/test -o output -m models/stock_1/best.pt --recursive
  
  # 不在标签文件中保存置信度
  python tests/test_labelimg.py -i images/test -o output -m models/stock_1/best.pt --no-conf
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='输入目录：包含待检测图片的目录'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='输出目录：保存标签文件和结果的目录'
    )
    
    parser.add_argument(
        '-m', '--model',
        type=str,
        default='yolov11n.pt',
        help='模型文件路径（支持 .pt, .onnx 等格式，默认: yolov11n.pt）'
    )
    
    parser.add_argument(
        '--conf',
        type=float,
        default=0.25,
        help='置信度阈值，低于此值的结果将被忽略（默认: 0.25）'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='输入图像尺寸（默认: 640）'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='递归搜索子目录中的图片'
    )
    
    parser.add_argument(
        '--annotated',
        action='store_true',
        help='保存带标注框的图像（在 output/annotated 目录）'
    )
    
    parser.add_argument(
        '--no-images',
        action='store_true',
        help='不保存原始图片（默认保存到 output/images 目录）'
    )
    
    parser.add_argument(
        '--no-labels',
        action='store_true',
        help='不保存标签文件（默认保存）'
    )
    
    parser.add_argument(
        '--no-conf',
        action='store_true',
        help='不在标签文件中保存置信度（默认保存）'
    )
    
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='不保存统计信息文件'
    )
    
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='静默模式：减少输出信息'
    )
    
    return parser.parse_args()


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志级别
    if args.quiet:
        logger.remove()
        logger.add(sys.stderr, level="WARNING")
    
    logger.info("=" * 60)
    logger.info("YOLOv11 目标检测预测")
    logger.info("=" * 60)
    logger.info(f"输入目录: {args.input}")
    logger.info(f"输出目录: {args.output}")
    logger.info(f"模型路径: {args.model}")
    logger.info(f"置信度阈值: {args.conf}")
    logger.info(f"图像尺寸: {args.imgsz}")
    logger.info(f"递归搜索: {'是' if args.recursive else '否'}")
    logger.info(f"保存原始图片: {'否' if args.no_images else '是'}")
    logger.info(f"保存标签: {'否' if args.no_labels else '是'}")
    logger.info(f"保存标注图像: {'是' if args.annotated else '否'}")
    logger.info(f"标签包含置信度: {'否' if args.no_conf else '是'}")
    logger.info("=" * 60)
    
    # 批量处理
    process_directory(
        args.input,
        args.model,
        args.output,
        args.conf,
        args.imgsz,
        args.recursive,
        save_labels=not args.no_labels,
        save_annotated=args.annotated,
        save_images=not args.no_images,
        save_conf=not args.no_conf,
        save_stats=not args.no_stats,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()

