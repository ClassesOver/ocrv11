# -*- coding: utf-8 -*-
"""
图片分类测试脚本
遍历指定目录，使用 classify_image 对图片进行分类，并按类别组织保存
"""
import cv2
import os
import sys
import shutil
import argparse
import traceback
import time
import json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import numpy

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

# 设置分类模型路径为当前根目录 + models/classification/best.onnx
config.CLASSIFICATION_MODEL_PATH = os.path.join(project_root, "models", "classification", "best.onnx")

# 支持的图像格式
IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}

# YOLOv 分类模型相关配置
_classification_model = None
_classification_model_path = None


def _load_classification_model() -> Optional[object]:
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
            device = config.GPUID if config.GPU else 'cpu'
            
            # 加载 YOLOv11 分类模型（task='classify' 自动识别为分类任务）
            _classification_model = YOLO(model_path, task='classify')
            _classification_model_path = model_path
            
            # 获取模型信息
            model_info = "YOLOv11 分类模型"
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


def classify_image(img: numpy.ndarray, confidence_threshold: float = 0.618) -> Optional[Dict]:
    """
    使用 YOLOv11 分类模型对图像进行分类检测（优化版）
    支持 YOLOv8/YOLOv11 分类模型
    
    Args:
        img: 输入图像 (numpy array)
        confidence_threshold: 置信度阈值，低于此值的结果将被忽略（默认 0.618）
        
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


def get_unique_output_path(output_dir: str, filename: str) -> str:
    """
    获取唯一的输出路径，如果文件已存在则添加序号
    
    Args:
        output_dir: 输出目录
        filename: 文件名
        
    Returns:
        唯一的输出路径
    """
    output_path = os.path.join(output_dir, filename)
    
    if os.path.exists(output_path):
        name, ext = os.path.splitext(filename)
        counter = 1
        while os.path.exists(output_path):
            output_path = os.path.join(output_dir, f"{name}_{counter}{ext}")
            counter += 1
    
    return output_path


def process_single_image(
    image_path: str,
    output_base_dir: str,
    confidence_threshold: float = 0.618,
    move_files: bool = False,
    verbose: bool = True
) -> Optional[Dict]:
    """
    处理单张图片，进行分类并保存到对应类别目录
    
    Args:
        image_path: 输入图片路径
        output_base_dir: 输出基础目录
        confidence_threshold: 置信度阈值
        move_files: 是否移动文件（True=移动，False=复制）
        verbose: 是否输出详细信息
        
    Returns:
        dict: 分类结果，包含 'predicted_class', 'confidence', 'class_id'
              如果分类失败则返回 None
    """
    try:
        # 读取图像
        img = cv2.imread(image_path)
        if img is None:
            logger.error(f"无法读取图像: {image_path}")
            return None
        
        # 进行分类
        result = classify_image(img, confidence_threshold=confidence_threshold)
        
        if result is None:
            if verbose:
                logger.warning(f"分类失败或置信度不足: {os.path.basename(image_path)}")
            return None
        
        # 获取分类结果
        predicted_class = result.get('predicted_class', 'unknown')
        confidence = result.get('confidence', 0.0)
        
        # 创建类别目录
        class_dir = os.path.join(output_base_dir, predicted_class)
        os.makedirs(class_dir, exist_ok=True)
        
        # 生成唯一输出路径
        filename = os.path.basename(image_path)
        output_path = get_unique_output_path(class_dir, filename)
        
        # 移动或复制文件
        if move_files:
            shutil.move(image_path, output_path)
            if verbose:
                logger.info(f"✓ 移动: {os.path.basename(image_path)} -> {predicted_class}/{os.path.basename(output_path)} (置信度: {confidence:.3f})")
        else:
            shutil.copy2(image_path, output_path)
            if verbose:
                logger.info(f"✓ 复制: {os.path.basename(image_path)} -> {predicted_class}/{os.path.basename(output_path)} (置信度: {confidence:.3f})")
        
        return result
        
    except Exception as e:
        logger.error(f"处理图片失败 {image_path}: {e}", exc_info=True)
        return None


def save_statistics(
    statistics: Dict,
    output_dir: str,
    success_count: int,
    fail_count: int,
    total_count: int,
    elapsed_time: float
) -> None:
    """
    保存统计信息到文件
    
    Args:
        statistics: 分类统计字典
        output_dir: 输出目录
        success_count: 成功数量
        fail_count: 失败数量
        total_count: 总数量
        elapsed_time: 耗时（秒）
    """
    stats_file = os.path.join(output_dir, 'classification_statistics.json')
    stats_data = {
        'summary': {
            'total': total_count,
            'success': success_count,
            'failed': fail_count,
            'success_rate': f"{(success_count / total_count * 100):.2f}%" if total_count > 0 else "0%",
            'elapsed_time_seconds': round(elapsed_time, 2),
            'average_time_per_image': round(elapsed_time / total_count, 3) if total_count > 0 else 0
        },
        'class_distribution': statistics
    }
    
    try:
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, ensure_ascii=False, indent=2)
        logger.info(f"统计信息已保存到: {stats_file}")
    except Exception as e:
        logger.warning(f"保存统计信息失败: {e}")


def process_directory(
    input_dir: str,
    output_dir: str,
    confidence_threshold: float = 0.618,
    move_files: bool = False,
    recursive: bool = False,
    save_stats: bool = True,
    verbose: bool = True
) -> Tuple[int, int, Dict]:
    """
    批量处理目录中的所有图片
    
    Args:
        input_dir: 输入目录
        output_dir: 输出目录
        confidence_threshold: 置信度阈值
        move_files: 是否移动文件
        recursive: 是否递归搜索子目录
        save_stats: 是否保存统计信息
        verbose: 是否输出详细信息
        
    Returns:
        (success_count, fail_count, class_statistics)
    """
    if not os.path.exists(input_dir):
        logger.error(f"输入目录不存在: {input_dir}")
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
    
    # 统计信息
    success_count = 0
    fail_count = 0
    class_statistics = {}  # {class_name: count}
    
    # 开始计时
    start_time = time.time()
    
    # 处理每张图片（使用进度条）
    progress_bar = tqdm(
        enumerate(image_files, 1),
        total=len(image_files),
        desc="分类处理",
        unit="张",
        disable=not verbose
    )
    
    for idx, image_path in progress_bar:
        filename = os.path.basename(image_path)
        
        # 更新进度条描述
        if HAS_TQDM:
            progress_bar.set_description(f"处理: {filename[:30]}")
        
        # 进行分类
        result = process_single_image(
            image_path,
            output_dir,
            confidence_threshold,
            move_files,
            verbose=verbose
        )
        
        if result:
            success_count += 1
            predicted_class = result.get('predicted_class', 'unknown')
            class_statistics[predicted_class] = class_statistics.get(predicted_class, 0) + 1
        else:
            fail_count += 1
    
    # 计算耗时
    elapsed_time = time.time() - start_time
    
    # 打印统计信息
    logger.info("\n" + "=" * 60)
    logger.info("处理完成！")
    logger.info("=" * 60)
    logger.info(f"成功分类: {success_count} 张")
    logger.info(f"分类失败: {fail_count} 张")
    logger.info(f"总计: {len(image_files)} 张")
    logger.info(f"耗时: {elapsed_time:.2f} 秒 (平均 {elapsed_time/len(image_files):.3f} 秒/张)")
    
    # 打印分类统计
    if class_statistics:
        logger.info("\n分类统计:")
        logger.info("-" * 60)
        for class_name, count in sorted(class_statistics.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / len(image_files) * 100) if len(image_files) > 0 else 0
            logger.info(f"  {class_name}: {count} 张 ({percentage:.1f}%)")
        logger.info("-" * 60)
    
    logger.info(f"\n输出目录: {output_dir}")
    
    # 保存统计信息
    if save_stats:
        save_statistics(class_statistics, output_dir, success_count, fail_count, len(image_files), elapsed_time)
    
    logger.info("=" * 60)
    
    return success_count, fail_count, class_statistics


def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='图片分类测试脚本 - 遍历目录对图片进行分类并按类别组织保存',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基本用法
  python tests/test_cls.py -i images/classify -o images/classified
  
  # 指定置信度阈值
  python tests/test_cls.py -i images/classify -o images/classified --threshold 0.7
  
  # 移动文件而不是复制
  python tests/test_cls.py -i images/classify -o images/classified --move
  
  # 递归搜索子目录
  python tests/test_cls.py -i images/classify -o images/classified --recursive
  
  # 静默模式（不显示详细信息）
  python tests/test_cls.py -i images/classify -o images/classified --quiet
        """
    )
    
    parser.add_argument(
        '-i', '--input',
        type=str,
        default=os.path.join(config.base_dir, 'images', 'classify'),
        help='输入目录：包含待分类图片的目录（默认: images/classify）'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=str,
        default=os.path.join(config.base_dir, 'images', 'classified'),
        help='输出目录：按类别组织的图片目录（默认: images/classified）'
    )
    
    parser.add_argument(
        '-t', '--threshold',
        type=float,
        default=0.618,
        help='置信度阈值，低于此值的结果将被忽略（默认: 0.618）'
    )
    
    parser.add_argument(
        '--move',
        action='store_true',
        help='移动文件而不是复制（默认: 复制）'
    )
    
    parser.add_argument(
        '-r', '--recursive',
        action='store_true',
        help='递归搜索子目录中的图片'
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
    logger.info("图片分类测试")
    logger.info("=" * 60)
    logger.info(f"输入目录: {args.input}")
    logger.info(f"输出目录: {args.output}")
    logger.info(f"置信度阈值: {args.threshold}")
    logger.info(f"操作模式: {'移动' if args.move else '复制'}")
    logger.info(f"递归搜索: {'是' if args.recursive else '否'}")
    logger.info("=" * 60)
    
    # 批量处理
    process_directory(
        args.input,
        args.output,
        args.threshold,
        args.move,
        args.recursive,
        save_stats=not args.no_stats,
        verbose=not args.quiet
    )


if __name__ == "__main__":
    main()
