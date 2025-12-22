#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
YOLOv分类数据集拆分脚本
将分类数据集拆分为训练集、验证集和测试集
"""

import os
import shutil
import argparse
import random
from pathlib import Path
from tqdm import tqdm


def split_dataset(source_dir, output_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    拆分YOLOv分类数据集
    
    Args:
        source_dir: 源数据集目录路径
        output_dir: 输出目录路径
        train_ratio: 训练集比例（默认0.7）
        val_ratio: 验证集比例（默认0.15）
        test_ratio: 测试集比例（默认0.15）
        seed: 随机种子（默认42）
    """
    # 验证比例
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError(f"比例总和必须等于1.0，当前为: {train_ratio + val_ratio + test_ratio}")
    
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        raise ValueError(f"源目录不存在: {source_dir}")
    
    # 设置随机种子
    random.seed(seed)
    
    # 创建输出目录结构
    train_dir = output_path / 'train'
    val_dir = output_path / 'val'
    test_dir = output_path / 'test'
    
    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    
    # 获取所有类别文件夹
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    if not class_dirs:
        raise ValueError(f"在 {source_dir} 中未找到类别文件夹")
    
    print(f"找到 {len(class_dirs)} 个类别文件夹")
    print(f"拆分比例: 训练集={train_ratio:.1%}, 验证集={val_ratio:.1%}, 测试集={test_ratio:.1%}")
    print(f"随机种子: {seed}")
    print("-" * 60)
    
    total_images = 0
    total_train = 0
    total_val = 0
    total_test = 0
    
    # 遍历每个类别
    for class_dir in tqdm(class_dirs, desc="处理类别"):
        class_name = class_dir.name
        
        # 获取该类别的所有图片文件
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.webp'}
        image_files = [f for f in class_dir.iterdir() 
                      if f.is_file() and f.suffix.lower() in image_extensions]
        
        if not image_files:
            print(f"警告: 类别 '{class_name}' 中没有找到图片文件，跳过")
            continue
        
        # 随机打乱
        random.shuffle(image_files)
        
        # 计算拆分数量
        n_total = len(image_files)
        n_train = int(n_total * train_ratio)
        n_val = int(n_total * val_ratio)
        n_test = n_total - n_train - n_val  # 确保所有图片都被分配
        
        # 拆分文件列表
        train_files = image_files[:n_train]
        val_files = image_files[n_train:n_train + n_val]
        test_files = image_files[n_train + n_val:]
        
        # 创建类别文件夹
        train_class_dir = train_dir / class_name
        val_class_dir = val_dir / class_name
        test_class_dir = test_dir / class_name
        
        train_class_dir.mkdir(exist_ok=True)
        val_class_dir.mkdir(exist_ok=True)
        test_class_dir.mkdir(exist_ok=True)
        
        # 复制文件
        for img_file in train_files:
            shutil.copy2(img_file, train_class_dir / img_file.name)
        
        for img_file in val_files:
            shutil.copy2(img_file, val_class_dir / img_file.name)
        
        for img_file in test_files:
            shutil.copy2(img_file, test_class_dir / img_file.name)
        
        # 统计信息
        total_images += n_total
        total_train += n_train
        total_val += n_val
        total_test += n_test
        
        print(f"类别 '{class_name}': 总计={n_total}, 训练={n_train}, 验证={n_val}, 测试={n_test}")
    
    print("-" * 60)
    print(f"拆分完成！")
    print(f"总计图片: {total_images}")
    print(f"训练集: {total_train} ({total_train/total_images:.1%})")
    print(f"验证集: {total_val} ({total_val/total_images:.1%})")
    print(f"测试集: {total_test} ({total_test/total_images:.1%})")
    print(f"\n输出目录: {output_path.absolute()}")


def main():
    parser = argparse.ArgumentParser(
        description='YOLOv分类数据集拆分工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 使用默认比例 (70% 训练, 15% 验证, 15% 测试)
  python split_classification_dataset.py --source ./dataset --output ./dataset_split
  
  # 自定义比例
  python split_classification_dataset.py --source ./dataset --output ./dataset_split --train 0.8 --val 0.1 --test 0.1
  
  # 指定随机种子
  python split_classification_dataset.py --source ./dataset --output ./dataset_split --seed 123
        """
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        required=True,
        help='源数据集目录路径（包含类别文件夹的目录）'
    )
    
    parser.add_argument(
        '--output', '-o',
        type=str,
        required=True,
        help='输出目录路径（将创建train/val/test子目录）'
    )
    
    parser.add_argument(
        '--train',
        type=float,
        default=0.7,
        help='训练集比例（默认: 0.7）'
    )
    
    parser.add_argument(
        '--val',
        type=float,
        default=0.15,
        help='验证集比例（默认: 0.15）'
    )
    
    parser.add_argument(
        '--test',
        type=float,
        default=0.15,
        help='测试集比例（默认: 0.15）'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='随机种子（默认: 42）'
    )
    
    args = parser.parse_args()
    
    try:
        split_dataset(
            source_dir=args.source,
            output_dir=args.output,
            train_ratio=args.train,
            val_ratio=args.val,
            test_ratio=args.test,
            seed=args.seed
        )
    except Exception as e:
        print(f"错误: {e}")
        return 1
    
    return 0


if __name__ == '__main__':
    exit(main())

