"""
Table Transformer 工具函数
用于表格检测和结构识别
参考: https://www.e2enetworks.com/blog/table-detection-and-transformation-using-tatr-table-transformer-using-tatr-on-e2e-cloud
"""

import torch
from transformers import TableTransformerForObjectDetection
from PIL import Image
from torchvision import transforms
import numpy as np
import cv2
import os
from loguru import logger
from typing import List, Dict, Tuple, Optional
from obj_det.table.table_common import (
    _empty_table_structure,
    _organize_table_structure,
    _generate_cells_from_rows_columns,
    preprocess_table_region,
    draw_detection_result,
    crop_rows_and_columns,
    save_cropped_images
)

# 尝试导入 config，如果不存在则使用默认值
try:
    import config

    GPU = getattr(config, "GPU", False)
    GPUID = getattr(config, "GPUID", 0)
    TABLE_IMG_SIZE = getattr(config, "TABLE_IMG_SIZE", 1000)
except ImportError:
    GPU = False
    GPUID = 0
    TABLE_IMG_SIZE = 1000


class MaxResize(object):
    """调整图像大小，保持宽高比"""

    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize(
            (int(round(scale * width)), int(round(scale * height)))
        )
        return resized_image


# 全局模型实例
_detector = None
_recognizer = None


def _get_detector():
    """获取或创建检测器实例（单例模式）"""
    global _detector
    if _detector is None:
        _detector = TableDetector()
    return _detector


def _get_recognizer():
    """获取或创建识别器实例（单例模式）"""
    global _recognizer
    if _recognizer is None:
        _recognizer = TableStructureRecognizer()
    return _recognizer


def outputs_to_objects(outputs, img_size, id2label):
    """
    将模型输出转换为对象列表

    Args:
        outputs: 模型输出
        img_size: 图像尺寸 (width, height)
        id2label: ID到标签的映射

    Returns:
        list: 检测到的对象列表
    """
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in pred_bboxes]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if class_label == 'no object':
            continue

        # 转换bbox格式: (center_x, center_y, width, height) -> (xmin, ymin, xmax, ymax)
        bbox = [
            (bbox[0] - bbox[2] / 2) * img_size[0],  # xmin
            (bbox[1] - bbox[3] / 2) * img_size[1],  # ymin
            (bbox[0] + bbox[2] / 2) * img_size[0],  # xmax
            (bbox[1] + bbox[3] / 2) * img_size[1]  # ymax
        ]

        objects.append({
            'label': class_label,
            'score': float(score),
            'bbox': bbox
        })

    return objects


class TableDetector:
    """表格检测器（本地模型）"""

    def __init__(self, model_name="microsoft/table-transformer-detection", device=None):
        """
        初始化表格检测器

        Args:
            model_name: 模型名称
            device: 设备 (cuda/cpu)，None则自动选择
        """
        if device is None:
            if GPU and torch.cuda.is_available():
                self.device = f"cuda:{GPUID}" if GPUID >= 0 else "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        logger.info(f"表格检测器使用设备: {self.device}")

        # 本地加载模型
        # 使用相对路径：项目根目录下的 models/table_transformer
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'table_transformer')
        cache_dir = os.path.normpath(cache_dir)  # 规范化路径
        self.model = TableTransformerForObjectDetection.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True  # 只用本地缓存，不联网
        )

        self.model.to(self.device)
        self.model.eval()

        # 获取标签映射
        self.id2label = self.model.config.id2label
        self.id2label[len(self.id2label)] = "no object"

        # 图像预处理
        self.detection_transform = transforms.Compose([
            MaxResize(800),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        logger.info("表格检测模型本地加载完成")

    def detect(self, image, threshold=0.7):
        """
        检测图像中的表格

        Args:
            image: PIL Image 对象或图像路径
            threshold: 置信度阈值

        Returns:
            list: 检测到的表格列表，每个表格包含 label, score, bbox
        """
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("image 必须是 PIL Image 对象或图像路径")

        original_size = image.size

        # 预处理
        pixel_values = self.detection_transform(image).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(pixel_values)

        # 后处理
        objects = outputs_to_objects(outputs, original_size, self.id2label)

        # 过滤低置信度结果
        objects = [obj for obj in objects if obj['score'] >= threshold]

        return objects


class TableStructureRecognizer:
    """表格结构识别器（本地模型）"""

    def __init__(self, model_name="microsoft/table-transformer-structure-recognition-v1.1-all", device=None):
        """
        初始化表格结构识别器

        Args:
            model_name: 模型名称
            device: 设备 (cuda/cpu)，None则自动选择
        """
        if device is None:
            if GPU and torch.cuda.is_available():
                self.device = f"cuda:{GPUID}" if GPUID >= 0 else "cuda"
            else:
                self.device = "cpu"
        else:
            self.device = device
        logger.info(f"结构识别器使用设备: {self.device}")

        # 本地加载模型
        # 使用相对路径：项目根目录下的 models/table_transformer
        cache_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'models', 'table_transformer')
        cache_dir = os.path.normpath(cache_dir)  # 规范化路径
        self.model = TableTransformerForObjectDetection.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True
        )

        self.model.to(self.device)
        self.model.eval()

        # 获取标签映射
        self.id2label = self.model.config.id2label
        self.id2label[len(self.id2label)] = "no object"

        # 图像预处理
        self.structure_transform = transforms.Compose([
            MaxResize(TABLE_IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        logger.info("表格结构识别模型本地加载完成")

    def recognize(self, image, threshold=0.6, auto_adjust_threshold=True):
        """
        识别表格结构

        Args:
            image: PIL Image 对象或图像路径（应该是裁剪后的表格图像）
            threshold: 置信度阈值
            auto_adjust_threshold: 如果未检测到行列，自动降低阈值重试

        Returns:
            dict: 包含 cells, rows, columns 等结构信息
        """
        # 加载图像
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif not isinstance(image, Image.Image):
            raise ValueError("image 必须是 PIL Image 对象或图像路径")

        original_size = image.size

        # 检查图像尺寸
        if original_size[0] < 200 or original_size[1] < 200:
            logger.warning(f"图像尺寸较小 {original_size}，可能影响识别效果")

        # 预处理
        pixel_values = self.structure_transform(image).unsqueeze(0).to(self.device)

        # 推理
        with torch.no_grad():
            outputs = self.model(pixel_values)

        # 后处理（先不过滤，获取所有结果）
        all_cells = outputs_to_objects(outputs, original_size, self.id2label)

        # 尝试不同的阈值
        best_structure = None
        tried_thresholds = [threshold]

        # 如果启用自动调整，准备多个阈值
        if auto_adjust_threshold:
            tried_thresholds = [threshold, threshold * 0.8, threshold * 0.6, threshold * 0.4, 0.3]

        for try_threshold in tried_thresholds:
            # 过滤低置信度结果
            cells = [cell for cell in all_cells if cell['score'] >= try_threshold]

            # 分类结构元素
            structure = {
                'all_cells': cells,
                'rows': [c for c in cells if c['label'] == 'table row'],
                'columns': [c for c in cells if c['label'] == 'table column'],
                'headers': [c for c in cells if c['label'] == 'table column header'],
                'spanning_cells': [c for c in cells if c['label'] == 'table spanning cell'],
            }

            # 统计信息
            structure['stats'] = {
                'total_cells': len(cells),
                'num_rows': len(structure['rows']),
                'num_columns': len(structure['columns']),
                'num_headers': len(structure['headers']),
                'num_spanning': len(structure['spanning_cells']),
                'threshold_used': try_threshold
            }

            # 如果找到了行和列，返回结果
            if structure['stats']['num_rows'] > 0 and structure['stats']['num_columns'] > 0:
                if try_threshold != threshold:
                    logger.info(f"自动调整阈值: {threshold:.2f} -> {try_threshold:.2f}")
                best_structure = structure
                break

            # 记录最佳结果
            if best_structure is None:
                best_structure = structure
            elif (structure['stats']['num_rows'] + structure['stats']['num_columns'] >
                  best_structure['stats']['num_rows'] + best_structure['stats']['num_columns']):
                best_structure = structure

        # 如果仍然没有检测到行列，给出警告
        if best_structure['stats']['num_rows'] == 0 or best_structure['stats']['num_columns'] == 0:
            logger.warning(f"未检测到完整的表格结构")
            logger.debug(f"尝试了阈值: {tried_thresholds}")
            logger.debug(f"最终结果: {best_structure['stats']['num_rows']} 行, "
                         f"{best_structure['stats']['num_columns']} 列")

            # 给出建议
            if best_structure['stats']['total_cells'] > 0:
                logger.info(f"检测到 {best_structure['stats']['total_cells']} 个元素，"
                            f"可能是无框表格或表格边界不明显")

        return best_structure


def get_cell_coordinates_by_row(table_structure):
    """
    按行获取单元格坐标

    Args:
        table_structure: recognize() 返回的结构字典

    Returns:
        list: 每行的单元格坐标列表
    """
    rows = table_structure['rows']
    columns = table_structure['columns']

    # 按Y坐标排序行，按X坐标排序列
    rows.sort(key=lambda x: x['bbox'][1])
    columns.sort(key=lambda x: x['bbox'][0])

    def find_cell_coordinates(row, column):
        """计算单元格坐标"""
        cell_bbox = [
            column['bbox'][0],  # xmin
            row['bbox'][1],  # ymin
            column['bbox'][2],  # xmax
            row['bbox'][3]  # ymax
        ]
        return cell_bbox

    # 生成单元格坐标
    cell_coordinates = []
    for row in rows:
        row_cells = []
        for column in columns:
            cell_bbox = find_cell_coordinates(row, column)
            row_cells.append({
                'column': column['bbox'],
                'cell': cell_bbox,
                'row_bbox': row['bbox']
            })

        # 按X坐标排序单元格
        row_cells.sort(key=lambda x: x['column'][0])

        cell_coordinates.append({
            'row': row['bbox'],
            'cells': row_cells,
            'cell_count': len(row_cells)
        })

    # 从上到下排序行
    cell_coordinates.sort(key=lambda x: x['row'][1])

    return cell_coordinates


def extract_table(img: np.ndarray,
                  enable_angle_correction: bool = True,
                  angle_threshold: float = 0.1,
                  enable_enhance: bool = True,
                  enable_perspective: bool = True,
                  structure_threshold: float = 0.6,
                  auto_adjust_threshold: bool = True,
                  selected_columns: Optional[List[int]] = None) -> Dict:
    """
    从图像中抽取表格结构（统一接口，参考 table_extract.py）
    
    Args:
        img: 输入图像 (numpy数组，BGR格式)
        enable_angle_correction: 是否检测并校正角度，默认 True
        angle_threshold: 角度阈值（度），超过此值才进行校正，默认 0.1
        enable_enhance: 是否启用图像增强，默认 True
        enable_perspective: 是否启用透视校正，默认 True
        structure_threshold: 结构识别置信度阈值，默认 0.6
        auto_adjust_threshold: 是否自动调整阈值，默认 True
        selected_columns: 指定要获取的列索引列表（从0开始），如果为None则返回所有列，例如 [0, 2, 3] 表示只获取第0、2、3列
        
    Returns:
        字典，包含以下字段：
            - rows: 行列表，每行包含该行的单元格（如果指定了selected_columns，则只包含指定列的单元格）
            - columns: 列列表，每列包含该列的单元格
            - cells: 所有单元格列表，每个单元格包含位置和内容信息
            - structure: 表格结构矩阵 (行x列)
            - processed_img: 预处理后的图像（如果进行了预处理）
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        logger.warning("输入图像无效")
        return _empty_table_structure()

    try:
        # 预处理图像（角度校正、透视校正等）
        processed_img = img.copy()
        processed_img = preprocess_table_region(
            processed_img,
            enable_enhance=enable_enhance,  # 默认开启，提升图像质量
            enable_perspective=enable_perspective,  # 默认开启，校正透视
            enable_angle_correction=enable_angle_correction,  # 根据参数决定
            denoise_strength=3,
            sharpen_strength=1.0,
            angle_threshold=angle_threshold,
            output_binary=False  # 保持彩色/灰度图，便于检测
        )

        # 转换为 PIL Image
        if len(processed_img.shape) == 3:
            # BGR to RGB
            pil_img = Image.fromarray(cv2.cvtColor(processed_img, cv2.COLOR_BGR2RGB))
        else:
            pil_img = Image.fromarray(processed_img).convert("RGB")

        # 使用结构识别器识别表格结构
        recognizer = _get_recognizer()
        structure_result = recognizer.recognize(
            pil_img,
            threshold=structure_threshold,
            auto_adjust_threshold=auto_adjust_threshold
        )

        # 只提取行和列，不识别单元格
        rows_raw = structure_result.get('rows', [])
        columns_raw = structure_result.get('columns', [])

        # 转换为统一格式
        rows = []
        columns = []

        # 处理行
        for row in rows_raw:
            bbox = row['bbox']
            rows.append({
                'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                'confidence': row.get('score', 1.0),
                'center': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            })

        # 处理列
        for col in columns_raw:
            bbox = col['bbox']
            columns.append({
                'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                'confidence': col.get('score', 1.0),
                'center': [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2]
            })

        # 根据行和列的交集生成单元格（不识别单元格）
        # 传入空的 cells 列表，_organize_table_structure 会自动根据行列生成单元格
        cells = []
        logger.debug(f"检测到: {len(rows)} 行, {len(columns)} 列，将根据行列交集生成单元格")

        # 组织表格结构（会自动根据行列生成单元格）
        table_structure = _organize_table_structure(rows, columns, cells)

        # 如果指定了 selected_columns，则过滤 rows 中的 cells，只保留指定列的数据
        if selected_columns is not None and len(selected_columns) > 0:
            # 将 selected_columns 转换为集合以便快速查找
            selected_cols_set = set(selected_columns)
            
            # 过滤每行的 cells，只保留指定列的单元格
            filtered_rows = []
            for row in table_structure['rows']:
                filtered_cells = [cell for cell in row.get('cells', [])
                                 if cell.get('col') in selected_cols_set]
                filtered_rows.append({
                    'bbox': row['bbox'],
                    'cells': filtered_cells
                })
            filtered_columns = []
            for col_idx, column in enumerate(table_structure['columns']):
                if col_idx in selected_cols_set:
                    filtered_columns.append(column)
            table_structure['columns'] = filtered_columns
            table_structure['rows'] = filtered_rows
            logger.debug(f"已过滤列: 只保留列索引 {selected_columns}，每行单元格数已更新")

        # 添加预处理后的图像到返回结果（如果进行了预处理）
        if enable_perspective or enable_enhance or enable_angle_correction:
            table_structure['processed_img'] = processed_img

        logger.debug(
            f"最终结果: {len(table_structure['rows'])} 行, {len(table_structure['columns'])} 列, {len(table_structure['cells'])} 单元格")

        return table_structure

    except Exception as e:
        logger.error(f"表格抽取错误: {e}", exc_info=True)
        return _empty_table_structure()


# 这些函数已移至 table_common.py，通过导入使用


# 这些函数已移至 table_common.py，通过导入使用
# 这些函数已移至 table_common.py，通过导入使用


def detect_and_extract_tables(image_path, detection_threshold=0.7):
    """
    检测图像中的所有表格并返回裁剪后的表格图像

    Args:
        image_path: 图像路径
        detection_threshold: 检测阈值

    Returns:
        list: 包含表格信息和裁剪图像的列表
    """
    # 加载图像
    image = Image.open(image_path).convert("RGB")

    # 初始化检测器
    detector = TableDetector()

    # 检测表格
    tables = detector.detect(image, threshold=detection_threshold)

    # 裁剪表格
    extracted_tables = []
    for i, table in enumerate(tables):
        bbox = table['bbox']
        cropped = image.crop(bbox)

        extracted_tables.append({
            'index': i,
            'bbox': bbox,
            'score': table['score'],
            'image': cropped
        })

    return extracted_tables


recognizer = _get_recognizer()

if __name__ == "__main__":
    # 使用新的统一接口示例
    img_path = r"/images/stock_v1/line.png"
    img = cv2.imread(img_path)
    # 使用统一接口提取表格
    result = extract_table(img, structure_threshold=0.6)
    print(f"检测结果: {len(result['rows'])} 行, {len(result['columns'])} 列, {len(result['cells'])} 单元格")

    # 获取预处理后的图像（如果有）
    draw_img = result.get('processed_img', img)

    # 在预处理后的图像上绘制检测结果
    result_img = draw_detection_result(draw_img, result)

    # 保存结果图像
    output_path = r"/images/stock_v1/line_detected_transformer.png"
    cv2.imwrite(output_path, result_img)
    print(f"检测结果已保存到: {output_path}")

    # 裁剪行和列
    cropped_data = crop_rows_and_columns(img, result, use_processed_img=True, padding=5)
    print(f"裁剪到 {len(cropped_data['rows'])} 行, {len(cropped_data['columns'])} 列")

    # 保存裁剪后的图像
    output_dir = r"/images/stock_v1/cropped_transformer"
    saved_paths = save_cropped_images(cropped_data, output_dir, prefix="table_transformer")
    print(f"已保存 {len(saved_paths)} 张裁剪图像到: {output_dir}")
