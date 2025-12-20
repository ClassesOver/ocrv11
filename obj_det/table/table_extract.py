from ultralytics import YOLO
import numpy as np
import cv2
import os
import config
from loguru import logger
from typing import List, Dict, Tuple, Optional
from obj_det.table.table_common import (
    _empty_table_structure,
    _organize_table_structure,
    _generate_cells_from_rows_columns,
    draw_detection_result,
    crop_rows_and_columns,
    save_cropped_images, preprocess_table_region
)

# 模型配置
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "table", "best.pt")
CLASS_NAMES = ['table row', 'table column', 'table spanning cell']
IMG_SIZE = getattr(config, "TABLE_IMG_SIZE", 640)

# 初始化模型
device = config.GPUID if getattr(config, "GPU", False) else "cpu"
model = YOLO(MODEL_PATH, task='detect') if os.path.exists(MODEL_PATH) else None


def extract_table(img: np.ndarray,
                  enable_angle_correction: bool = True,
                  angle_threshold: float = 0.1,
                  enable_enhance: bool = True,
                  enable_perspective: bool = True) -> Dict:
    """
    从图像中抽取表格结构
    
    Args:
        img: 输入图像 (numpy数组)
        enable_angle_correction: 是否检测并校正角度，默认 True
        angle_threshold: 角度阈值（度），超过此值才进行校正，默认  0.1
        enable_enhance: 是否启用图像增强，默认 True
        enable_perspective: 是否启用透视校正，默认 True
        
    Returns:
        字典，包含以下字段：
            - rows: 行列表，每行包含该行的单元格
            - columns: 列列表，每列包含该列的单元格
            - cells: 所有单元格列表，每个单元格包含位置和内容信息
            - structure: 表格结构矩阵 (行x列)
            - processed_img: 预处理后的图像（如果进行了预处理）
    """
    if model is None:
        logger.warning("表格模型未找到，返回空结构")
        return _empty_table_structure()

    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        logger.warning("输入图像无效")
        return _empty_table_structure()

    try:
        # 预处理图像（角度校正、透视校正等）
        processed_img = img.copy()
        processed_img = preprocess_table_region(
            processed_img,
            enable_enhance=enable_enhance,           # 默认关闭，避免影响检测
            enable_perspective=enable_perspective,    # 默认开启，校正透视
            enable_angle_correction=enable_angle_correction,     # 根据参数决定
            angle_threshold=angle_threshold,
            output_binary=True  # 保持灰度图，便于检测
        )


        # 执行检测（使用预处理后的图像）
        results = model.predict(
            source=processed_img,
            imgsz=IMG_SIZE,
            device=device,
            verbose=False,
            half=False,
        )

        if not results or len(results) == 0:
            return _empty_table_structure()

        result = results[0]
        boxes = result.boxes

        if boxes is None or len(boxes) == 0:
            return _empty_table_structure()

        # 分类检测结果
        rows, columns, cells = _classify_detections(boxes, result.names)
        
        # 调试信息：打印检测到的类别
        logger.debug(f"检测到: {len(rows)} 行, {len(columns)} 列, {len(cells)} 单元格")
        if len(cells) == 0 and len(rows) > 0 and len(columns) > 0:
            logger.debug(f"类别名称映射: {result.names}")
            # 如果没有检测到单元格，根据行和列的交集生成单元格
            cells = _generate_cells_from_rows_columns(rows, columns)

        # 组织表格结构
        table_structure = _organize_table_structure(rows, columns, cells)
        
        # 添加预处理后的图像到返回结果（如果进行了预处理）
        if enable_perspective or enable_enhance or enable_angle_correction:
            table_structure['processed_img'] = processed_img

        return table_structure

    except Exception as e:
        logger.error(f"表格抽取错误: {e}", exc_info=True)
        return _empty_table_structure()


def _classify_detections(boxes, names: Dict[int, str]) -> Tuple[List, List, List]:
    """
    将检测结果分类为行、列和单元格
    
    Args:
        boxes: YOLO检测结果
        names: 类别名称映射
        
    Returns:
        (rows, columns, cells) 三个列表
    """
    rows = []
    columns = []
    cells = []
    
    # 收集所有检测到的类别名称用于调试
    detected_classes = set()
    
    # 从 CLASS_NAMES 获取类别名称
    row_class = CLASS_NAMES[0] if len(CLASS_NAMES) > 0 else 'table row'
    column_class = CLASS_NAMES[1] if len(CLASS_NAMES) > 1 else 'table column'
    cell_class = CLASS_NAMES[2] if len(CLASS_NAMES) > 2 else 'table spanning cell'

    for box in boxes:
        cls = int(box.cls[0].item())
        conf = float(box.conf[0].item())
        xyxy = box.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = [int(v) for v in xyxy]

        class_name = names.get(cls, '')
        detected_classes.add(class_name)
        
        box_info = {
            'bbox': [x1, y1, x2, y2],
            'confidence': conf,
            'center': [(x1 + x2) / 2, (y1 + y2) / 2]
        }

        # 使用 CLASS_NAMES 进行精确匹配
        if class_name == row_class:
            rows.append(box_info)
        elif class_name == column_class:
            columns.append(box_info)
        elif class_name == cell_class:
            cells.append(box_info)
    
    if detected_classes:
        logger.debug(f"检测到的类别: {detected_classes}")
        logger.debug(f"期望的类别: row={row_class}, column={column_class}, cell={cell_class}")

    return rows, columns, cells


# 这些函数已移至 table_common.py


def _get_cell_position(cell: Dict, rows: List, columns: List) -> Tuple[int, int]:
    """
    确定单元格在表格中的位置（行索引和列索引）
    
    Returns:
        (row_idx, col_idx) 如果无法确定则返回 (-1, -1)
    """
    cell_center = cell['center']

    # 找到最接近的行
    row_idx = -1
    min_row_dist = float('inf')
    for idx, row in enumerate(rows):
        row_center_y = row['center'][1]
        dist = abs(cell_center[1] - row_center_y)
        if dist < min_row_dist:
            min_row_dist = dist
            row_idx = idx

    # 找到最接近的列
    col_idx = -1
    min_col_dist = float('inf')
    for idx, col in enumerate(columns):
        col_center_x = col['center'][0]
        dist = abs(cell_center[0] - col_center_x)
        if dist < min_col_dist:
            min_col_dist = dist
            col_idx = idx

    return row_idx, col_idx


# 此函数已移至 table_common.py


def _boxes_overlap_vertically(box1: List, box2: List) -> bool:
    """检查两个框是否垂直重叠"""
    y1_min, y1_max = box1[0], box1[1]
    y2_min, y2_max = box2[0], box2[1]
    return not (y1_max < y2_min or y2_max < y1_min)


def _boxes_overlap_horizontally(box1: List, box2: List) -> bool:
    """检查两个框是否水平重叠"""
    x1_min, x1_max = box1[0], box1[1]
    x2_min, x2_max = box2[0], box2[1]
    return not (x1_max < x2_min or x2_max < x1_min)


# 这些函数已移至 table_common.py，通过导入使用


if __name__ == "__main__":
    img = cv2.imread(r"/images/stock_v1/line.png")
    r = extract_table(img,)
    
    # 获取预处理后的图像（如果有）
    draw_img = r.get('processed_img', img)
    
    # 在预处理后的图像上绘制检测结果
    result_img = draw_detection_result(draw_img, r)
    
    # 保存结果图像
    output_path = r"/images/stock_v1/line_detected.png"
    cv2.imwrite(output_path, result_img)
    print(f"检测结果已保存到: {output_path}")
    
    # 裁剪行和列
    cropped_data = crop_rows_and_columns(img, r, use_processed_img=True, padding=5)
    print(f"裁剪到 {len(cropped_data['rows'])} 行, {len(cropped_data['columns'])} 列")
    
    # 保存裁剪后的图像
    output_dir = r"/images/stock_v1/cropped"
    saved_paths = save_cropped_images(cropped_data, output_dir, prefix="table")
    print(f"已保存 {len(saved_paths)} 张裁剪图像到: {output_dir}")
