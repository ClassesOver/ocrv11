"""
表格处理的公共函数模块
用于 table_extract.py 和 table_transformers.py 的公共功能
"""

import numpy as np
import cv2
import os
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from loguru import logger


def _empty_table_structure() -> Dict:
    """返回空的表格结构"""
    return {
        'rows': [],
        'columns': [],
        'cells': [],
        'structure': []
    }


def _generate_cells_from_rows_columns(rows: List, columns: List) -> List:
    """
    根据行和列的交集生成单元格
    
    Args:
        rows: 行检测框列表
        columns: 列检测框列表
        
    Returns:
        单元格列表
    """
    cells = []
    
    if not rows or not columns:
        return cells
    
    # 按位置排序
    rows_sorted = sorted(rows, key=lambda x: x['center'][1])
    columns_sorted = sorted(columns, key=lambda x: x['center'][0])
    
    for row in rows_sorted:
        row_y1, row_y2 = row['bbox'][1], row['bbox'][3]
        for col in columns_sorted:
            col_x1, col_x2 = col['bbox'][0], col['bbox'][2]
            
            # 计算交集区域
            cell_x1 = max(row['bbox'][0], col_x1)
            cell_y1 = max(row_y1, col['bbox'][1])
            cell_x2 = min(row['bbox'][2], col_x2)
            cell_y2 = min(row_y2, col['bbox'][3])
            
            # 确保是有效的交集
            if cell_x1 < cell_x2 and cell_y1 < cell_y2:
                cell_info = {
                    'bbox': [cell_x1, cell_y1, cell_x2, cell_y2],
                    'confidence': min(row.get('confidence', 1.0), col.get('confidence', 1.0)),
                    'center': [(cell_x1 + cell_x2) / 2, (cell_y1 + cell_y2) / 2]
                }
                cells.append(cell_info)
    
    logger.debug(f"根据行和列生成了 {len(cells)} 个单元格")
    return cells


def _calculate_cell_from_row_column(row_box: Dict, col_box: Dict, row_idx: int, col_idx: int) -> Optional[Dict]:
    """
    根据行和列的交集计算单元格
    
    Args:
        row_box: 行检测框
        col_box: 列检测框
        row_idx: 行索引
        col_idx: 列索引
        
    Returns:
        单元格信息字典，如果无效则返回 None
    """
    # 计算交集区域
    cell_x1 = max(row_box['bbox'][0], col_box['bbox'][0])
    cell_y1 = max(row_box['bbox'][1], col_box['bbox'][1])
    cell_x2 = min(row_box['bbox'][2], col_box['bbox'][2])
    cell_y2 = min(row_box['bbox'][3], col_box['bbox'][3])
    
    # 确保是有效的交集
    if cell_x1 < cell_x2 and cell_y1 < cell_y2:
        return {
            'bbox': [cell_x1, cell_y1, cell_x2, cell_y2],
            'confidence': min(row_box.get('confidence', 1.0), col_box.get('confidence', 1.0)),
            'center': [(cell_x1 + cell_x2) / 2, (cell_y1 + cell_y2) / 2],
            'row': row_idx,
            'col': col_idx
        }
    
    return None


def _build_structure_matrix(rows: List, columns: List, cells: List) -> List[List]:
    """
    构建表格结构矩阵
    
    Returns:
        二维列表，表示表格结构
    """
    if not rows or not columns:
        return []

    # 初始化矩阵
    structure = [[None for _ in range(len(columns))] for _ in range(len(rows))]

    # 填充单元格
    for cell in cells:
        row_idx = cell.get('row', -1)
        col_idx = cell.get('col', -1)
        if 0 <= row_idx < len(rows) and 0 <= col_idx < len(columns):
            structure[row_idx][col_idx] = {
                'bbox': cell['bbox'],
                'confidence': cell.get('confidence', 0.0)
            }

    return structure


def _organize_table_structure(rows: List, columns: List, cells: List) -> Dict:
    """
    组织表格结构
    
    Args:
        rows: 行检测框列表
        columns: 列检测框列表
        cells: 单元格检测框列表（可选，如果为空则根据行列计算）
        
    Returns:
        表格结构字典
    """
    # 按位置排序
    rows_sorted = sorted(rows, key=lambda x: x['center'][1])
    columns_sorted = sorted(columns, key=lambda x: x['center'][0])

    # 如果没有检测到单元格，根据行列交集生成
    if not cells:
        cells = _generate_cells_from_rows_columns(rows_sorted, columns_sorted)

    # 构建行列表：每行的单元格根据该行与所有列的交集计算
    table_rows = []
    for row_idx, row_box in enumerate(rows_sorted):
        row_cells = []
        for col_idx, col_box in enumerate(columns_sorted):
            cell = _calculate_cell_from_row_column(row_box, col_box, row_idx, col_idx)
            if cell:
                row_cells.append(cell)
        table_rows.append({
            'bbox': row_box['bbox'],
            'cells': row_cells
        })

    # 构建列列表：每列的单元格根据该列与所有行的交集计算
    table_columns = []
    for col_idx, col_box in enumerate(columns_sorted):
        col_cells = []
        for row_idx, row_box in enumerate(rows_sorted):
            cell = _calculate_cell_from_row_column(row_box, col_box, row_idx, col_idx)
            if cell:
                col_cells.append(cell)
        table_columns.append({
            'bbox': col_box['bbox'],
            'cells': col_cells
        })

    # 构建单元格列表（带位置信息）
    cells_with_position = []
    for row_idx, row_box in enumerate(rows_sorted):
        for col_idx, col_box in enumerate(columns_sorted):
            cell = _calculate_cell_from_row_column(row_box, col_box, row_idx, col_idx)
            if cell:
                cells_with_position.append({
                    'bbox': cell['bbox'],
                    'confidence': cell.get('confidence', 1.0),
                    'row': row_idx,
                    'col': col_idx
                })

    # 构建结构矩阵
    structure = _build_structure_matrix(rows_sorted, columns_sorted, cells_with_position)

    return {
        'rows': table_rows,
        'columns': table_columns,
        'cells': cells_with_position,
        'structure': structure
    }


def _detect_table_angle(img: np.ndarray) -> float:
    """
    检测表格图像的倾斜角度
    
    Args:
        img: 输入图像
        
    Returns:
        倾斜角度（度），正值表示顺时针旋转，需要逆时针纠正
    """
    try:
        # 转换为灰度图
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)

        # 使用霍夫线变换检测直线
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None or len(lines) == 0:
            return 0.0

        # 计算角度
        angles = []
        for line in lines:
            rho, theta = line[0]
            # 转换为角度（度）
            # theta 范围是 [0, π]，转换为 [-90, 90] 度
            angle = np.degrees(theta) - 90
            # 只考虑接近水平或垂直的线（-45到45度范围）
            if -45 <= angle <= 45:
                angles.append(angle)

        if not angles:
            return 0.0

        # 计算平均角度
        avg_angle = np.mean(angles)

        # 如果角度接近90度，可能是垂直线，需要调整
        if abs(avg_angle) > 45:
            avg_angle = avg_angle - 90 if avg_angle > 0 else avg_angle + 90

        # 返回负值，因为霍夫变换的角度定义与旋转方向相反
        # 如果图像顺时针倾斜，我们需要逆时针旋转来纠正
        return -avg_angle

    except Exception as e:
        logger.error(f"角度检测错误: {e}", exc_info=True)
        return 0.0


def _rotate_image(img: np.ndarray, angle: float) -> np.ndarray:
    """
    旋转图像
    
    Args:
        img: 输入图像
        angle: 旋转角度（度），正值表示逆时针旋转
        
    Returns:
        旋转后的图像
    """
    try:
        if abs(angle) < 0.1:
            return img

        h, w = img.shape[:2]
        center = (w // 2, h // 2)

        # 计算旋转矩阵
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)

        # 计算新的图像尺寸
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))

        # 调整旋转矩阵的平移部分
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        # 执行旋转
        rotated = cv2.warpAffine(img, rotation_matrix, (new_w, new_h),
                                 flags=cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_CONSTANT,
                                 borderValue=(255, 255, 255) if len(img.shape) == 3 else 255)

        return rotated

    except Exception as e:
        logger.error(f"图像旋转错误: {e}", exc_info=True)
        return img





def draw_detection_result(img: np.ndarray, result: Dict,
                          row_color: Tuple[int, int, int] = (0, 255, 0),
                          column_color: Tuple[int, int, int] = (255, 0, 0),
                          thickness: int = 3,
                          show_labels: bool = True,
                          shadow_offset: int = 2,
                          inner_border: bool = True) -> np.ndarray:
    """
    在图像上绘制检测结果，只标记行和列（增强边框效果）
    
    Args:
        img: 输入图像
        result: extract_table 返回的结果字典
        row_color: 行的颜色 (B, G, R)，默认绿色
        column_color: 列的颜色 (B, G, R)，默认红色
        thickness: 线条粗细，默认 3
        show_labels: 是否显示标签，默认 True
        shadow_offset: 阴影偏移量，默认 2
        inner_border: 是否绘制内边框，默认 True
        
    Returns:
        绘制了检测结果的图像
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return img
    
    # 创建副本
    result_img = img.copy()
    
    # 如果是灰度图，转换为彩色
    if len(result_img.shape) == 2:
        result_img = cv2.cvtColor(result_img, cv2.COLOR_GRAY2BGR)
    
    rows = result.get('rows', [])
    columns = result.get('columns', [])
    
    def _draw_enhanced_rectangle(img, x1, y1, x2, y2, color, thickness, shadow_offset, inner_border):
        """绘制增强边框的矩形"""
        # 绘制阴影（外边框）
        shadow_color = (0, 0, 0)  # 黑色阴影
        cv2.rectangle(
            img,
            (x1 + shadow_offset, y1 + shadow_offset),
            (x2 + shadow_offset, y2 + shadow_offset),
            shadow_color,
            thickness + 2
        )
        
        # 绘制主边框（外边框，更粗）
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness + 1)
        
        # 绘制内边框（高亮效果）
        if inner_border:
            # 计算内边框位置（向内缩进）
            inner_offset = max(1, thickness // 2)
            inner_color = tuple(min(255, c + 50) for c in color)  # 更亮的颜色
            cv2.rectangle(
                img,
                (x1 + inner_offset, y1 + inner_offset),
                (x2 - inner_offset, y2 - inner_offset),
                inner_color,
                1
            )
    
    # 绘制行
    for idx, row in enumerate(rows):
        bbox = row.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            _draw_enhanced_rectangle(
                result_img, x1, y1, x2, y2,
                row_color, thickness, shadow_offset, inner_border
            )
            
            if show_labels:
                label = f"Row {idx}"
                # 计算文字大小
                font_scale = 0.7
                font_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                # 在框的上方绘制文字背景（带阴影）
                label_y = y1 - text_height - baseline - 8
                # 文字阴影
                cv2.rectangle(
                    result_img,
                    (x1 + shadow_offset, label_y + shadow_offset),
                    (x1 + text_width + shadow_offset, y1 + shadow_offset),
                    (0, 0, 0),
                    -1
                )
                # 文字背景
                cv2.rectangle(
                    result_img,
                    (x1, label_y),
                    (x1 + text_width + 10, y1),
                    row_color,
                    -1
                )
                # 绘制文字
                cv2.putText(
                    result_img,
                    label,
                    (x1 + 5, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness
                )
    
    # 绘制列
    for idx, col in enumerate(columns):
        bbox = col.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            _draw_enhanced_rectangle(
                result_img, x1, y1, x2, y2,
                column_color, thickness, shadow_offset, inner_border
            )
            
            if show_labels:
                label = f"Col {idx}"
                # 计算文字大小
                font_scale = 0.7
                font_thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                # 在框的左侧绘制文字背景（带阴影）
                label_x = x1 + text_width + 10
                # 文字阴影
                cv2.rectangle(
                    result_img,
                    (x1 + shadow_offset, y1 + shadow_offset),
                    (label_x + shadow_offset, y1 + text_height + baseline + 5 + shadow_offset),
                    (0, 0, 0),
                    -1
                )
                # 文字背景
                cv2.rectangle(
                    result_img,
                    (x1, y1),
                    (label_x, y1 + text_height + baseline + 8),
                    column_color,
                    -1
                )
                # 绘制文字
                cv2.putText(
                    result_img,
                    label,
                    (x1 + 5, y1 + text_height + 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness
                )
    
    return result_img


def crop_rows_and_columns(img: np.ndarray, result: Dict,
                          use_processed_img: bool = True,
                          padding: int = 0) -> Dict:
    """
    裁剪检测到的行和列
    
    Args:
        img: 原始输入图像
        result: extract_table 返回的结果字典
        use_processed_img: 是否使用预处理后的图像进行裁剪，默认 True
        padding: 裁剪时的边距（像素），默认 0
        
    Returns:
        字典，包含以下字段：
            - rows: 行图像列表，每个元素是裁剪后的行图像
            - columns: 列图像列表，每个元素是裁剪后的列图像
            - row_bboxes: 行边界框列表
            - column_bboxes: 列边界框列表
    """
    if img is None or not isinstance(img, np.ndarray) or img.size == 0:
        return {
            'rows': [],
            'columns': [],
            'row_bboxes': [],
            'column_bboxes': []
        }
    
    # 选择使用的图像（预处理后的或原始的）
    crop_img = result.get('processed_img', img) if use_processed_img else img
    
    # 确保图像是有效的
    if crop_img is None or not isinstance(crop_img, np.ndarray) or crop_img.size == 0:
        crop_img = img
    
    rows_data = result.get('rows', [])
    columns_data = result.get('columns', [])
    
    cropped_rows = []
    cropped_columns = []
    row_bboxes = []
    column_bboxes = []
    
    # 裁剪行
    for idx, row in enumerate(rows_data):
        bbox = row.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            
            # 添加边距
            h, w = crop_img.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # 确保边界有效
            if x1 < x2 and y1 < y2:
                cropped_row = crop_img[y1:y2, x1:x2]
                if cropped_row.size > 0:
                    cropped_rows.append(cropped_row)
                    row_bboxes.append([x1, y1, x2, y2])
    
    # 裁剪列
    for idx, col in enumerate(columns_data):
        bbox = col.get('bbox', [])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            
            # 添加边距
            h, w = crop_img.shape[:2]
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(w, x2 + padding)
            y2 = min(h, y2 + padding)
            
            # 确保边界有效
            if x1 < x2 and y1 < y2:
                cropped_col = crop_img[y1:y2, x1:x2]
                if cropped_col.size > 0:
                    cropped_columns.append(cropped_col)
                    column_bboxes.append([x1, y1, x2, y2])
    
    return {
        'rows': cropped_rows,
        'columns': cropped_columns,
        'row_bboxes': row_bboxes,
        'column_bboxes': column_bboxes
    }


def save_cropped_images(cropped_data: Dict, output_dir: str, prefix: str = "table") -> List[str]:
    """
    保存裁剪后的行和列图像
    
    Args:
        cropped_data: crop_rows_and_columns 返回的字典
        output_dir: 输出目录路径
        prefix: 文件名前缀，默认 "table"
        
    Returns:
        保存的文件路径列表
    """
    saved_paths = []
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 保存行图像
    rows = cropped_data.get('rows', [])
    for idx, row_img in enumerate(rows):
        if row_img.size > 0:
            filename = f"{prefix}_row_{idx:03d}.png"
            filepath = output_path / filename
            cv2.imwrite(str(filepath), row_img)
            saved_paths.append(str(filepath))
    
    # 保存列图像
    columns = cropped_data.get('columns', [])
    for idx, col_img in enumerate(columns):
        if col_img.size > 0:
            filename = f"{prefix}_column_{idx:03d}.png"
            filepath = output_path / filename
            cv2.imwrite(str(filepath), col_img)
            saved_paths.append(str(filepath))
    
    return saved_paths


# ==================== 表格扫描件预处理工具函数 ====================

def order_points(pts: np.ndarray) -> np.ndarray:
    """
    对四个点进行排序：左上、右上、右下、左下
    
    Args:
        pts: 四个点的坐标数组
        
    Returns:
        排序后的四个点
    """
    # 初始化排序后的点
    rect = np.zeros((4, 2), dtype="float32")
    
    # 计算点的和与差
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    
    # 左上角点：和最小
    rect[0] = pts[np.argmin(s)]
    # 右下角点：和最大
    rect[2] = pts[np.argmax(s)]
    # 右上角点：差最小
    rect[1] = pts[np.argmin(diff)]
    # 左下角点：差最大
    rect[3] = pts[np.argmax(diff)]
    
    return rect


def enhance_scanned_table_image(img: np.ndarray, 
                                denoise_strength: int = 3,
                                sharpen_strength: float = 1.0,
                                output_binary: bool = True) -> np.ndarray:
    """
    针对表格扫描件的专门图像增强处理（保护文字清晰度）
    
    Args:
        img: 输入图像（numpy数组）
        denoise_strength: 去噪强度（1-10），默认 3（保守设置）
        sharpen_strength: 锐化强度（0.5-2.0），默认 1.0（温和锐化）
        output_binary: 是否输出二值化黑白图像，默认 True
        
    Returns:
        增强后的图像（二值化或灰度图）
    """
    try:
        if img is None or img.size == 0:
            return img
        
        # 转换为灰度图（如果是彩色）
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            is_color = True
        else:
            gray = img.copy()
            is_color = False
        
        h, w = gray.shape
        if h < 20 or w < 20:
            return img
        
        logger.debug("✓ 步骤2: 扫描件图像增强")
        
        # 1. 去噪（保留文字边缘，保守处理）
        logger.debug("  ├─ 去噪处理（保留文字细节）...")
        # 限制去噪强度，保护文字清晰度
        effective_denoise = max(1, min(denoise_strength, 4))  # 限制在1-4之间
        denoised = cv2.fastNlMeansDenoising(gray, None, h=effective_denoise * 2, 
                                            templateWindowSize=7, searchWindowSize=21)
        
        # 2. 对比度增强（适度增强，避免过曝）
        logger.debug("  ├─ 对比度增强（CLAHE，保护文字）...")
        # 使用较低的clipLimit，避免文字过曝和细节丢失
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        # 3. 背景归一化（保守处理，避免影响文字）
        brightness_std = np.std(enhanced)
        
        # 提高阈值，只在明显不均匀时才处理
        if brightness_std > 50:
            logger.debug("  ├─ 背景归一化（保守处理）...")
            # 使用更大的核，避免影响文字
            background_kernel_size = max(min(w, h) // 15, 20)
            background_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, 
                                                          (background_kernel_size, background_kernel_size))
            background = cv2.morphologyEx(enhanced, cv2.MORPH_CLOSE, background_kernel, iterations=1)
            # 使用更温和的归一化，与原图混合保留细节
            normalized = cv2.subtract(background, enhanced)
            normalized = cv2.bitwise_not(normalized)
            # 与原图混合，保留更多细节
            enhanced = cv2.addWeighted(enhanced, 0.3, normalized, 0.7, 0)
        else:
            logger.debug("  ├─ 跳过背景归一化（亮度均匀，保护文字）")
        
        # 4. 表格线条增强（跳过，保护文字清晰度）
        logger.debug("  ├─ 跳过表格线条增强（保护文字）")
        # 不进行线条增强，直接使用enhanced
        
        # 5. 锐化处理（温和锐化，保护文字清晰度）
        if sharpen_strength > 0:
            logger.debug("  ├─ 锐化处理（增强文字边缘）...")
            # 使用温和的锐化，避免过度处理
            effective_sharpen = min(sharpen_strength, 1.0)  # 限制在1.0以内
            # 使用更小的锐化系数和更精细的高斯模糊
            gaussian = cv2.GaussianBlur(enhanced, (0, 0), 1.5)
            sharpened = cv2.addWeighted(enhanced, 1.0 + effective_sharpen * 0.2, 
                                       gaussian, -effective_sharpen * 0.2, 0)
            sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        else:
            sharpened = enhanced
        
        # 6. 二值化处理（最小化形态学操作，保护文字）
        if output_binary:
            logger.debug("  └─ 二值化处理（保护文字）")
            # 使用 Otsu 自适应阈值
            _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 最小化形态学操作，优先保护文字清晰度
            # 只对明显的长表格线条进行连接，其他区域完全不动
            binary_h, binary_w = binary.shape  # 使用不同的变量名避免混淆
            
            # 只连接非常明显的表格线条（更长的线条）
            h_kernel_long = cv2.getStructuringElement(cv2.MORPH_RECT, (max(binary_w // 15, 30), 1))
            v_kernel_long = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(binary_h // 15, 30)))
            
            # 检测长线条
            h_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, h_kernel_long, iterations=1)
            v_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, v_kernel_long, iterations=1)
            table_lines = cv2.bitwise_or(h_lines, v_lines)
            
            # 只对表格线条区域进行最小化连接
            if np.sum(table_lines) > 0:  # 如果有表格线条
                kernel_connect = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))  # 最小化连接
                table_lines_connected = cv2.morphologyEx(table_lines, cv2.MORPH_CLOSE, kernel_connect, iterations=1)
                
                # 合并：表格线条用连接后的，文字区域完全保持原样
                binary = cv2.bitwise_or(
                    cv2.bitwise_and(binary, cv2.bitwise_not(table_lines)),  # 非线条区域（文字）完全保持原样
                    table_lines_connected  # 线条区域使用连接后的
                )
            # 不进行去噪操作，避免误删文字细节
            logger.debug("    跳过去噪操作（保护文字细节）")
            
            result = binary
            logger.info("  ✓ 图像增强完成（高对比度黑白图像，文字保护模式）")
        else:
            result = sharpened
            logger.info("  ✓ 图像增强完成（灰度图）")
        
        # 转换回彩色格式（如果原图是彩色）
        if is_color and len(result.shape) == 2:
            result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
        
        return result
        
    except Exception as e:
        logger.error(f"图像增强错误: {e}", exc_info=True)
        return img


def detect_table_line_angle(img: np.ndarray) -> float:
    """
    检测表格线条的平均角度，用于精确角度校正
    
    Args:
        img: 输入图像（numpy数组）
        
    Returns:
        平均角度（度），如果检测失败返回 0
    """
    try:
        if img is None or img.size == 0:
            return 0
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        h, w = gray.shape
        if h < 20 or w < 20:
            return 0
        
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, 
                               minLineLength=min(w, h) // 4, maxLineGap=10)
        
        if lines is None or len(lines) == 0:
            return 0
        
        h_angles = []
        v_angles = []
        
        for line in lines:
            x1, y1, x2, y2 = line[0]
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            
            if abs(angle) < 15 or abs(angle) > 165:
                if angle > 165:
                    angle = angle - 180
                h_angles.append(angle)
            elif 75 < abs(angle) < 105:
                if angle > 0:
                    v_angles.append(angle - 90)
                else:
                    v_angles.append(angle + 90)
        
        all_angles = []
        if len(h_angles) >= 3:
            all_angles.extend(h_angles)
        if len(v_angles) >= 3:
            all_angles.extend(v_angles)
        
        if len(all_angles) == 0:
            return 0
        
        median_angle = np.median(all_angles)
        mad = np.median(np.abs(np.array(all_angles) - median_angle))
        
        if mad > 0:
            filtered_angles = [a for a in all_angles if abs(a - median_angle) <= 3 * mad]
        else:
            filtered_angles = all_angles
        
        if len(filtered_angles) == 0:
            return 0
        
        avg_angle = np.mean(filtered_angles)
        return avg_angle
        
    except Exception as e:
        logger.error(f"角度检测错误: {e}", exc_info=True)
        return 0


def detect_and_correct_table_by_lines(img: np.ndarray) -> np.ndarray:
    """
    基于表格线条检测最边缘的水平线和垂直线，进行透视修正和裁剪
    
    Args:
        img: 输入图像（numpy数组）
        
    Returns:
        校正并裁剪后的图像，如果失败则返回原图
    """
    try:
        if img is None or img.size == 0:
            return img
        
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()
        
        h, w = gray.shape
        
        if h < 50 or w < 50:
            return img
        
        # 增强对比度
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        
        # 二值化
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # 检测水平线和垂直线
        horizontal_kernel_size = max(w // 30, 15)
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_kernel_size, 1))
        horizontal_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        horizontal_lines = cv2.dilate(horizontal_lines, horizontal_kernel, iterations=2)
        
        vertical_kernel_size = max(h // 30, 15)
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_kernel_size))
        vertical_lines = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        vertical_lines = cv2.dilate(vertical_lines, vertical_kernel, iterations=2)
        
        # 找到最边缘的线条
        h_contours_result = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(h_contours_result) == 3:
            _, h_contours, _ = h_contours_result
        else:
            h_contours, _ = h_contours_result
        
        v_contours_result = cv2.findContours(vertical_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(v_contours_result) == 3:
            _, v_contours, _ = v_contours_result
        else:
            v_contours, _ = v_contours_result
        
        if len(h_contours) < 2 or len(v_contours) < 2:
            logger.debug(f"线条检测不足（水平: {len(h_contours)}, 垂直: {len(v_contours)}），使用原图")
            return img
        
        # 获取边缘线条坐标
        h_y_coords = []
        for contour in h_contours:
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            if w_rect > w * 0.3:
                h_y_coords.append(y + h_rect // 2)
        
        v_x_coords = []
        for contour in v_contours:
            x, y, w_rect, h_rect = cv2.boundingRect(contour)
            if h_rect > h * 0.3:
                v_x_coords.append(x + w_rect // 2)
        
        if len(h_y_coords) < 2 or len(v_x_coords) < 2:
            logger.debug(f"有效线条不足（水平: {len(h_y_coords)}, 垂直: {len(v_x_coords)}），使用原图")
            return img
        
        top_y = min(h_y_coords)
        bottom_y = max(h_y_coords)
        left_x = min(v_x_coords)
        right_x = max(v_x_coords)
        
        logger.debug(f"检测到表格边界: 上={top_y}, 下={bottom_y}, 左={left_x}, 右={right_x}")
        
        # 构建四个角点
        margin = 5
        pts = np.array([
            [left_x - margin, top_y - margin],
            [right_x + margin, top_y - margin],
            [right_x + margin, bottom_y + margin],
            [left_x - margin, bottom_y + margin]
        ], dtype="float32")
        
        pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
        pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)
        
        # 透视变换
        rect = order_points(pts)
        (tl, tr, br, bl) = rect
        
        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))
        
        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))
        
        if maxWidth < 10 or maxHeight < 10:
            logger.debug(f"计算的宽高过小 ({maxWidth}x{maxHeight})，使用原图")
            return img
        
        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]], dtype="float32")
        
        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(img, M, (maxWidth, maxHeight),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT,
                                    borderValue=(255, 255, 255) if len(img.shape) == 3 else 255)
        
        logger.debug(f"透视校正完成，校正后尺寸: {maxWidth}x{maxHeight}")
        return warped
        
    except Exception as e:
        logger.error(f"线条检测与校正错误: {e}", exc_info=True)
        return img


def preprocess_table_region(img: np.ndarray, 
                            enable_enhance: bool = True,
                            enable_perspective: bool = True, 
                            enable_angle_correction: bool = True,
                            denoise_strength: int = 3,
                            sharpen_strength: float = 1.0,
                            angle_threshold: float = 0.2,
                            output_binary: bool = True) -> np.ndarray:
    """
    表格区域预处理管道（统一入口）
    
    完整处理流程：
    步骤1: 裁剪 line 标签区域（调用方完成）
    步骤2: 扫描件图像增强
        ├─ 去噪（非局部均值去噪）
        ├─ 对比度增强（CLAHE）
        ├─ 背景归一化
        ├─ 表格线条增强
        ├─ 锐化处理
    步骤3: 精确角度校正（先校正角度，便于后续透视修正）
        ├─ 检测所有线条角度
        ├─ 过滤异常值（MAD）
        └─ 旋转校正（阈值可配置）
    步骤4: 透视修正（在角度校正后进行，检测更准确）
        ├─ 检测表格线条
        ├─ 找到边缘线条
        └─ 透视变换 + 裁剪
    步骤5: 保存最终结果（调用方完成）
    
    优化效果：
    ✅ 清晰的表格线条
    ✅ 均匀的背景
    ✅ 高对比度的黑白图像
    ✅ 完整连接的表格线
    ✅ 更精确的透视校正
    ✅ 完全水平和垂直的表格
    
    Args:
        img: 输入图像（numpy数组）
        enable_enhance: 是否启用图像增强
        enable_perspective: 是否启用透视校正
        enable_angle_correction: 是否启用角度校正
        denoise_strength: 去噪强度（1-10）
        sharpen_strength: 锐化强度（0.5-2.0）
        angle_threshold: 角度阈值（度）
        output_binary: 是否输出二值化黑白图像，默认 True
        
    Returns:
        处理后的图像，如果处理失败返回原图
    """
    try:
        if img is None or img.size == 0:
            return img
        
        result = img.copy()
        
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        logger.info("🚀 开始表格预处理管道")
        
        # 步骤2: 图像增强
        if enable_enhance:
            result = enhance_scanned_table_image(result, denoise_strength, sharpen_strength, output_binary)
            if result.size == 0:
                logger.warning("图像增强失败，使用原图")
                return img
        else:
            logger.debug("✗ 步骤2: 跳过图像增强")
        
        # 步骤3: 精确角度校正（先校正角度，便于后续透视修正）
        if enable_angle_correction:
            logger.debug("✓ 步骤3: 精确角度校正")
            angle = detect_table_line_angle(result)
            if abs(angle) > angle_threshold:
                result = _rotate_image(result, -angle)
                logger.info(f"  ✓ 角度校正: {angle:.3f}° -> 0°")
            else:
                logger.debug(f"  - 角度偏差: {angle:.3f}°（无需校正）")
            
            if result.size == 0:
                logger.warning("角度校正失败，使用原图")
                return img
        else:
            logger.debug("✗ 步骤3: 跳过角度校正")
        
        # 步骤4: 透视校正（在角度校正后进行，检测更准确）
        if enable_perspective:
            logger.debug("✓ 步骤4: 透视修正")
            original_shape = result.shape[:2]
            result = detect_and_correct_table_by_lines(result)
            if result.shape[:2] != original_shape:
                logger.info(f"  ✓ 透视校正完成: {original_shape} -> {result.shape[:2]}")
            else:
                logger.debug("  - 无需透视校正")
            if result.size == 0:
                logger.warning("透视校正失败，使用原图")
                return img
        else:
            logger.debug("✗ 步骤4: 跳过透视校正")
        
        logger.info("✅ 预处理管道完成！")
        logger.info("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
        return result
        
    except Exception as e:
        logger.error(f"❌ 预处理管道错误: {e}", exc_info=True)
        return img


def crop_line_region(img: np.ndarray, xyxy: np.ndarray, padding_size: int = 3) -> Optional[np.ndarray]:
    """
    从图像中裁剪 line 区域（带 padding）
    
    Args:
        img: 输入图像
        xyxy: 边界框坐标 [x1, y1, x2, y2]
        padding_size: 边缘容错像素数
        
    Returns:
        裁剪后的区域，如果失败返回 None
    """
    try:
        if img is None or img.size == 0:
            return None
        
        x1, y1, x2, y2 = map(int, xyxy)
        img_height, img_width = img.shape[:2]
        
        # 添加 padding 并确保不超出边界
        x1_padded = max(0, x1 - padding_size)
        y1_padded = max(0, y1 - padding_size)
        x2_padded = min(img_width, x2 + padding_size)
        y2_padded = min(img_height, y2 + padding_size)
        
        # 裁剪
        cropped = img[y1_padded:y2_padded, x1_padded:x2_padded]
        
        if cropped.size == 0:
            return None
        
        return cropped
        
    except Exception as e:
        logger.error(f"裁剪错误: {e}", exc_info=True)
        return None


def save_processed_image(img: np.ndarray, output_path: str, filename: str) -> bool:
    """
    保存处理后的图像
    
    Args:
        img: 要保存的图像
        output_path: 输出目录
        filename: 文件名
        
    Returns:
        成功返回 True，失败返回 False
    """
    try:
        if img is None or img.size == 0:
            logger.warning("图像无效，跳过保存")
            return False
        
        filepath = os.path.join(output_path, filename)
        cv2.imwrite(filepath, img)
        
        h, w = img.shape[:2]
        logger.info(f"保存: {filename} (尺寸: {w}x{h})")
        return True
        
    except Exception as e:
        logger.error(f"保存错误: {e}", exc_info=True)
        return False

