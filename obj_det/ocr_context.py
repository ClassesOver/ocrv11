from obj_det.vat_detect import invoice_detection as vat
from obj_det.vat_detect_v2 import invoice_detection as vat_v2
from obj_det.stock_detect_v2 import stock_detection_v2 as stock_v2
from obj_det.stock_detect import stock_detection as stock_v1
from obj_det.bill_detect import bill_detection as bill
from obj_det.table.table_transformers_extract import extract_table
from settings import ocr_predict
from paddleocr import TextRecognition as _TextRecognition, TextDetection as _TextDetection
from loguru import logger
from typing import List, Optional
import cv2
import numpy as np
import config
import platform
import os

class TextRecognition(_TextRecognition):

    def _get_extra_paddlex_predictor_init_args(self):
        res = super()._get_extra_paddlex_predictor_init_args()
        # 获取 CPU 核心数用于 HPI 配置
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
        except:
            cpu_count = 4  # 默认值
        
        res.update({
            'hpi_config': {
                'backend': 'onnxruntime',
                'backend_config': {
                    'cpu_num_threads': max(10, min(cpu_count, 16))
                }
            }
        })
        return res

class TextDetection(_TextDetection):

    def _get_extra_paddlex_predictor_init_args(self):
        res = super()._get_extra_paddlex_predictor_init_args()
        # 获取 CPU 核心数用于 HPI 配置
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
        except:
            cpu_count = 4  # 默认值

        res.update({
            'hpi_config': {
                'backend': 'onnxruntime',
                'backend_config': {
                    'cpu_num_threads': max(10, min(cpu_count, 16))
                }
            }
        })
        return res


class TextOcrModel(object):
    def __init__(self):
        self._chinese_ocr_instance = ocr_predict
        
        # 优化配置：启用 GPU 和性能优化参数
        use_gpu = config.GPU if hasattr(config, 'GPU') else False
        
        # 根据操作系统决定是否启用 HPI（High Performance Inference）
        is_linux = platform.system().lower() == 'linux'

        # PaddleOCR 识别模型配置：支持通过配置/环境变量切换
        model_name = getattr(config, "PADDLE_REC_MODEL_NAME", "PP-OCRv4_mobile_rec")
        
        # HPI 配置：通过环境变量控制（默认启用以获得更好性能）
        enable_hpi_env = os.getenv("PADDLE_ENABLE_HPI", "").strip().lower()
        enable_hpi = is_linux and enable_hpi_env not in ["0", "false", "no"]

        # 统一获取 CPU 核心数（避免重复调用）
        try:
            import multiprocessing
            cpu_count = multiprocessing.cpu_count()
        except:
            cpu_count = 4  # 默认值，用于后续计算
            logger.warning("无法获取 CPU 核心数，使用默认值 4")

        # 批处理配置：批量大小（可在 config 中覆盖）
        # CPU 环境推荐配置：
        #   - 根据 CPU 核心数动态调整：batch_size = cpu_count * 2
        #   - 但需要设置合理范围：最小 4，最大 32（避免内存溢出）
        # GPU 环境推荐配置：batch_size = 16-32
        # 注意：batch_size 过大会导致内存占用增加，需要根据实际内存情况调整
        if hasattr(config, "OCR_BATCH_SIZE") and config.OCR_BATCH_SIZE is not None:
            # 使用手动配置的值
            self._batch_size = config.OCR_BATCH_SIZE
            logger.info(f"使用手动配置的 batch_size={self._batch_size}")
        else:
            # 根据设备类型设置默认 batch_size
            if use_gpu:
                self._batch_size = 16  # GPU 默认值
            else:
                # CPU 环境：根据 CPU 核心数动态调整
                # batch_size = cpu_count * 2，但限制在合理范围内
                self._batch_size = max(4, min(cpu_count * 2, 32))
                logger.info(f"根据 CPU 核心数({cpu_count})自动设置 batch_size={self._batch_size}")
        
        # 验证 batch_size 合理性（无论手动配置还是自动配置都需要验证）
        original_batch_size = self._batch_size
        if self._batch_size < 1:
            logger.warning(f"batch_size={self._batch_size} 过小，调整为 4")
            self._batch_size = 4
        elif self._batch_size > 64:
            logger.warning(f"batch_size={self._batch_size} 过大，调整为 32（建议范围：4-32）")
            self._batch_size = 32
        elif original_batch_size != self._batch_size:
            logger.info(f"batch_size 已从 {original_batch_size} 调整为 {self._batch_size}")
        
        # MKLDNN 配置：Linux 平台下通过环境变量控制（默认：CPU 模式下启用）
        enable_mkldnn_env = os.getenv("PADDLE_ENABLE_MKLDNN", "").strip().lower()
        if enable_mkldnn_env:
            # 如果设置了环境变量，使用环境变量的值
            enable_mkldnn = is_linux and enable_mkldnn_env in ["1", "true", "yes"]
        else:
            # 默认：CPU 模式下启用，GPU 模式下禁用
            enable_mkldnn = not use_gpu and is_linux

        # CPU 线程数配置优化：
        # CPU 环境：推荐设置为 CPU 核心数，但不超过 batch_size * 2，最小为 4
        # GPU 环境：使用较少线程（8），避免占用过多 CPU 资源
        if use_gpu:
            cpu_threads = 8  # GPU 模式下使用较少线程
        else:
            # CPU 线程数策略：
            # 1. 优先使用 CPU 核心数（充分利用多核）
            # 2. 但不超过 batch_size * 2（避免过度并行导致上下文切换开销）
            # 3. 最小为 4（保证基本性能）
            # 4. 最大为 32（避免过多线程导致性能下降）
            cpu_threads = max(4, min(cpu_count, self._batch_size * 2, 32))
            logger.info(f"根据 CPU 核心数({cpu_count})和 batch_size({self._batch_size})设置 cpu_threads={cpu_threads}")

        self._paddle_ocr_instance = TextRecognition(
            model_name=model_name,
            device='gpu' if use_gpu else 'cpu',
            enable_mkldnn=enable_mkldnn,
            enable_hpi=enable_hpi,
            cpu_threads=cpu_threads,
        )
        logger.info(
            f"PaddleOCR 初始化成功: model={model_name}, device={'gpu' if use_gpu else 'cpu'}, batch_size={self._batch_size}, cpu_threads={cpu_threads}, hpi={enable_hpi}, mkldnn={enable_mkldnn}")

        # PP-OCRv4 实例按需加载，默认禁用
        # 可通过 config.ENABLE_PADDLE_OCR_V4 = True 启用
        self._paddle_ocr_v4_instance = None
        self._paddle_ocr_v4_enabled = getattr(config, "ENABLE_PADDLE_OCR_V4", False)
        self._paddle_ocr_v4_config = {
            'model_name': 'PP-OCRv4_mobile_rec',
            'device': 'gpu' if use_gpu else 'cpu',
            'enable_mkldnn': enable_mkldnn,
            'enable_hpi': enable_hpi,
            'cpu_threads': cpu_threads
        }
        
        # 预热：避免首次调用的长延迟（不影响后续性能）
        try:
            warmup_img = np.zeros((32, 32, 3), dtype=np.uint8)
            _ = self._paddle_ocr_instance.predict(warmup_img)
        except Exception as e:
            logger.debug(f"PaddleOCR 预热失败: {e}")

        # 初始化 TextDetection（用于单元格文本检测和裁剪）
        # 是否启用文本检测裁剪（可通过 config 控制）
        enable_text_detection_crop = getattr(config, "ENABLE_TEXT_DETECTION_CROP", True)
        self._text_detector = None
        if enable_text_detection_crop:
            try:
                self._text_detector = TextDetection(device='gpu' if use_gpu else 'cpu',
                                                    enable_mkldnn=enable_mkldnn,
                                                    enable_hpi=enable_hpi,
                                                    cpu_threads=cpu_threads, )
                logger.info(f"TextDetection 初始化成功: device={'gpu' if use_gpu else 'cpu'}")
            except Exception as e:
                logger.warning(f"TextDetection 初始化失败: {e}，将不使用文本检测裁剪")
                self._text_detector = None

        self.ocr = self._ocr
        self.vat = vat
        self.vat_v2 = vat_v2
        self.stock_v1 = stock_v1
        self.stock_v2 = stock_v2
        # 默认入库单检测使用新版（药品），但仍保留别名以兼容调用
        self.stock = self.stock_v2
        self.bill = bill
        
        # 性能优化：图像预处理参数（可在 config 中覆盖）
        self._max_img_size = getattr(config, "OCR_MAX_IMG", 960)  # 最大尺寸
        self._min_img_size = getattr(config, "OCR_MIN_IMG", 32)   # 最小尺寸（过小不缩放）

    def _get_paddle_ocr_v4_instance(self):
        """
        按需加载 PP-OCRv4 实例
        
        Returns:
            TextRecognition 实例，如果禁用或加载失败则返回 None
        """
        # 如果已加载，直接返回
        if self._paddle_ocr_v4_instance is not None:
            return self._paddle_ocr_v4_instance
        
        # 如果未启用，返回 None
        if not self._paddle_ocr_v4_enabled:
            return None
        
        # 按需加载
        try:
            logger.info("正在按需加载 PP-OCRv4 实例...")
            self._paddle_ocr_v4_instance = TextRecognition(**self._paddle_ocr_v4_config)
            logger.info(f"PP-OCRv4 实例加载成功: device={self._paddle_ocr_v4_config['device']}")
            return self._paddle_ocr_v4_instance
        except Exception as e:
            logger.warning(f"PP-OCRv4 实例加载失败: {e}，将不使用 v4 模型")
            self._paddle_ocr_v4_enabled = False  # 禁用，避免重复尝试
            return None



    def _ocr(self, img, use_paddle_first=True, fallback=True):
        """
        混合使用 PaddleOCR 和 ChineseOCR
        
        Args:
            img: 输入图像
            use_paddle_first: 是否优先使用 PaddleOCR (默认 True)
            fallback: 如果第一个模型失败，是否回退到另一个模型 (默认 True)
            
        Returns:
            识别的文本字符串
        """
        try:
            if use_paddle_first:
                # 先尝试 PaddleOCR
                paddle_result = self._get_paddle_text(img)
                if paddle_result and paddle_result.strip():
                    return paddle_result

                # 如果 PaddleOCR 结果为空且允许回退，使用 ChineseOCR
                if fallback:
                    logger.debug("PaddleOCR 结果为空，回退到 ChineseOCR")
                    return self._chinese_ocr_instance(img)
                return paddle_result
            else:
                # 先尝试 ChineseOCR
                chinese_result = self._chinese_ocr_instance(img)
                if chinese_result and chinese_result.strip():
                    return chinese_result

                # 如果 ChineseOCR 结果为空且允许回退，使用 PaddleOCR
                if fallback:
                    logger.debug("ChineseOCR 结果为空，回退到 PaddleOCR")
                    return self._get_paddle_text(img)
                return chinese_result

        except Exception as e:
            logger.error(f"混合 OCR 识别错误: {e}")
            # 出错时尝试使用备用模型
            if fallback:
                try:
                    if use_paddle_first:
                        return self._chinese_ocr_instance(img)
                    else:
                        return self._get_paddle_text(img)
                except:
                    return ""
            return ""

    def _preprocess_image(self, img):
        """
        图像预处理，优化识别速度
        
        Args:
            img: 输入图像
            
        Returns:
            预处理后的图像
        """
        if img is None:
            return None
        
        # 检查是否为空数组
        if not isinstance(img, np.ndarray) or img.size == 0:
            return None
        
        h, w = img.shape[:2]
        
        # 如果图像过小，不处理，避免过度放大
        if h < self._min_img_size or w < self._min_img_size:
            return img

        # 如果图像过大，缩放以提升速度（默认限制 960，可通过 config 调整）
        max_dim = max(h, w)
        if max_dim > self._max_img_size:
            scale = self._max_img_size / max_dim
            new_h, new_w = int(h * scale), int(w * scale)
            # 使用 INTER_AREA 进行缩小，速度快且质量好
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        
        # 保证内存连续，提高底层推理效率
        if not img.flags['C_CONTIGUOUS']:
            img = np.ascontiguousarray(img)
        
        return img
    
    def _get_paddle_text(self, img):
        """
        PaddleOCR 辅助函数，提取文本内容（加速优化版）
        
        优化项：
        1. 图像预处理（缩放大图）
        2. 移除冗余日志
        3. 快速返回
        4. 简化异常处理
        """
        try:
            # 快速空值检查
            if img is None:
                return ""
            
            if not isinstance(img, np.ndarray) or img.size == 0:
                return ""
            
            # 图像预处理（加速）
            img = self._preprocess_image(img)
            if img is None:
                return ""
            
            # 执行识别
            result = self._paddle_ocr_instance.predict(img)
            
            # 快速返回结果（无冗余日志）
            if result and len(result) > 0 and 'rec_text' in result[0]:
                return result[0]['rec_text']
            
            return ""
            
        except Exception as e:
            return ""
    
    def _get_paddle_text_batch(self, images, preprocess=True, v4=False):
        """
        PaddleOCR 批量识别辅助函数（性能优化版）
        
        Args:
            images: 图像列表
            
        Returns:
            识别结果列表
            
        优化项：
        1. 使用 PaddleOCR 原生批处理能力
        2. 列表推导式优化
        3. 减少函数调用和检查
        4. 优化内存使用
        """
        # 快速空值检查
        if not images:
            return []
        
        try:
            # 批量预处理图像 - 使用列表推导式
            def _preprocess_image(_img):
                if preprocess:
                    return self._preprocess_image(_img)
                else:
                    return _img
            processed_data = [
                (idx, _preprocess_image(img))
                for idx, img in enumerate(images)
                if img is not None and isinstance(img, np.ndarray) and img.size > 0
            ]
            
            # 如果没有有效图像，快速返回
            if not processed_data:
                return [""] * len(images)
            
            # 分离索引和处理后的图像
            valid_indices, processed_images = zip(*[
                (idx, img) for idx, img in processed_data if img is not None
            ])
            
            # 批量执行识别（PaddleOCR 原生支持）
            if v4:
                v4_instance = self._get_paddle_ocr_v4_instance()
                if v4_instance is None:
                    # 如果 v4 实例不可用，回退到标准实例
                    logger.debug("PP-OCRv4 实例不可用，回退到标准 PaddleOCR 实例")
                    batch_results = self._paddle_ocr_instance.predict(list(processed_images),
                                                                      batch_size=self._batch_size)
                else:
                    batch_results = v4_instance.predict(list(processed_images),
                                                        batch_size=self._batch_size)
            else:
                batch_results = self._paddle_ocr_instance.predict(list(processed_images),
                                                                  batch_size=self._batch_size)
            
            # 构建结果列表 - 使用字典映射优化查找
            result_map = {
                idx: batch_results[i].get('rec_text', '')
                for i, idx in enumerate(valid_indices)
                if i < len(batch_results)
            }
            
            # 返回完整结果列表
            return [result_map.get(i, '') for i in range(len(images))]
            
        except Exception as e:
            return [""] * len(images)
    
    def batch_ocr(self, images, use_paddle_first=True, preprocess=True, v4=False):
        """
        批量 OCR 识别
        
        Args:
            images: 图像列表
            use_paddle_first: 是否优先使用 PaddleOCR (默认 True)
            preprocess:
            
        Returns:
            识别结果列表
            
        优化说明：
        - PaddleOCR: 使用原生批处理，显著提升吞吐量
        - ChineseOCR: 逐个处理图像
        """
        if not images:
            return []
        
        try:
            if use_paddle_first:
                return self._get_paddle_text_batch(images, preprocess=preprocess, v4=v4)
            else:
                # 使用 ChineseOCR（逐个处理）
                results = []
                for img in images:
                    def _preprocess_image(_img):
                        if preprocess:
                            return self._preprocess_image(_img)
                        else:
                            return _img
                    img = _preprocess_image(img)
                    try:
                        results.append(self._chinese_ocr_instance(img) if img is not None else "")
                    except:
                        results.append("")
                return results
                
        except Exception as e:
            logger.error(f"批量 OCR 识别错误: {e}")
            return [""] * len(images)
    
    def table_recognize(self, img, selected_columns: Optional[List[int]] = None):
        """
        表格识别推理（使用 Table Transformer）
        
        Args:
            img: 输入图像（numpy数组）
            selected_columns: 指定要获取的列索引列表（从0开始），如果为None则返回所有列，例如 [0, 2, 3] 表示只获取第0、2、3列
            
        Returns:
            字典，包含以下字段：
                - rows: 行列表，每行包含该行的单元格（如果指定了selected_columns，则只包含指定列的单元格）
                - columns: 列列表，每列包含该列的单元格
                - cells: 所有单元格列表
                - structure: 表格结构矩阵
        """
        try:
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                logger.warning("输入图像无效")
                return {}
            
            # 使用 Table Transformer 提取表格结构
            table_result = extract_table(
                img,
                enable_angle_correction=True,
                angle_threshold=0.1,
                enable_enhance=False,
                enable_perspective=True,
                structure_threshold=0.6,
                auto_adjust_threshold=True,
                selected_columns=selected_columns
            )
            
            return table_result
            
        except Exception as e:
            logger.error(f"表格识别错误: {e}", exc_info=True)
            return {}
    
    def _convert_rows_to_html(self, rows):
        """
        将行数据转换为HTML表格格式
        
        Args:
            rows: 行数据列表
            
        Returns:
            HTML字符串
        """
        try:
            if not rows:
                return ""
            
            html_parts = ['<table>']
            for row in rows:
                html_parts.append('<tr>')
                for cell in row:
                    text = cell.get('text', '') if isinstance(cell, dict) else str(cell)
                    html_parts.append(f'<td>{text}</td>')
                html_parts.append('</tr>')
            html_parts.append('</table>')
            
            return ''.join(html_parts)
            
        except Exception as e:
            logger.error(f"转换行数据为HTML错误: {e}", exc_info=True)
            return ""
    
    def _extract_rows_from_table_ocr_pred(self, table_ocr_pred):
        """
        从table_ocr_pred中提取行数据
        
        Args:
            table_ocr_pred: 表格OCR预测结果，包含cells、rec_texts、bbox等信息
            
        Returns:
            行数据列表，每行是一个字典列表，包含该行的所有单元格信息
        """
        try:
            rows = []
            
            # 方法1: 如果有cells信息，直接使用
            if 'cells' in table_ocr_pred and table_ocr_pred['cells']:
                cells = table_ocr_pred['cells']
                # 按行索引分组
                row_dict = {}
                for cell in cells:
                    row_idx = cell.get('row', -1)
                    if row_idx not in row_dict:
                        row_dict[row_idx] = []
                    row_dict[row_idx].append(cell)
                
                # 按行索引排序，每行内按列索引排序
                for row_idx in sorted(row_dict.keys()):
                    row_cells = sorted(row_dict[row_idx], key=lambda x: x.get('col', 0))
                    rows.append(row_cells)
                return rows
            
            # 方法2: 如果有rec_texts和bbox，根据bbox的y坐标分组为行
            if 'rec_texts' in table_ocr_pred and 'bbox' in table_ocr_pred:
                rec_texts = table_ocr_pred['rec_texts']
                bboxes = table_ocr_pred['bbox']
                
                if not rec_texts or not bboxes or len(rec_texts) != len(bboxes):
                    return []
                
                # 创建单元格列表，包含文本和bbox
                cells = []
                for i, (text, bbox) in enumerate(zip(rec_texts, bboxes)):
                    # bbox格式可能是 [x1, y1, x2, y2] 或 [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
                    if isinstance(bbox, list) and len(bbox) > 0:
                        if isinstance(bbox[0], (list, tuple)) and len(bbox[0]) >= 2:
                            # 多边形格式，取最小y和最大y
                            y_coords = [point[1] for point in bbox if len(point) >= 2]
                            y_center = (min(y_coords) + max(y_coords)) / 2 if y_coords else 0
                        else:
                            # 矩形格式 [x1, y1, x2, y2]
                            if len(bbox) >= 4:
                                y_center = (bbox[1] + bbox[3]) / 2
                            else:
                                y_center = 0
                    else:
                        y_center = 0
                    
                    cells.append({
                        'text': text,
                        'bbox': bbox,
                        'y_center': y_center,
                        'index': i
                    })
                
                # 按y_center分组为行（允许一定误差）
                if not cells:
                    return []
                
                # 对y_center进行聚类分组
                cells_sorted = sorted(cells, key=lambda x: x['y_center'])
                rows_grouped = []
                current_row = [cells_sorted[0]]
                y_threshold = 10  # y坐标差异阈值，用于判断是否同一行
                
                for i in range(1, len(cells_sorted)):
                    cell = cells_sorted[i]
                    prev_cell = cells_sorted[i-1]
                    # 如果y坐标差异小于阈值，认为是同一行
                    if abs(cell['y_center'] - prev_cell['y_center']) < y_threshold:
                        current_row.append(cell)
                    else:
                        # 新行，先对当前行按x坐标排序
                        current_row_sorted = sorted(current_row, key=lambda x: self._get_bbox_x_center(x.get('bbox', [])))
                        rows_grouped.append(current_row_sorted)
                        current_row = [cell]
                
                # 处理最后一行
                if current_row:
                    current_row_sorted = sorted(current_row, key=lambda x: self._get_bbox_x_center(x.get('bbox', [])))
                    rows_grouped.append(current_row_sorted)
                
                return rows_grouped
            
            # 方法3: 如果只有rec_texts，每行一个文本（简单情况）
            if 'rec_texts' in table_ocr_pred:
                rec_texts = table_ocr_pred['rec_texts']
                if rec_texts:
                    return [[{'text': text} for text in rec_texts]]
            
            return []
            
        except Exception as e:
            logger.error(f"提取表格行数据错误: {e}", exc_info=True)
            return []
    
    def _get_bbox_x_center(self, bbox):
        """获取bbox的x中心坐标"""
        try:
            if isinstance(bbox, list) and len(bbox) > 0:
                if isinstance(bbox[0], (list, tuple)) and len(bbox[0]) >= 1:
                    # 多边形格式，取最小x和最大x
                    x_coords = [point[0] for point in bbox if len(point) >= 1]
                    return (min(x_coords) + max(x_coords)) / 2 if x_coords else 0
                else:
                    # 矩形格式 [x1, y1, x2, y2]
                    if len(bbox) >= 4:
                        return (bbox[0] + bbox[2]) / 2
            return 0
        except:
            return 0
    
    def _crop_text_regions(self, img, dt_boxes, padding=5):
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

    def _detect_and_crop_cell_text(self, cell_img):
        """
        对单元格图像进行文本检测并裁剪
        
        Args:
            cell_img: 单元格图像
            
        Returns:
            裁剪后的图像
        """
        if self._text_detector is None or cell_img is None or cell_img.size == 0:
            return cell_img
        
        try:
            # 使用 TextDetection 检测文本
            result = self._text_detector.predict(cell_img)
            
            # 提取边界框
            dt_boxes = []
            if result is not None:
                # 处理不同的返回格式
                if isinstance(result, list):
                    for item in result:
                        if isinstance(item, dict):
                            for key in ['dt_boxes', 'dt_polys', 'points', 'polys']:
                                if key in item:
                                    boxes = item[key]
                                    if isinstance(boxes, (list, np.ndarray)):
                                        if len(boxes) > 0:
                                            if isinstance(boxes[0], (list, np.ndarray)):
                                                dt_boxes.extend(boxes)
                                            else:
                                                dt_boxes.append(boxes)
                                    break
                        elif isinstance(item, (list, np.ndarray)):
                            if len(item) > 0:
                                dt_boxes.append(item)
                elif isinstance(result, dict):
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
                    if len(result.shape) >= 2:
                        dt_boxes = result.tolist() if len(result.shape) == 3 else [result.tolist()]
            
            if dt_boxes and len(dt_boxes) > 0:
                # 裁剪文本区域
                cropped_img = self._crop_text_regions(cell_img, dt_boxes, padding=5)
                return cropped_img
            else:
                return cell_img
                
        except Exception as e:
            logger.debug(f"单元格文本检测失败: {e}，使用原图")
            return cell_img
    
    def ocr_table_cells(self, img, selected_columns: Optional[List[int]] = None, saveImage=False):
        """
        表格识别并批量OCR识别单元格（封装table_recognize）
        
        Args:
            img: 输入图像（numpy数组）
            selected_columns: 指定要获取的列索引列表（从0开始），如果为None则返回所有列，例如 [0, 2, 3] 表示只获取第0、2、3列
            saveImage: 是否保存单元格图像，默认为 False
            
        Returns:
            二维列表，格式为 [[row1_cell1, row1_cell2, ...], [row2_cell1, row2_cell2, ...], ...]
            用于 stock_detect 的 line 结果返回
            如果指定了 selected_columns，则返回的列表中只包含指定列的数据，列索引保持原始位置
        """
        try:
            if img is None or not isinstance(img, np.ndarray) or img.size == 0:
                logger.warning("输入图像无效")
                return []
            
            # 先进行表格识别
            table_result = self.table_recognize(img, selected_columns=selected_columns)
            
            if not isinstance(table_result, dict):
                return []
            
            rows = table_result.get('rows', [])
            if not rows:
                return []
            
            # 使用预处理后的图像（如果有）进行OCR，否则使用原图
            ocr_img = table_result.get('processed_img', img)
            if ocr_img is None or ocr_img.size == 0:
                ocr_img = img
            
            h, w = ocr_img.shape[:2]
            
            # 如果启用保存图片，创建保存目录
            if saveImage:
                cells_fp = os.path.join('images', 'table_cells')
                os.makedirs(cells_fp, exist_ok=True)
            
            # 收集所有单元格图像用于批量OCR
            cell_images = []
            cell_positions = []  # 记录每个单元格的行列位置
            
            for row_idx, row in enumerate(rows):
                if isinstance(row, dict) and 'cells' in row:
                    for col_idx, cell in enumerate(row['cells']):
                        if isinstance(cell, dict) and 'bbox' in cell:
                            bbox = cell['bbox']
                            if len(bbox) == 4:
                                x1, y1, x2, y2 = map(int, bbox)
                                # 确保坐标在图像范围内
                                x1 = max(0, min(x1, w))
                                y1 = max(0, min(y1, h))
                                x2 = max(0, min(x2, w))
                                y2 = max(0, min(y2, h))
                                
                                if x1 < x2 and y1 < y2:
                                    cell_img = ocr_img[y1:y2, x1:x2]
                                    if cell_img.size > 0:
                                        # 对单元格图像进行文本检测并裁剪
                                        cell_img = self._detect_and_crop_cell_text(cell_img)
                                        
                                        cell_images.append(cell_img)
                                        cell_positions.append((row_idx, col_idx))
                                        
                                        # 保存单元格图像（保存裁剪后的图像）
                                        if saveImage:
                                            cell_path = os.path.join(cells_fp, f'row_{row_idx}_col_{col_idx}.png')
                                            cv2.imwrite(cell_path, cell_img)
            
            # 批量OCR识别
            if not cell_images:
                return []

            # 使用批量OCR提高效率
            ocr_texts = self.batch_ocr(cell_images, use_paddle_first=True, preprocess=False, v4=False)
            
            # 按行列组织结果
            # 先找到最大行列数
            max_row = max(pos[0] for pos in cell_positions) if cell_positions else -1
            max_col = max(pos[1] for pos in cell_positions) if cell_positions else -1
            
            # 初始化二维列表
            row_texts = [[''] * (max_col + 1) for _ in range(max_row + 1)]
            
            # 填充OCR结果
            for (row_idx, col_idx), text in zip(cell_positions, ocr_texts):
                if 0 <= row_idx <= max_row and 0 <= col_idx <= max_col:
                    row_texts[row_idx][col_idx] = text if text else ''
            
            return row_texts
            
        except Exception as e:
            logger.error(f"表格单元格OCR识别错误: {e}", exc_info=True)
            return []
    
    def batch_table_recognize(self, images):
        """
        批量表格识别推理

        Args:
            images: 图像列表

        Returns:
            识别结果列表（字典格式）
        """
        if not images:
            return []

        results = []
        for idx, img in enumerate(images):
            try:
                logger.debug(f"处理第 {idx+1}/{len(images)} 张表格图像")
                result = self.table_recognize(img)
                results.append(result)
            except Exception as e:
                logger.error(f"批量表格识别第 {idx} 张图像失败: {e}")
                results.append({})
        
        return results


context = TextOcrModel()
