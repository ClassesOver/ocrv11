from ultralytics import YOLO
import cv2
import os
from loguru import logger
from obj_det.table.table_common import (
    preprocess_table_region,
    crop_line_region,
    save_processed_image
)

# 配置路径
images_dir = r"E:\stock\traindata\train\images"
output_dir = r"E:\stock\traindata\train\tables"
model_path = r"/models/stock_1/best.pt"

# 边缘容错（padding）像素数，在裁剪时增加边缘区域
padding = 3

# 角度检测和校正配置
detect_angle = True  # 是否检测并校正表格倾斜角度
angle_threshold = 0.1  # 角度阈值（度），超过此值才进行校正，降低阈值使表格更精确

# 透视校正配置
correct_perspective = True  # 是否进行透视校正

# 扫描件图像增强配置
enhance_scanned_image = True  # 是否对扫描件进行增强处理
denoise_strength = 3  # 去噪强度（1-10），针对扫描件噪点
sharpen_strength = 1 # 锐化强度（0.5-2.0），增强表格线条清晰度
output_binary = True  # 是否输出高对比度黑白图像（True=黑白，False=灰度）

# 支持的图像格式
image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp'}
# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 加载模型
logger.info(f"正在加载模型: {model_path}")
model = YOLO(model_path)
names = model.names
logger.info(f"模型类别: {names}")

# 获取所有图像文件
image_files = []
if os.path.exists(images_dir):
    for file in os.listdir(images_dir):
        file_path = os.path.join(images_dir, file)
        if os.path.isfile(file_path):
            ext = os.path.splitext(file)[1].lower()
            if ext in image_extensions:
                image_files.append(file_path)
else:
    logger.error(f"目录不存在: {images_dir}")
    exit(1)

logger.info(f"找到 {len(image_files)} 张图像")

# 统计信息
total_processed = 0
line_images_count = 0
total_cropped_regions = 0
error_count = 0

# 遍历每张图像进行预测
for idx, image_path in enumerate(image_files, 1):
    try:
        # 读取图像
        im0 = cv2.imread(image_path)
        if im0 is None:
            logger.error(f"[{idx}/{len(image_files)}] 无法读取图像: {image_path}")
            total_processed += 1
            error_count += 1
            continue
        
        # 进行预测
        results = model.predict(
            source=im0,
            imgsz=640,
            device='cpu',
            verbose=False,
            half=False,
        )
        
        # 检查是否有 "line" 标签
        has_line = False
        line_boxes = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.boxes
            
            if boxes is not None and len(boxes) > 0:
                for box in boxes:
                    cls = int(box.cls[0].item())
                    label = names[cls]
                    if label == "line":
                        has_line = True
                        xyxy = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].item())
                        line_boxes.append((xyxy, conf))
        
        # 如果有 line 标签，处理每个 line 区域
        if has_line:
            filename = os.path.basename(image_path)
            name_without_ext = os.path.splitext(filename)[0]
            ext = os.path.splitext(filename)[1]
            
            saved_count = 0
            for line_idx, (xyxy, conf) in enumerate(line_boxes):
                logger.info(f"\n━━━ 处理 line {line_idx + 1}/{len(line_boxes)} (置信度: {conf:.2f}) ━━━")
                
                # 步骤1: 裁剪 line 区域
                logger.debug("✓ 步骤1: 裁剪 line 标签区域")
                line_region = crop_line_region(im0, xyxy, padding)
                if line_region is None:
                    logger.warning("line 区域裁剪失败，跳过")
                    continue
                
                # 步骤2-4: 预处理管道（增强、透视校正、角度校正）
                processed_region = preprocess_table_region(
                    line_region,
                    enable_enhance=enhance_scanned_image,
                    enable_perspective=correct_perspective,
                    enable_angle_correction=detect_angle,
                    denoise_strength=denoise_strength,
                    sharpen_strength=sharpen_strength,
                    angle_threshold=angle_threshold,
                    output_binary=output_binary
                )
                
                if processed_region is None or processed_region.size == 0:
                    logger.warning("预处理失败，跳过")
                    continue
                
                # 步骤5: 保存结果
                logger.debug("✓ 步骤5: 保存最终结果")
                cropped_filename = f"{name_without_ext}_line_{line_idx + 1}_conf{conf:.2f}{ext}"
                if save_processed_image(processed_region, output_dir, cropped_filename):
                    saved_count += 1
            
            line_images_count += 1
            total_cropped_regions += saved_count
            logger.info(f"[{idx}/{len(image_files)}] ✓ {filename} - 检测到 {len(line_boxes)} 个 line，已保存 {saved_count} 个裁剪区域")
        else:
            filename = os.path.basename(image_path)
            logger.debug(f"[{idx}/{len(image_files)}] - {filename} - 未检测到 line")
        
        total_processed += 1
        
    except Exception as e:
        filename = os.path.basename(image_path) if image_path else "unknown"
        logger.error(f"[{idx}/{len(image_files)}] ✗ 处理 {filename} 时出错: {str(e)}", exc_info=True)
        total_processed += 1
        error_count += 1
        continue

logger.info("\n" + "="*50)
logger.info("🎉 处理完成!")
logger.info("="*50)
logger.info(f"📊 统计信息:")
logger.info(f"  - 总共处理: {total_processed} 张图像")
logger.info(f"  - 检测到 line: {line_images_count} 张")
logger.info(f"  - 保存区域: {total_cropped_regions} 个")
if error_count > 0:
    logger.warning(f"  - 处理错误: {error_count} 张图像")
logger.info(f"\n⚙️  处理配置:")
logger.info(f"  - 边缘容错 (padding): {padding} 像素")
logger.info(f"  - 扫描件增强: {'✓ 已启用' if enhance_scanned_image else '✗ 已禁用'}")
if enhance_scanned_image:
    logger.info(f"    • 去噪强度: {denoise_strength}")
    logger.info(f"    • 锐化强度: {sharpen_strength}")
    logger.info(f"    • 输出格式: {'高对比度黑白图像' if output_binary else '灰度图'}")
logger.info(f"  - 角度检测: {'✓ 已启用' if detect_angle else '✗ 已禁用'}")
if detect_angle:
    logger.info(f"    • 角度阈值: {angle_threshold}°")
logger.info(f"  - 透视校正: {'✓ 已启用' if correct_perspective else '✗ 已禁用'}")
logger.info(f"\n📁 输出目录: {output_dir}")
logger.info("="*50)
