from typing import Optional, List, Any
import numpy as np
import cv2
import requests
from obj_det.objd_util import  detection_img, text_ocr, paddle_ocr
from werkzeug.datastructures import FileStorage
from flask import request, Flask, jsonify, send_file, after_this_request
import traceback
from pydantic import BaseModel
from flask_pydantic import validate
from loguru import logger
import config
import base64
import os
import zipfile
import tempfile
import shutil
from decorator import decorator
from obj_det.cls_det_pipeline import process_image_pipeline, save_yolo_label, IMAGE_EXTENSIONS

app = Flask(__name__)
app.config.from_object(config)

def img_decode(content: bytes):
    np_arr = np.frombuffer(content, dtype=np.uint8)
    return cv2.imdecode(np_arr, cv2.IMREAD_COLOR)


def auth(method):
    def loop(method, *args, **kwargs):
        if request.method != "POST":
            return jsonify({'result': "禁止使用GET请求!", 'code': 400})
        if request.headers.get('secret') != '6aac5f82-141b-44a4-817f-369c64b12b19':
            return jsonify({'result': "非法访问!", 'code': 401})
        return method(*args, **kwargs)

    return decorator(loop, method)


class RequestBodyModel(BaseModel):
    data: Optional[List]


# Example 1: query parameters only
@app.route("/ocr", methods=["POST"])
@validate()
@auth
def ocr(body: RequestBodyModel):
    list_invoice = []
    try:
        if body.data:
            for d in body.data:
                try:
                    base64_data = get_base64_file(d.get('data'))
                    ocr_data = detection_img(img_decode(base64.decodebytes(base64_data)))
                    list_invoice.append({'attachment_id': d.get('attachment_id'),
                                         'success': True,
                                         'message': "",
                                         'data': ocr_data})
                except Exception as e:
                    logger.error(traceback.format_exc())
                    list_invoice.append({'attachment_id': d.get('attachment_id'),
                                         'success': False,
                                         'message': str(e),
                                         'data': {}})

    except:
        jsonify({'result': list_invoice, 'code': 500})
    return jsonify({'result': list_invoice, 'code': 200})


@app.route("/test_ocr", methods=["POST"])
@auth
def test_ocr():
    list_invoice = []
    storage = request.files['file']
    # 从请求参数中获取 saveImage，默认为 False
    saveImage = request.form.get('saveImage', 'false').lower() in ('true', '1', 'yes')
    try:
        base64_data = get_base64_file(storage)
        ocr_data = detection_img(img_decode(base64.decodebytes(base64_data)), saveImage=saveImage)
        list_invoice.append({'attachment_id': 1,
                             'success': True,
                             'message': "",
                             'data': ocr_data})
    except Exception as e:
        logger.error(traceback.format_exc())
        list_invoice.append({'attachment_id': 1,
                             'success': False,
                             'message': str(e),
                             'data': {}})
    return jsonify({'result': list_invoice, 'code': 200})




@app.route("/chineseOcr", methods=["POST"])
@auth
def chineseOcr():
    storage = request.files['file']
    try:
        base64_data = get_base64_file(storage)
        ocr_data = text_ocr(img_decode(base64.decodebytes(base64_data)))
        list_invoice = ocr_data
    except Exception as e:
        logger.error(traceback.format_exc())
        list_invoice = "识别错误"
    return jsonify({'result': list_invoice, 'code': 200})


@app.route("/paddle_ocr", methods=["POST"])
@auth
def paddle_ocr_api():
    storage = request.files['file']
    try:
        base64_data = get_base64_file(storage)
        ocr_data = paddle_ocr(img_decode(base64.decodebytes(base64_data)))
        list_invoice = ocr_data
    except Exception as e:
        logger.error(traceback.format_exc())
        list_invoice = "识别错误"
    return jsonify({'result': list_invoice, 'code': 200})


@app.route("/classify_and_detect", methods=["POST"])
@auth
def classify_and_detect():
    """
    分类检测批量处理接口
    
    - 接收包含图片的zip压缩包
    - 对每张图片先进行分类，再根据分类结果选择对应的检测模型进行预测
    - 返回包含images和labels两个目录的zip压缩包
    
    请求参数（form-data）:
        file: zip压缩包文件
        classification_conf: 分类置信度阈值（可选，默认0.618）
        detection_conf: 检测置信度阈值（可选，默认使用模型配置）
        save_conf: 标签文件是否包含置信度（可选，默认True）
    """
    temp_dir = None
    
    try:
        # 验证文件存在
        if 'file' not in request.files:
            return jsonify({'result': "请上传zip压缩包文件", 'code': 400})
        
        file_storage = request.files['file']
        if not file_storage.filename or not file_storage.filename.endswith('.zip'):
            return jsonify({'result': "请上传zip压缩包文件", 'code': 400})
        
        logger.info(f"收到分类检测批量处理请求，文件名: {file_storage.filename}")
        
        # 获取参数
        classification_conf = float(request.form.get('classification_conf', 0.618))
        detection_conf_str = request.form.get('detection_conf', None)
        detection_conf = float(detection_conf_str) if detection_conf_str else None
        save_conf = request.form.get('save_conf', 'false').lower() in ('true', '1', 'yes')
        
        # 创建临时目录
        temp_dir = tempfile.mkdtemp(prefix='cls_det_')
        input_dir = os.path.join(temp_dir, 'input')
        output_dir = os.path.join(temp_dir, 'output')
        os.makedirs(input_dir, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        
        images_dir = os.path.join(output_dir, 'images')
        labels_dir = os.path.join(output_dir, 'labels')
        os.makedirs(images_dir, exist_ok=True)
        os.makedirs(labels_dir, exist_ok=True)
        
        # 保存上传的zip文件
        zip_path = os.path.join(temp_dir, 'input.zip')
        file_storage.save(zip_path)
        
        file_size = os.path.getsize(zip_path)
        logger.info(f"zip文件大小: {file_size / 1024:.2f} KB")
        
        # 解压zip文件
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(input_dir)
        
        logger.info("解压完成，开始处理图片...")
        
        # 收集所有图片文件
        image_files = []
        for root, dirs, files in os.walk(input_dir):
            for file_name in files:
                ext = os.path.splitext(file_name)[1].lower()
                if ext in IMAGE_EXTENSIONS:
                    image_files.append(os.path.join(root, file_name))
        
        if not image_files:
            return jsonify({'result': "zip压缩包中未找到有效的图片文件", 'code': 400})
        
        logger.info(f"找到 {len(image_files)} 张图片")
        
        # 统计信息
        success_count = 0
        fail_count = 0
        classification_stats = {}
        detection_stats = {}
        
        # 处理每张图片
        for idx, image_path in enumerate(image_files, 1):
            try:
                filename = os.path.basename(image_path)
                logger.info(f"处理第 {idx}/{len(image_files)} 张: {filename}")
                
                # 读取图片
                img = cv2.imread(image_path)
                if img is None:
                    logger.warning(f"无法读取图片: {filename}")
                    fail_count += 1
                    continue
                
                # 执行分类-检测流水线
                result = process_image_pipeline(
                    img,
                    classification_conf=classification_conf,
                    detection_conf=detection_conf
                )
                
                if not result['success']:
                    logger.warning(f"处理失败 {filename}: {result.get('error', '未知错误')}")
                    fail_count += 1
                    continue
                
                # 获取分类结果
                classification = result['classification']
                detection = result['detection']
                
                # 统计分类结果
                class_name = classification['predicted_class']
                classification_stats[class_name] = classification_stats.get(class_name, 0) + 1
                
                # 统计检测结果
                if detection:
                    for det in detection:
                        det_class = det['class_name']
                        detection_stats[det_class] = detection_stats.get(det_class, 0) + 1
                
                # 保存图片到images目录
                image_output_path = os.path.join(images_dir, filename)
                cv2.imwrite(image_output_path, img)
                
                # 保存标签文件到labels目录
                label_name = os.path.splitext(filename)[0] + '.txt'
                label_path = os.path.join(labels_dir, label_name)
                
                if detection:
                    save_yolo_label(label_path, detection, save_conf=save_conf)
                else:
                    # 如果没有检测到目标，创建空标签文件
                    with open(label_path, 'w', encoding='utf-8') as f:
                        pass
                
                success_count += 1
                logger.info(f"✓ 处理成功: {filename} (分类: {class_name}, 检测到 {len(detection) if detection else 0} 个目标)")
                
            except Exception as e:
                logger.error(f"处理图片失败 {image_path}: {e}\n{traceback.format_exc()}")
                fail_count += 1
        
        logger.info(f"处理完成: 成功 {success_count} 张, 失败 {fail_count} 张")
        logger.info(f"分类统计: {classification_stats}")
        logger.info(f"检测统计: {detection_stats}")
        
        # 创建输出zip文件
        output_zip_path = os.path.join(temp_dir, 'result.zip')
        with zipfile.ZipFile(output_zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_ref:
            # 添加images目录
            for root, dirs, files in os.walk(images_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    arc_name = os.path.relpath(file_path, output_dir)
                    zip_ref.write(file_path, arc_name)
            
            # 添加labels目录
            for root, dirs, files in os.walk(labels_dir):
                for file_name in files:
                    file_path = os.path.join(root, file_name)
                    arc_name = os.path.relpath(file_path, output_dir)
                    zip_ref.write(file_path, arc_name)
        
        logger.info(f"生成输出zip文件: {output_zip_path}")
        
        # 设置清理回调（在响应发送后执行）
        @after_this_request
        def cleanup(response):
            """清理临时文件"""
            try:
                if temp_dir and os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
                    logger.debug(f"已清理临时目录: {temp_dir}")
            except Exception as e:
                logger.warning(f"清理临时目录失败: {e}")
            return response
        
        # 返回zip文件
        return send_file(
            output_zip_path,
            mimetype='application/zip',
            as_attachment=True,
            download_name='result.zip'
        )
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"分类检测批量处理失败: {error_msg}\n{traceback.format_exc()}")
        
        # 清理临时目录
        if temp_dir and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)
        
        return jsonify({'result': f"处理失败: {error_msg}", 'code': 500})


@app.route("/get_image", methods=["GET"])
@auth
def get_image():
    """
    根据checksum获取保存的图片文件
    
    请求参数（query string）:
        checksum: 图片的MD5 checksum（32位十六进制字符串，必需）
        prefix: 可选的目录前缀，默认为 'signatures'，也支持 'images' 及其子目录
    
    返回:
        图片文件（如果存在），否则返回错误信息
    """
    try:
        checksum = request.args.get('checksum', '').strip()
        prefix = request.args.get('prefix', 'signatures').strip()
        
        # 验证checksum参数
        if not checksum:
            return jsonify({'result': "请提供checksum参数", 'code': 400})
        
        # 验证checksum格式（MD5是32位十六进制字符串）
        checksum_lower = checksum.lower()
        if len(checksum_lower) != 32 or not all(c in '0123456789abcdef' for c in checksum_lower):
            return jsonify({'result': "checksum格式错误，应为32位十六进制字符串", 'code': 400})
        
        # 验证并清理prefix参数，防止路径遍历攻击
        if not prefix:
            prefix = 'signatures'
        
        # 防止路径遍历攻击：不允许包含 ..、绝对路径、特殊字符
        if '..' in prefix or os.path.isabs(prefix) or ('\\' in prefix and os.name != 'nt'):
            return jsonify({'result': "目录前缀不合法", 'code': 400})
        
        # 清理路径分隔符，统一使用系统分隔符
        prefix = prefix.replace('\\', '/').strip('/')
        if not prefix:
            prefix = 'signatures'
        
        # 构建文件路径
        filename = f"{checksum_lower}.png"
        filepath = os.path.join(config.base_dir, 'images', prefix, filename)
        
        # 规范化路径并验证安全性
        filepath = os.path.normpath(filepath)
        
        # 确保路径在base_dir范围内，防止路径遍历
        base_dir_normalized = os.path.normpath(config.base_dir)
        try:
            # 使用commonpath确保路径在base_dir内（跨平台安全）
            common_path = os.path.commonpath([base_dir_normalized, filepath])
            if common_path != base_dir_normalized:
                logger.warning(f"路径遍历攻击尝试: {filepath}")
                return jsonify({'result': "路径不合法", 'code': 403})
        except ValueError:
            # 如果路径不在同一驱动器（Windows）或完全不同，视为不合法
            logger.warning(f"路径不在base_dir范围内: {filepath}")
            return jsonify({'result': "路径不合法", 'code': 403})
        
        # 检查文件是否存在
        if os.path.exists(filepath) and os.path.isfile(filepath):
            logger.debug(f"返回图片: {filepath}")
            return send_file(
                filepath,
                mimetype='image/png',
                as_attachment=False
            )
        
        # 如果指定路径不存在，尝试从常见位置查找
        common_prefixes = ['signatures', 'images/signatures']
        for common_prefix in common_prefixes:
            if common_prefix == prefix:
                continue  # 已经尝试过了
            
            test_path = os.path.join(config.base_dir, common_prefix, filename)
            test_path = os.path.normpath(test_path)
            
            # 验证路径安全性
            try:
                test_common_path = os.path.commonpath([base_dir_normalized, test_path])
                is_safe = (test_common_path == base_dir_normalized)
            except ValueError:
                is_safe = False
            
            if is_safe and os.path.exists(test_path) and os.path.isfile(test_path):
                logger.debug(f"从备用路径返回图片: {test_path}")
                return send_file(
                    test_path,
                    mimetype='image/png',
                    as_attachment=False
                )
        
        # 文件不存在
        logger.warning(f"图片不存在 (checksum: {checksum}, prefix: {prefix})")
        return jsonify({'result': f"图片不存在 (checksum: {checksum})", 'code': 404})
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"获取图片失败: {error_msg}\n{traceback.format_exc()}")
        return jsonify({'result': f"获取图片失败: {error_msg}", 'code': 500})


def get_base64_file(data):
    base64_str = b""
    if isinstance(data, FileStorage):
        assert data.content_type and data.content_type.startswith('image'), '上传的文件不支持，请转为图片上传！'
        base64_str = base64.b64encode(data.read())
    if isinstance(data, str):
        if data.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://')):
            response = requests.get(data)
            assert response.headers.get('Content-Type').startswith('image'), '上传的文件不支持，请转为图片上传！'
            base64_str = base64.encodebytes(response.content)
        else:
            base64_str = data.encode("utf-8")
    if isinstance(data, bytes):
        base64_str = bytes
    assert base64_str, '简析图片失败！'
    return base64_str


if __name__ == '__main__':
    app.run('0.0.0.0', port=8078, debug=False, threaded=False, processes=1)
