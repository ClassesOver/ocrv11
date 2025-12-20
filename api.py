from typing import Optional, List, Any
import numpy as np
import cv2
import requests
from obj_det.objd_util import  detection_img, text_ocr, paddle_ocr
from werkzeug.datastructures import FileStorage
from flask import request, Flask, jsonify
import traceback
from pydantic import BaseModel
from flask_pydantic import validate
from loguru import logger
import config
import base64
from decorator import decorator

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
