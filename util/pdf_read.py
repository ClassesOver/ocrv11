# -*-coding=utf-8-*-
import os
import logging
import pyzbar
from PIL import Image
import zxing
import random
# import zbarlight

logger = logging.getLogger(__name__)
if not logger.handlers: logging.basicConfig(level=logging.INFO)
DEBUG = (logging.getLevelName(logger.getEffectiveLevel()) == 'DEBUG')

#
# def ocr_qrcode_zbar(filename):
#     img = Image.open(filename)
#     width, height = img.size
#     raw = img.tobytes()
#
#     scanner = pyzbar.Scanner()
#     scanner.parse_config('enable')
#     # 把图像装换成数据
#     zarimage = zbar.Image(width, height, 'Y800', raw)
#     # 扫描器进行扫描
#     scanner.scan(zarimage)
#
#     data = ''
#     for symbol in zarimage:
#         # 对结果进行一些有用的处理
#         data += symbol.data
#     if data:
#         logger.debug(u'识别二维码:%s,内容: %s' % (filename, data))
#     else:
#         logger.error(u'识别zbar二维码出错:%s' % (filename))
#         img.save('%s-zbar.jpg' % filename)
#     return data
#
#
# def ocr_qrcode_zbarlight(filename):
#     img = Image.open(filename)
#
#     width, height = img.size
#     raw = img.tobytes()
#
#     # 把图像装换成数据
#     data = zbarlight.qr_code_scanner(raw, width, height)
#
#     if data:
#         logger.debug(u'识别二维码:%s,内容: %s' % (filename, data))
#     else:
#         logger.error(u'识别zbarlight二维码出错:%s' % (filename))
#         img.save('%s-zbar.jpg' % filename)
#     return data


def ocr_qrcode_zxing(filename):
    # 在当前目录生成临时文件，规避java的路径问题
    img = Image.open(filename)
    ran = int(random.random() * 100000)
    img.save('%s%s.jpg' % (os.path.basename(filename).split('.')[0], ran))
    zx = zxing.BarCodeReader()
    data = ''
    zxdata = zx.decode('%s%s.jpg' % (os.path.basename(filename).split('.')[0], ran))
    # 删除临时文件
    os.remove('%s%s.jpg' % (os.path.basename(filename).split('.')[0], ran))
    if zxdata:
        logger.debug(u'zxing识别二维码:%s,内容: %s' % (filename, zxdata.parsed))
        data = zxdata.parsed
    else:
        logger.error(u'识别zxing二维码出错:%s' % (filename))
        img.save('%s-zxing.jpg' % filename)
    return data


if __name__ == '__main__':
    filename = r'/Users/fengzetao/PycharmProjects/ocr/code/images/invoice/qrcode.png'
    # zx = zxing.BarCodeReader()
    # print(zx.decode('qrcode22472.jpg'))
    # # zbar二维码识别
    # ltext = ocr_qrcode_zbar(filename)
    #
    # # zbarlight二维码识别
    # ltext = ocr_qrcode_zbarlight(filename)

    # zxing二维码识别
    print(ocr_qrcode_zxing(filename))