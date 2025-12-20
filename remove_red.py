# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 22:46
# @Author  : FengZeTao zetao.feng@hand-china.com
# @File    : remove_red.py
# 如遇到看不懂且无备注的部分，请联系相关人

# -*- encoding: utf-8 -*-
import cv2
import numpy as np


class SealRemove(object):
    """
    印章处理类
    """
    def remove_red_seal(self, image):
        """
        去除红色印章
        """

        # 获得红色通道
        blue_c, green_c, red_c = cv2.split(image)

        # 多传入一个参数cv2.THRESH_OTSU，并且把阈值thresh设为0，算法会找到最优阈值
        thresh, ret = cv2.threshold(red_c, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 实测调整为95%效果好一些
        filter_condition = int(thresh * 0.95)

        _, red_thresh = cv2.threshold(red_c, filter_condition, 255, cv2.THRESH_BINARY)

        # 把图片转回 3 通道
        result_img = np.expand_dims(red_thresh, axis=2)
        result_img = np.concatenate((result_img, result_img, result_img), axis=-1)

        return result_img

if __name__ == '__main__':
    image = '/Users/fengzetao/PycharmProjects/ocr/code/images/invoice/checkcode.png'
    img = cv2.imread(image)
    seal_rm = SealRemove()
    rm_img = seal_rm.remove_red_seal(img)
    cv2.imwrite("/Users/fengzetao/PycharmProjects/ocr/code/images/invoice/text_checkout_code_no_red.png",rm_img)