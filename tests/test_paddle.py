# -*- coding: utf-8 -*-
# @Time    : 2022/11/4 22:24
# @Author  : FengZeTao zetao.feng@hand-china.com
# @File    : test_paddle.py
# 如遇到看不懂且无备注的部分，请联系相关人
import cv2
import numpy as np



image = cv2.imread('/Users/fengzetao/PycharmProjects/ocr/code/images/invoice/qrcode.png')

image = cv2.resize(image, (600, 600))
height, width = image.shape[:2]
#size = (int(width * 0.25), int(height * 0.25))
#shrink = cv2.resize(image, size, interpolation=cv2.INTER_AREA)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imwrite('/Users/fengzetao/PycharmProjects/ocr/code/images/qrcodes/gray.jpg',gray)

ret2, image_binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
ret, binary = cv2.threshold(gray, ret2 * 0.85, 255, cv2.THRESH_BINARY)

#ret, binary = cv2.threshold(gray, 135, 255, cv2.THRESH_BINARY)
cv2.imwrite('/Users/fengzetao/PycharmProjects/ocr/code/images/qrcodes/binary.jpg',binary)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (80, 80))
iOpen = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
iClose = cv2.morphologyEx(iOpen, cv2.MORPH_CLOSE, kernel)
cv2.imwrite('/Users/fengzetao/PycharmProjects/ocr/code/images/qrcodes/iClose.jpg',iClose)
# cv2.imwrite('tempcolse.jpg',iClose)

img = 255 - iClose

cv2.imwrite('/Users/fengzetao/PycharmProjects/ocr/code/images/qrcodes/img.jpg',img)



def Get_cnt(edged):
    cnts = cv2.findContours(edged.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0]
    docCnt = None
    if len(cnts) > 0:
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
        for c in cnts:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                docCnt = approx
                break
    return docCnt


def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def four_point_transform(simage, pts,gap=50):
    # print(pts)
    rect = order_points(pts)

    (tl, tr, br, bl) = rect
    tl[0] = tl[0]-gap
    tl[1] = tl[1]-gap

    tr[0] = tr[0]+gap
    tr[1] = tr[1]-gap


    br[0] = br[0]+gap
    br[1] = br[1]+gap

    bl[0] = bl[0]-gap
    bl[1] = bl[1]+gap

    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    warped = cv2.copyMakeBorder(warped,50,50,50,50, cv2.BORDER_CONSTANT,value=[255,255,255])
    return warped



#69,64   555,551
#sx,sy = _black_edges(img)
#print(sx,sy)

ss = Get_cnt(img)
print(ss)

warped = four_point_transform(image,ss.reshape(4, 2),gap=35)#这个gap阈值控制下仿射变换的余量，避免有些二维码变换后识别不出来

cv2.imwrite('/Users/fengzetao/PycharmProjects/ocr/code/images/qrcodes/warped.jpg',warped)
