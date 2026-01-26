from qreader import QReader
import pyzbar.pyzbar as pyzbar
import cv2

reader = QReader(weights_folder=r'D:\ocr\ocrv5\ocr\code\models\qrdet')
img = cv2.imread(r'E:\s\qrcode.png')
print(pyzbar.decode(img))
code = reader.detect_and_decode(img)
print(code)
