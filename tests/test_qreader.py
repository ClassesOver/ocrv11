from qreader import QReader
import cv2

reader = QReader(weights_folder=r'D:\ocr\ocrv5\ocr\code\models\qrdet')
img = cv2.imread(r'D:\ocr\ocrv5\ocr\code\images\stock_v1\qrcode.png')
code = reader.detect_and_decode(img)
print(code)
