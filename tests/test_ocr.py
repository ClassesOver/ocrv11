from  paddleocr import PaddleOCR
from paddleocr import DocImgOrientationClassification



imgori = DocImgOrientationClassification()
def predict_ori_img(img):
    res = imgori.predict(img)
    for r in res:
        if 'label_names' in r:
            for i in r['label_names']:
                return int(i)

print(predict_ori_img(r'E:\0108\stock1\images\IMAGE2008458086037196800.jpg'))

