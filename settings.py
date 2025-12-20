# -*- coding: utf-8 -*-
"""
@author: lywen
"""
from loguru import logger
from config import *


if AngleModelFlag == 'tf' or ocrFlag == 'keras':

    if GPU:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPUID)
        import tensorflow as tf

    else:
        # CPU启动
        os.environ["CUDA_VISIBLE_DEVICES"] = ''


from crnn.keys import alphabetChinese, alphabetEnglish
if ocrFlag == 'torch':
    from crnn.network_torch import CRNN
    if chineseModel:
        alphabet = alphabetChinese
        if LSTMFLAG:
            ocrModel = ocrModelTorchLstm
        else:
            ocrModel = ocrModelTorchDense

    else:
        ocrModel = ocrModelTorchEng
        alphabet = alphabetEnglish
        LSTMFLAG = True
elif ocrFlag == 'opencv':
    from crnn.network_dnn import CRNN
    ocrModel = ocrModelOpencv
    alphabet = alphabetChinese
else:
    logger.debug("err,ocr engine in keras\opencv\darknet")

nclass = len(alphabet)+1
if ocrFlag == 'opencv':
    crnn = CRNN(alphabet=alphabet)
else:
    crnn = CRNN(32, 1, nclass, 256, leakyRelu=False, lstmFlag=LSTMFLAG, GPU=GPU, alphabet=alphabet)
if os.path.exists(ocrModel):
    crnn.load_weights(ocrModel)
else:
    logger.debug("download model or tranform model with tools!")

ocr_predict = crnn.ocr


