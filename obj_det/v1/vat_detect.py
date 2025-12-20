import torch

from models.experimental import attempt_load
from utils.datasets import letterbox
from utils.general import non_max_suppression, scale_coords, set_logging
from utils.torch_utils import select_device
import threading
import config
from util.tool import *

lock = threading.Lock()

converter = {'invoice_code': 'invoice_code',
             'invoice_code2': 'invoice_code2',
             'invoice_number': 'invoice_number',
             'invoice_number2': 'invoice_number2',
             'bill_date': 'billing_date',
             'check_code': 'check_code',
             'check_code2': 'check_code2',
             'title': 'title',
             'total': 'total_amount',
             'tax': 'tax',
             'amount_with_tax': 'amount_with_tax',
             'invoice_type': 'invoice_type'}

if config.ocrRange == "complex":
    converter.update({
        'buy_title': 'buy_title',
        'buy_tax': 'buy_tax',
        'buy_addr': 'buy_addr',
        'buy_bank': 'buy_bank',
        'sale_title': 'sale_title',
        'sale_tax': 'sale_tax',
        'sale_addr': 'sale_addr',
        'sale_bank': 'sale_bank'
    })
    # converter.update({
    #     'buy_title': 'buy_title',
    #     'buy_bank': 'buy_bank',
    #     'sale_title': 'sale_title',
    #     'sale_bank': 'sale_bank'
    # })
type_converter_name = {'01': '增值税专用发票', '04': '增值税普通发票',
                       '08': '增值税电子专用发票', '10': '增值税电子普通发票',
                       '31': '电子发票（增值税专用发票）', '32': '电子发票（增值税普通发票）',
                       '': '未识别的发票'}
type_converter = {'增值税专用发票': '01', '增值税普通发票': '04',
                  '增值税电子专用发票': '08', '增值税电子普通发票': '10',
                  '电子发票（增值税专用发票）': '31', '电子发票（增值税普通发票）': '32'}

pub_weights = f"models/vat/best_{config.ocrRange}.pt"
pub_img_size = 640
pub_augment = False
pub_conf_thres = 0.25
pub_iou_thres = 0.24
pub_classes = None
pub_agnostic_nms = False

# Initialize
set_logging()
if config.GPU:
    pub_device = "%s" % config.GPUID
else:
    pub_device = 'cpu'
device = select_device(pub_device)
half = device.type != 'cpu'  # half precision only supported on CUDA

# Load model
if config.GPU:
    model = attempt_load(pub_weights, device=lambda storage, loc: storage.cuda(config.GPUID))
else:
    model = attempt_load(pub_weights, device='cpu')  # load FP32 model
stride = int(model.stride.max())  # model stride


def get_check_code(code1, code2):
    if not code2:
        return get_num(code1)
    if code1 and '验码' in code1:
        return get_num(code1)
    if code2 and '验码' in code2:
        return get_num(code2)
    return max(get_num(code1), get_num(code2))


def judge_invoice_type(title, invoice):
    invoice_type = None
    if not title:
        return False
    if title.startswith('电子发票'):
        if "普通" in title:
            invoice_type = "32"
        else:
            invoice_type = "31"
    else:
        if "专用" in title:
            if '电子' in title:
                invoice_type = '08'
            else:
                invoice_type = '01'
            invoice['invoice_type'] = "01"
        if "普通" in title:
            if '电子' in title:
                invoice_type = '10'
            else:
                invoice_type = '04'
    if not invoice_type:
        if invoice.get('check_code'):
            invoice_type = "04"
        else:
            invoice_type = "01"
    invoice['invoice_type'] = invoice_type


def judge_invoice_repeat_data(invoice):
    # 对发票代码和发票号进行处理
    invoice_code = invoice.get('invoice_code', '')
    invoice_code2 = invoice.get('invoice_code2', '')
    if invoice_code != invoice_code2:
        if (len(invoice_code) != 12 and len(invoice_code2) == 12) or len(invoice_code) < len(invoice_code2):
            invoice['invoice_code'] = invoice_code2
    invoice_number = invoice.get('invoice_number', '')
    invoice_number2 = invoice.get('invoice_number2', '')
    if invoice_number != invoice_number2:
        in1 = len(invoice_number)
        in2 = len(invoice_number2)
        if in1 == 8:
            invoice_number_really = invoice_number
        elif in2 == 8:
            invoice_number_really = invoice_number2
        elif in2 > in1:
            invoice_number_really = invoice_number2
        else:
            invoice_number_really = invoice_number
        invoice['invoice_number'] = invoice_number_really
    invoice['check_code'] = get_check_code(invoice.get('check_code'), invoice.get('check_code2'))
    if invoice.__contains__('check_code2'):
        del invoice['check_code2']
    if invoice.__contains__('invoice_code2'):
        del invoice['invoice_code2']
    if invoice.__contains__('invoice_number2'):
        del invoice['invoice_number2']


def handle_buy_sale(paddleOCR, chineseModel, label, img):
    text = paddleOCR(img)
    if not text:
        text = chineseModel(img)
    if label in ('buy_tax', 'sale_tax'):
        text = get_tax(text)
    if label in ('buy_title', 'sale_title'):
        text = get_title(text)
    else:
        text = get_addr_bank(text)
    return text.strip()


def invoice_detection(img_numpy, invoice=None, context=None):
    names = model.module.names if hasattr(model, 'module') else model.names
    img = letterbox(img_numpy, pub_img_size, stride=stride)[0]
    # Convert
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)

    lock.acquire(timeout=3)
    # Inference
    pred = model(img, augment=pub_augment)[0]
    lock.release()

    # Apply NMS
    pred = non_max_suppression(pred, pub_conf_thres, pub_iou_thres, classes=pub_classes, agnostic=pub_agnostic_nms)
    # Process detections
    paddleOCR = context.paddleOCR.get_text
    chineseModel = context.chineseModel
    for i, det in enumerate(pred):  # detections per image
        im0 = img_numpy
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
            labels = {}
            for *xyxy, conf, cls in reversed(det):
                label = names[int(cls)]
                if label not in converter.keys():
                    continue
                # 前两个控制竖向坐标，后两个控制横向
                newimg_list = [int(xyxy[1] - 5), int(xyxy[3] + 5), int(xyxy[0] - 12), int(xyxy[2] + 12)]
                if label == 'check_code' and label in labels:
                    label = 'check_code2'
                if label == 'buy_bank':
                    type = paddleOCR(im0[int(xyxy[1] - 5):int(xyxy[3] + 5), int(xyxy[0] - 100):int(xyxy[0])])
                    if '电话' in type:
                        labels["buy_addr"] = newimg_list
                        continue
                if config.SaveImg:
                    invoice_fp = os.path.join('images', 'invoice')
                    path = os.path.join(invoice_fp, '%s.png' % label)
                    cv2.imwrite(path, im0[newimg_list[0]:newimg_list[1], newimg_list[2]:newimg_list[3]])
                labels[label] = newimg_list
            if 'invoice_number' in labels and ('invoice_code' not in labels or 'invoice_number2' not in labels):
                labels['invoice_number'][3] += 48
            labels = {key: im0[newimg_list[0]:newimg_list[1], newimg_list[2]:newimg_list[3]] for key, newimg_list in
                      labels.items()}
            has_qrcode = False
            if labels.__contains__('qrcode'):
                has_qrcode = qrcode_pyzbar(labels.get('qrcode'), invoice)
                if has_qrcode:
                    if invoice.get('invoice_type') == '32':
                        title = '电子发票（普通发票）'
                    elif invoice.get('invoice_type') == '31':
                        title = '电子发票（专用发票）'
                    else:
                        title = chineseModel(labels.get('title'))
                    invoice['title'] = title
                    if invoice.get('invoice_type') in ['01', '04']:
                        invoice['amount_with_tax'] = get_float(chineseModel(labels.get('amount_with_tax')))
                        tax = labels.get('tax')
                        if not paddleOCR(tax).replace('★', '').replace('*', ''):
                            invoice[converter['tax']] = get_float("")
                        else:
                            invoice['tax'] = get_float(chineseModel(tax))
                    if invoice.get('invoice_type') in ['31', '32']:
                        invoice['total_amount'] = get_float(chineseModel(labels.get('total')))
                        tax = labels.get('tax')
                        if not paddleOCR(tax).replace('★', '').replace('*', ''):
                            invoice[converter['tax']] = get_float("")
                        else:
                            invoice['tax'] = get_float(chineseModel(tax))
                    if config.ocrRange == 'complex':
                        for label, newimg in labels.items():
                            if label.startswith(('buy_', 'sale_')):
                                invoice[converter[label]] = handle_buy_sale(paddleOCR, label, newimg)
            if not has_qrcode:
                for label, newimg in labels.items():
                    if label == 'qrcode':
                        continue
                    if label in ('check_code', 'check_code2'):
                        text = paddleOCR(newimg)
                    elif label in ('invoice_number', 'invoice_number2'):
                        text = get_num(paddleOCR(newimg))
                    elif label in ('invoice_code2', 'invoice_code'):
                        text = get_num(paddleOCR(newimg))[-12:]
                    elif label == 'bill_date':
                        text = get_date(paddleOCR(newimg))
                        # text = get_date(chineseModel(newimg).strip())
                    elif label in ('total', 'amount_with_tax', 'tax'):
                        ch_str = chineseModel(newimg)
                        pd_str = paddleOCR(newimg)
                        is_ch = ch_str.startswith(('¥', 'Y'))
                        is_pd = pd_str.startswith(('¥', 'Y'))
                        s = ch_str
                        if is_pd and not is_ch:
                            s = pd_str
                        text = get_float(s.replace('★', '').replace('*', '') or "")
                    elif label.startswith(('buy_', 'sale_')):
                        text = handle_buy_sale(paddleOCR, chineseModel, label, newimg)
                    else:
                        text = chineseModel(newimg).strip()
                    invoice[converter[label]] = text
                title = invoice.get('title')
                if not title and 'title' in labels:
                    if invoice.get('check_code'):
                        invoice['title'] = title = '增值税普通发票'
                    else:
                        invoice['title'] = title = '增值税专用发票'
                judge_invoice_type(title, invoice)
            if "¥ 0.00" == invoice.get('tax'):
                comp = re.compile('-?[0-9]\d*\.*')
                total_amount = float(''.join(list(''.join(comp.findall(invoice.get('total_amount'))))))
                amount_with_tax = float(''.join(list(''.join(comp.findall(invoice.get('amount_with_tax'))))))
                invoice['tax'] ='¥ {}'.format(round(total_amount - amount_with_tax,2))
            if '-' not in invoice.get(
                    'tax') and ('-' in invoice.get('total_amount') or '-' in invoice.get('amount_with_tax')):
                invoice['tax'] = invoice['tax'].replace('¥ ', '¥ -')

            for val in converter.values():
                if invoice.get(val) is None:
                    if val in ('total', 'tax', 'amount_with_tax'):
                        invoice[val] = '¥ 0.00'
                    else:
                        invoice[val] = ""
            invoice['invoice_type_name'] = type_converter_name[invoice['invoice_type']]
            judge_invoice_repeat_data(invoice)
    return invoice
