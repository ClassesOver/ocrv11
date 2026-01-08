import requests

url = "http://127.0.0.1:8078/classify_and_detect"

payload = {}
headers = {
    'secret': '6aac5f82-141b-44a4-817f-369c64b12b19',
    'User-Agent': 'Apifox/1.0.0 (https://apifox.com)'
}

# 使用上下文管理器自动关闭文件
with open('D:\\ocr\\ocrv5\\ocr\\code\\tests\\images\\classified\\stock2.zip', 'rb') as f:
    files = [
        ('file', ('stock1.zip', f, 'application/zip'))
    ]
    response = requests.post(url, headers=headers, data=payload, files=files)

    if response.status_code == 200:
        with open('result.zip', 'wb') as output_file:
            output_file.write(response.content)
        print("文件已保存到: result.zip")
    else:
        print(f"请求失败: {response.status_code}")
        print(response.text)