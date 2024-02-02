import base64
import re
from io import BytesIO
from PIL import Image

def url_to_img(url):
    result=re.search("data:image/(?P<ext>.*?);base64,(?P<data>.*)", url, re.DOTALL)
    if result:
        ext = result.groupdict().get("ext")
        img = result.groupdict().get("data")
        img=base64.b64decode(img)
        return img,ext
    else:
        return base64.b64decode(url), type_of_base64_picture(url)



    return 'Not Found' 'Not Found'


def path_to_base64(img_path):
    with open(img_path, 'rb') as f:
        img = f.read()
        base64_str = base64.b64encode(img)  # base64编码
        #print ( str(base64_str, 'utf-8'))
    return str(base64_str, 'utf-8')
    


def type_of_base64_picture(base64_data):
    try:
        # 解码base64数据
        decoded_data = base64.b64decode(base64_data)
        
        # 使用PIL库打开图像
        image = Image.open(BytesIO(decoded_data))
        
        # 获取图像格式
        image_format = image.format
        
        return image_format
    except Exception as e:
        print(f"Error detecting image format: {e}")
        return None
