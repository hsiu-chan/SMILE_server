from flask import Flask, request, Blueprint,redirect, send_file, abort, jsonify, Response




from pathlib import PurePath
import json

import os

from lib.Smile import SMILE
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER, DEVICE


# 允許的文件類型
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# 文件類型檢查
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def check_file_permission(file_path):
    # 检查文件是否存在
    if not os.path.exists(file_path):
        return False

    # 检查当前用户是否有读取权限
    if not os.access(file_path, os.R_OK):
        return False

    return True

SmileDetect_json_blueprint = Blueprint('SmileDetect_json_blueprint', __name__)

@SmileDetect_json_blueprint.route('/SmileDetect_upload_json', methods=['POST'])
def upload_img():
    # 检查是否有文件被上传
    if 'file' not in request.files:
        return json.dump({'error':["'file' not in request"]})
    
    file = request.files['file']

    # 如果用户未选择文件
    if file.filename == '':
        return json.dump({'error':["filename is empty"]})
    
    # 文件格式不允许
    if not allowed_file(file.filename):
        return json.dump({'error':["file not allowed"]})
    

    filename = secure_filename(file.filename) # 确保文件名的安全性

    file_path =PurePath(UPLOAD_FOLDER)/filename
    print('file_path:',file_path)


    
    # 保存文件到上传文件夹
    file.save(file_path)
    return add(str(file_path))

     




def add(file_path): ## 辨識微笑並回傳結果


    
    #output=f'{OUTPUT_FOLDER}output.png' ## 輸出路徑
    
    nowfig=SMILE(file_path, DEVICE,filter=0.75)
    #if not nowfig.find_all_tooth():
        #return {'message':"Face not found"}
    
    nowfig.find_all_tooth()
    

    

    """multipartData = MultipartEncoder(
        fields={
            'info': (None, json.dumps(nowfig.smile_info), 'application/json'),
            'error': '\n'.join(nowfig.error),
            'image': ('smile_result', open(output, 'rb'), 'image/png')
        }
    )"""

    return json.dumps(nowfig.smile_info)



    