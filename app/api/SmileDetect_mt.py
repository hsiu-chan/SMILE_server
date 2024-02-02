from flask import Flask, request, Blueprint,redirect, send_file, abort

#import uuid
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from pathlib import PurePath

import os

from lib.Smile import SMILE
from werkzeug.utils import secure_filename

from config import UPLOAD_FOLDER, DEVICE # 引用全局变量



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

SmileDetect_blueprint_mt = Blueprint('SmileDetect_blueprint_mt', __name__)

@SmileDetect_blueprint_mt.route('/SmileDetect_upload_mt', methods=['POST'])
def upload_img():
    # 检查是否有文件被上传
    if 'file' not in request.files:
        return redirect(request.url)
    
    file = request.files['file']

    # 如果用户未选择文件
    if file.filename == '':
        return redirect(request.url)
    
    # 文件格式不允许
    if not allowed_file(file.filename):
        return redirect(request.url)
    

    #if request.method== 'POST':
       

    filename = secure_filename(file.filename) # 确保文件名的安全性
    print(filename)

    file_path =PurePath(UPLOAD_FOLDER)/filename


    
    # 保存文件到上传文件夹
    file.save(file_path)
    return add(file_path)

     




def add(file_path):


    
    nowfig=SMILE(file_path, DEVICE)
    if not nowfig.find_all_tooth():
        return {'msg':"Face not found"}
        



    image_path = nowfig.output_path

    del nowfig






    return send_file( image_path, mimetype='image/jpeg')
