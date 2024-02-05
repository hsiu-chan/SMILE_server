from flask import Flask, request, Blueprint,jsonify,current_app
from lib.Base64Converter import url_to_img,path_to_base64

#from lib.SMILE import SMILE
#import uuid
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

import cv2
import mediapipe as mp
import os
from config import MODEL_PATH, DEVICE,OUTPUT_FOLDER
#from flask_mail import Mail, Message
from lib.Smile import SMILE


Model = YOLO(MODEL_PATH)
mp_face_detection = mp.solutions.face_detection
mp_face_mesh = mp.solutions.face_mesh
face_mesh=mp_face_mesh.FaceMesh(
            min_detection_confidence=0.2,
            min_tracking_confidence=0.2)


SmileDetect_blueprint = Blueprint('SmileDetect_blueprint', __name__)

@SmileDetect_blueprint.route('/SmileDetect_upload', methods=['POST'])
def upload_img():
    if request.method== 'POST':
        return add(request.get_json())
     




def add(data):
    #img_src=str(data.get('image'))

    img,ext=url_to_img(data['image'])
    
    #id=uuid.uuid4()
    #filename = "upload_fig/{}.{}".format(id, ext)
    filename = "{}.{}".format('input', ext)
    print (filename)

    with open(filename, "wb") as f:
        f.write(img)
    
    nowfig=SMILE(filename, DEVICE)
    nowfig.find_all_tooth()
    #mask,sc=nowfig.predict([[50,14]])

    b64=path_to_base64(nowfig.output_path)
    del nowfig


    return {'msg': 'success','filename':filename,"result":b64, "score":100}
