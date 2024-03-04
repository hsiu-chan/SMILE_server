
import os

ROOT_PATH=os.path.dirname(os.path.abspath(__file__))+'/'

UPLOAD_FOLDER =ROOT_PATH+"uploads/"

OUTPUT_FOLDER =ROOT_PATH+"outputs/"


FLASK_PORT = 8888

DEVICE = 'cpu'
"""YOLO設備(cpu,cuda,mps)"""

MODEL_PATH = ROOT_PATH+"models/best.pt"
"""YOLO模型"""

RANDOM_FOREST_MODEL = ROOT_PATH+'models/random_forest_model.joblib'
