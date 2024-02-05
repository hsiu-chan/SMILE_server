
import os

ROOT_PATH=os.path.dirname(os.path.abspath(__file__))+'/'

UPLOAD_FOLDER =ROOT_PATH+"uploads/"

OUTPUT_FOLDER =ROOT_PATH+"outputs/"

MODEL_PATH = ROOT_PATH+"models/best.pt"

FLASK_PORT = 8888

DEVICE = 'cpu'
