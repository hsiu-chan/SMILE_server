from app import create_app
from config import FLASK_PORT
app = create_app()

if __name__ == '__main__':
    from werkzeug.serving import run_simple
    run_simple('0.0.0.0', 8777, app)
    #app.run('0.0.0.0',port=8888)
