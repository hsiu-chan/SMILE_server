from flask import Flask 
from api.SmileDetect import SmileDetect_blueprint
from api.SmileDetect_mt import SmileDetect_blueprint_mt
from website import website_pages_blueprint,home_blueprint #所有網頁

from config import UPLOAD_FOLDER



def create_app():#Application Factories
    app = Flask(__name__, static_url_path='/static/', 
            static_folder='static/')
    
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


        
    app.register_blueprint(website_pages_blueprint)
    app.register_blueprint(home_blueprint)

    ###API###
    app.register_blueprint(SmileDetect_blueprint)
    app.register_blueprint(SmileDetect_blueprint_mt)
    
    return app
