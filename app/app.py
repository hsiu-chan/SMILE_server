import bcrypt
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from flask_mail import Mail
from config import UPLOAD_FOLDER, SECRET_KEY

# 初始化 SQLAlchemy
db = SQLAlchemy()
login_manager = LoginManager()

# 創建應用工廠函數
def create_app():
    app = Flask(__name__, static_url_path='/static/', static_folder='static/')
    app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

    # mysql 設定
    app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql://user:password@db:3306/smile_user_data'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.secret_key = SECRET_KEY

    # mail 設定
    #app.config['MAIL_SERVER'] = 'mail'  # Docker Compose 服務名稱
    #app.config['MAIL_PORT'] = 1025  # MailHog 的 SMTP 端口
    #app.config['MAIL_USERNAME'] = None  # MailHog 不需要身份驗證
    #app.config['MAIL_PASSWORD'] = None  # MailHog 不需要身份驗證
    #app.config['MAIL_USE_TLS'] = False
    #app.config['MAIL_USE_SSL'] = False

    # 初始化擴展
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'Login_blueprint.login'

    # 註冊 Blueprint
    from api.SmileDetect import SmileDetect_blueprint
    from api.SmileDetect_mt import SmileDetect_blueprint_mt
    from api.SmileDetect_json import SmileDetect_json_blueprint
    from website import website_pages_blueprint, home_blueprint
    from api.login import Login_blueprint  # 確保這個導入在 app 的路徑中

    app.register_blueprint(website_pages_blueprint)
    app.register_blueprint(home_blueprint)
    app.register_blueprint(SmileDetect_blueprint)
    app.register_blueprint(SmileDetect_blueprint_mt)
    app.register_blueprint(SmileDetect_json_blueprint)
    app.register_blueprint(Login_blueprint, url_prefix='/api')  # 為 API 設定前綴

    return app


# 載入用戶的回調函數
@login_manager.user_loader
def load_user(user_id):
    from lib.User import User
    return User.query.get(int(user_id))


# 啟動應用
if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=8777)
