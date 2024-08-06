from flask import Blueprint, request, jsonify
from flask_login import login_user
from app import db, User

Login_blueprint = Blueprint('Login_blueprint', __name__)

# 登入路由
@Login_blueprint.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']
    user = User.query.filter_by(username=username).first()
    if user and user.password == password:
        login_user(user)
        return jsonify(message="success")
    return jsonify(message='Invalid username or password')

# 註冊路由
@Login_blueprint.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    email = request.form['email']
    
    if User.query.filter_by(username=username).first():
        return jsonify(message='Username already exists')
    
    if User.query.filter_by(email=email).first():
        return jsonify(message='Email already registered')
    
    new_user = User(username=username, password=password, email=email)
    db.session.add(new_user)
    db.session.commit()
    
    login_user(new_user)
    return jsonify(message="User registered successfully")
