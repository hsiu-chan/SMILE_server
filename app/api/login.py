from datetime import datetime
from flask import Blueprint, flash, request, jsonify
from flask_login import login_required, login_user, logout_user
from app import db
from lib.User import User
import bcrypt


Login_blueprint = Blueprint('Login_blueprint', __name__)

# 登入路由
@Login_blueprint.route('/login', methods=['POST'])
def login():
    
    
    username = request.form['username']
    password = request.form['password']
    user = User.query.filter_by(username=username).first()
    if user and  user.check_password(password):
        login_user(user)
        return jsonify(message="success")
    return jsonify(message='Invalid username or password')

# 註冊路由
@Login_blueprint.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    password = request.form['password']
    email = request.form['email']
    gender = request.form["gender"]
    birthday = request.form["birthday"]
    birthday = datetime.strptime(birthday, '%Y-%m-%d').date()
    
    if User.query.filter_by(username=username).first():
        return jsonify(message='Username already exists')
    
    if User.query.filter_by(email=email).first():
        return jsonify(message='Email already registered')
    
    try:
        new_user = User(username=username, 
                        password=password, 
                        email=email,
                        gender=gender,
                        birthdate=birthday)
        db.session.add(new_user)
        db.session.commit()
        login_user(new_user)
    except Exception as e:
        return jsonify(message=f"Register Failure: {e}")
    return jsonify(message="User registered successfully")


@Login_blueprint.route('/logout')
@login_required
def logout():
    logout_user()
    flash("Already Logout")
    return jsonify(message="Already Logout")
