# 用戶模型定義

import re
import bcrypt
from flask_login import UserMixin
from app import db
from sqlalchemy import Column, Integer, String, Enum, Date
from datetime import datetime



class User(db.Model, UserMixin):
    __tablename__ = 'users'

    id = db.Column(Integer, primary_key=True)
    username = db.Column(String(50), unique=True, nullable=False)
    password_hash = db.Column(String(128), nullable=False)
    email = db.Column(String(255), unique=True, nullable=False)
    gender = db.Column(Enum('male', 'female', 'other'), nullable=False)
    birthdate = db.Column(Date, nullable=False)



    def __init__(self, username: str, password: str, email: str, gender:str, birthdate:datetime.date):
        # 驗證 username 格式 (例如：只允許字母和數字，長度 3 到 20 個字符)
        if not re.match(r'^[a-zA-Z0-9]{3,20}$', username):
            raise ValueError("Username must be between 3 and 20 characters long and contain only letters and numbers.")
        if gender not in ('male', 'female', 'other'):
            raise ValueError("Gender must be 'male', 'female', or 'other'.")
        
        if not re.match(r'^[\w\.-]+@[\w\.-]+\.\w+$', email):
            raise ValueError("Invalid email address format.")
        
        

        self.username = username
        self.set_password(password)
        self.email = email
        self.gender = gender
        self.birthdate = birthdate

    
    def set_password(self, password):
        def hash_password(password:str) -> str:
            hashed = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
            return hashed.decode('utf-8')
        self.password_hash = hash_password(password)

    def check_password(self, password:str) -> bool:
        return bcrypt.checkpw(password.encode('utf-8'), self.password_hash.encode('utf-8'))

    
