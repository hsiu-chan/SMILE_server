-- 創建數據庫
CREATE SCHEMA IF NOT EXISTS `smile_user_data`;

-- 創建用戶
CREATE USER 'user'@'%' IDENTIFIED BY 'password';

-- 授予用戶對數據庫的所有權限
GRANT ALL PRIVILEGES ON `smile_user_data`.* TO 'user'@'%';

-- 確保權限更新
FLUSH PRIVILEGES;


-- 選擇資料庫
USE smile_user_data;

-- 儲存用戶的資料表
CREATE TABLE IF NOT EXISTS `smile_user_data`.`users` (
  `id` INT NOT NULL AUTO_INCREMENT,
  `username` VARCHAR(50) NOT NULL,
  `password` VARCHAR(255) NOT NULL,
  `email` VARCHAR(255) NOT NULL,
  PRIMARY KEY (`id`)
) COMMENT = '用戶的資料表';
