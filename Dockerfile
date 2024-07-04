# 阶段1：构建阶段
FROM python:3.10 AS builder

WORKDIR /root

COPY requirements.txt /root/

RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY . /root/


# 阶段2：运行阶段
FROM python:3.10-slim

# 安装 libgl1-mesa-glx 和 libglib2.0-0
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1-mesa-glx libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# 複製整個應用程式到容器
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

COPY --from=builder /root/app /app

# 设置环境变量 PYTHONPATH
ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages

# 设置 LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/mesa

# volume
VOLUME /app

# 開放應用程式所需的端口
EXPOSE 8777

# 工作目錄
WORKDIR /app

# 定義啟動應用程式的命令，使用 Gunicorn
CMD ["gunicorn", "--bind", "0.0.0.0:8777", "main:app"]
# CMD ["python", "main.py"]
