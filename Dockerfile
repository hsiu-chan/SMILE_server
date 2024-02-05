# 阶段1：构建阶段
FROM python:3.10 AS builder

WORKDIR /root

COPY . /root/

RUN apt-get update
RUN apt-get install -y libgl1-mesa-glx

RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# 阶段2：运行阶段
FROM python:3.10-slim



# 安装 libgl1-mesa-glx
RUN apt-get update && \
    apt-get install -y libgl1-mesa-glx libglib2.0-0


# 複製整個應用程式到容器
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /root/app /app

# 设置环境变量 PYTHONPATH
ENV PYTHONPATH=/usr/local/lib/python3.10/site-packages

# 设置 LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/mesa

# volume
VOLUME /app

# 開放應用程式所需的端口
EXPOSE 8888

# 设置工作目录
WORKDIR /app
# 定義啟動應用程式的命令
CMD ["python", "main.py"]
