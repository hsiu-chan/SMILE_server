



# 阶段2：运行阶段
FROM python:3.10

# 工作目錄
WORKDIR /root

# 安裝依賴
COPY requirements.txt /root/requirements.txt

RUN apt-get update &&\
    apt-get install -y libgl1-mesa-glx

RUN pip install --upgrade pip &&\
    pip install -r requirements.txt


# 设置 LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu/mesa


# 複製整個應用程式到容器
COPY . /root/

# volume
VOLUME /root/app

# 開放應用程式所需的端口
EXPOSE 8888



# 定義啟動應用程式的命令
CMD ["python", "app/main.py"]
