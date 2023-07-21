FROM python:3.9.7-slim
WORKDIR /EASYEDIT

COPY requirements.txt .

RUN /usr/local/bin/python3 -m pip install --upgrade pip

# RUN cat requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8080

COPY edit.py .

# 设置环境变量
ENV FLASK_APP=edit.py

# 启动命令
CMD ["flask", "run", "--host", "0.0.0.0"]