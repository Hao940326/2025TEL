# 使用 PyTorch 官方 Docker 映像（含 CUDA）
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# 安裝系統套件：OpenCV 需要 libgl
RUN apt update && apt install -y libgl1 libglib2.0-0

# 設定工作目錄
WORKDIR /app

# 複製專案檔案進容器
COPY . .

# 安裝 Python 套件
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# 預設執行命令
CMD ["python", "main.py"]
