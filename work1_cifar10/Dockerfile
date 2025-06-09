# PyTorch 2.6.0 + CUDA 12.4 기반
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

# 작업 디렉토리 설정
WORKDIR /app

# 현재 프로젝트 전체 복사
COPY . /app

# 최신 pip 사용
RUN pip install --upgrade pip

# 정확한 버전으로 패키지 설치
RUN pip install torch==2.6.0+cu124 torchvision==0.21.0+cu124 \
    -f https://download.pytorch.org/whl/torch_stable.html && \
    pip install matplotlib==3.10.0 numpy==2.0.2

# 메인 실행 명령 → python main.py 실행
CMD ["python", "main.py"]
