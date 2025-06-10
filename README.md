# CNN모델을 활용한 CIFAR-10

📌 This is a simple CNN model to classify CIFAR-10 images using PyTorch.

## - 프로젝트 구조

```
models/
    custom_cnn.py
utils/
    train.py
    validate.py
    test.py
    visualize.py
main.py
requirements.txt
Dockerfile
```
---
## - 실행법 (for Local PC)

### 1️⃣ Requirements 설치

```
pip install -r requirements.txt
```

### 2️⃣ 실행

```
python main.py
```
---
## - 도커 실행법 (for Docker User)

### 1️⃣ 도커 이미지 빌드

```
docker build -t cifar10-cnn .
```

### 2️⃣ 컨테이너 실행

```
docker run --rm cifar10-cnn
```

---

## - 데이터셋

본 프로젝트는 CIFAR-10 데이터셋을 사용합니다.  
CIFAR-10은 10개의 클래스(비행기, 자동차, 새, 고양이 등)에 대한 60,000장의 32x32 컬러 이미지로 구성된 공개 이미지 데이터셋입니다.

데이터는 별도로 다운로드할 필요 없이, 스크립트 실행 시 `torchvision` 라이브러리를 통해 자동으로 다운로드됩니다.

---

## - 결과

Example accuracy after training for 10 epochs:

```
Train Accuracy: ~85%  
Validation Accuracy: ~83%
```

*Example visualization of predictions:*

```
[Class: Cat] → Predicted: Cat  
[Class: Plane] → Predicted: Plane  
[Class: Dog] → Predicted: Dog  
...
```

---
Author: 06unoh