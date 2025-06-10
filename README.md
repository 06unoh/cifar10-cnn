# CNN모델을 활용한 CIFAR-10

📌 This is a simple CNN model to classify CIFAR-10 images using PyTorch.

## 프로젝트 구조

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

## 실행법 (for Local PC)

### 1️⃣ Requirements 설치

```
pip install -r requirements.txt
```

### 2️⃣ 실행

```
python main.py
```

## 도커 실행법 (for Docker User)

### 1️⃣ 도커 이미지 빌드

```
docker build -t cifar10-cnn .
```

### 2️⃣ 컨테이너 실행

```
docker run --rm cifar10-cnn
```

---

## 데이터 셋

The CIFAR-10 dataset will be automatically downloaded by torchvision when running the script.  
No manual download is required.

---

## 결과

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