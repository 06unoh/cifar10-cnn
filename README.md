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

## How to Run

### 1️⃣ Install requirements

```
pip install -r requirements.txt
```

### 2️⃣ Run the main script

```
python main.py
```

## Docker Usage

### 1️⃣ Build the docker image

```
docker build -t cifar10-cnn .
```

### 2️⃣ Run the container

```
docker run --rm cifar10-cnn
```

---

## Dataset

The CIFAR-10 dataset will be automatically downloaded by torchvision when running the script.  
No manual download is required.

---

## Results

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

---
Author: 06unoh