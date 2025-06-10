# CIFAR-10 CNN Classifier

📌 This is a simple CNN model to classify CIFAR-10 images using PyTorch.

## Project Structure

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

Author: 06unoh