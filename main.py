# main.py

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import torchvision
import torchvision.transforms as transforms

from models.custom_cnn import CustomCNN
from utils.train import train
from work1_cifar10.utils.evaluate import evaluate
from utils.test import test
from utils.visualize import visualize

train_tf=transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])
test_tf=transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
])

temp_ds=torchvision.datasets.CIFAR10('./', True, download=True)

generator=torch.Generator().manual_seed(42)
train_indices, val_indices=random_split(range(len(temp_ds)), [45000, 5000], generator=generator)

trainset=Subset(torchvision.datasets.CIFAR10('./', True, train_tf, download=True), train_indices)
valset=Subset(torchvision.datasets.CIFAR10('./', True, test_tf, download=True), val_indices)
testset=torchvision.datasets.CIFAR10('./', False, test_tf, download=True)

trainloader=DataLoader(trainset, 32, True)
valloader=DataLoader(valset, 32, False)
testloader=DataLoader(testset, 32, False)

# main training
if __name__=="__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=CustomCNN().to(device)
    criterion=nn.CrossEntropyLoss()
    optimizer=optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(40):
        avg_loss=train(model, trainloader, criterion, optimizer, device)
        avg_val_loss, val_acc=evaluate(model, valloader, criterion, device)
        print(f'epoch: {epoch+1}, Train Loss: {avg_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {val_acc:.2f}%')
        
    test(model, testloader, device)   
    visualize(model, testloader)
    
    