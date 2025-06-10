# test.py
import torch

def test(model, testloader, device):
    model.eval()   
    total=0
    correct=0
    
    with torch.no_grad():
        for imgs, labels in testloader:
            imgs, labels=imgs.to(device), labels.to(device)
            
            outputs=model(imgs)
            
            _, preds=torch.max(outputs, dim=1)
            total+=imgs.size(0)
            correct+=(preds==labels).sum().item()
            
    accuracy=(correct*100)/total
    print(f'Accuracy: {accuracy:.2f}%')
            