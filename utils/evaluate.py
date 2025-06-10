# evaluate.py
import torch

def evaluate(model, valloader, criterion, device):
    model.eval()
    avg_val_loss=0.0   
    total=0
    correct=0
    
    with torch.no_grad():
        for imgs, labels in valloader:
            imgs, labels=imgs.to(device), labels.to(device)
            
            outputs=model(imgs)
            loss=criterion(outputs, labels)
            
            _, preds=torch.max(outputs, dim=1)
            
            avg_val_loss+=loss.item()*imgs.size(0)
            total+=labels.size(0)
            correct+=(preds==labels).sum().item()
    avg_val_loss/=total
    accuracy=(correct*100)/total
    return avg_val_loss, accuracy, 
    
