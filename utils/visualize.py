# visualize.py
import matplotlib.pyplot as plt
import numpy as np
import torch

def visualize_prediction(model, testloader):
    mean = torch.tensor([0.4914, 0.4822, 0.4465]).numpy()
    std = torch.tensor([0.2470, 0.2435, 0.2616]).numpy()
    
    dataiter=iter(testloader)
    imgs, labels=next(dataiter)
    
    model.eval()
    with torch.no_grad():
        outputs=model(imgs)
        _, preds=torch.max(outputs, dim=1)
    
    fig, axes=plt.subplots(3, 3, figsize=(8, 8))
    axes=axes.flatten()
    for i in range(9):
        img=imgs[i].cpu().permute(1,2,0).numpy() # (C, H, W) -> (H, W, C) :for imshow 
        img=img*std+mean
        img=np.clip(img, 0 ,1)
        axes[i].imshow(img)
        axes[i].set_title(f'Label: {labels[i].item()},  Pred: {preds[i].item()}')
        axes[i].axis('off')
    plt.tight_layout()
    plt.show()
    

    