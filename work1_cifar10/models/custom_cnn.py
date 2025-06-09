# custom_cnn.py

from torch import nn

class CustomCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=nn.Conv2d(
            in_channels=3,
            out_channels=64,
            kernel_size=3,
            padding=1
        )        
        self.norm1=nn.BatchNorm2d(64)
        
        self.conv2=nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            padding=1
        )        
        self.norm2=nn.BatchNorm2d(128)
        
        self.conv3=nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=3,
            padding=1
        )
        self.norm3=nn.BatchNorm2d(256)
        self.pool=nn.MaxPool2d(2, 2)
        self.relu=nn.ReLU()
        self.dropout=nn.Dropout(0.2)
        self.fc1=nn.Linear(4*4*256,64)
        self.fc2=nn.Linear(64,10)
                
    def forward(self, x):
        x=self.pool(self.relu(self.norm1(self.conv1(x))))
        x=self.pool(self.relu(self.norm2(self.conv2(x))))
        x=self.pool(self.relu(self.norm3(self.conv3(x))))
        
        x=x.view(-1,4*4*256)
        x=self.relu(self.fc1(x))
        x=self.dropout(x)
        x=self.fc2(x)