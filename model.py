import torch
from torch import nn


class MNISTCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=3,
                padding=1
        
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=2,
                stride=2
            ),
            nn.Flatten(),
            nn.Linear(
                in_features=32*7*7,
                out_features=10
            )
        )
    def forward(self,x):
        return self.network(x)


#Debug

model = MNISTCNN()
x = torch.randn(32,1,28,28)

output = model(x)

print(output.shape)