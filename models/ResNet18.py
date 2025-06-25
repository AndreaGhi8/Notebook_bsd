import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class MLP(nn.Module):

    def __init__(self, input_dim=512, embed_dim=4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = self.pool(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.norm(x)
        x = self.act(x)
        return x

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.fc = MLP(512, 4)

        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.backbone(x)
        x = self.fc(x)
        x = F.normalize(x, p=2, dim=1)
        return x    