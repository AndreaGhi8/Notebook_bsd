import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights

class MLP(nn.Module):

    def __init__(self, input_dim=2048, embed_dim=768, bn=8):
        super().__init__()
        self.linear = nn.Linear(input_dim, embed_dim)
        self.norm = nn.BatchNorm1d(embed_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.linear(x)
        x = x.flatten(0, 1)
        x = self.norm(x)
        x = self.act(x)
        x = x.view(b, -1, x.size(-1))
        x = x.flatten(1)
        return x

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        resnet = resnet18(weights=ResNet18_Weights.DEFAULT)
        resnet.avgpool = nn.Identity()
        resnet.fc = nn.Identity()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.fc = MLP(512, 4)

        for param in self.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        #print(f"input shape: {x.shape}")
        x = self.backbone(x)
        #print(f"shape feature map dopo backbone: {x.shape}")
        x = self.fc(x)
        #print(f"shape output mlp prima normalizzazione: {x.shape}")
        x = F.normalize(x, p=2, dim=1)
        return x