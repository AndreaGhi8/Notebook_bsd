import torch.nn as nn
import torch

from torch import Tensor
from typing import Type

class BasicBlock(nn.Module):

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        expansion: int = 1,
        downsample: nn.Module = None
    ) -> None:
        super(BasicBlock, self).__init__()
        self.expansion = expansion
        self.downsample = downsample
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels*self.expansion,
            kernel_size=3,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels*self.expansion)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(
        self,
        img_channels: int,
        num_layers: int,
        block: Type[BasicBlock],
        num_classes: int = 1000
    ) -> None:
        super(ResNet, self).__init__()
        if num_layers == 18:
            layers = [2, 2, 2, 2]
            self.expansion = 1
        self.in_channels = 64
        self.conv1 = nn.Conv2d(
            in_channels=img_channels,
            out_channels=self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*self.expansion, num_classes)

    def _make_layer(
        self,
        block: Type[BasicBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.in_channels,
                    out_channels*self.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm2d(out_channels * self.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels, out_channels, stride, self.expansion, downsample
            )
        )
        self.in_channels = out_channels * self.expansion
        for i in range(1, blocks):
            layers.append(block(
                self.in_channels,
                out_channels,
                expansion=self.expansion
            ))
        return nn.Sequential(*layers)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out1 = self.maxpool(x)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)

        return [out2, out3, out4, out5]
    
class MLP(nn.Module):

    def __init__(self, input_dim=2048, embed_dim=768, bn=8):
        super().__init__()
        self.pool_size = bn
        self.proj = nn.Linear(input_dim, embed_dim)
        self.norm = nn.BatchNorm1d(bn*bn)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        x = nn.functional.adaptive_avg_pool2d(x, (self.pool_size, self.pool_size))
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.proj(x)
        x = self.act(x)
        return x
    
class ResNetDecoder(nn.Module):
    
    def __init__(self, feature_dims=[256, 512, 1024, 2048], out_channels=1, output_size=(256, 256)):
        super(ResNetDecoder, self).__init__()
        self.output_size = output_size
        
        self.up1 = nn.ConvTranspose2d(feature_dims[3], feature_dims[2], kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(feature_dims[2]*2, feature_dims[2], kernel_size=3, padding=1)
        self.pred4 = nn.Conv2d(feature_dims[2], out_channels, kernel_size=1)

        self.up2 = nn.ConvTranspose2d(feature_dims[2], feature_dims[1], kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(feature_dims[1]*2, feature_dims[1], kernel_size=3, padding=1)
        self.pred3 = nn.Conv2d(feature_dims[1], out_channels, kernel_size=1)

        self.up3 = nn.ConvTranspose2d(feature_dims[1], feature_dims[0], kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(feature_dims[0]*2, feature_dims[0], kernel_size=3, padding=1)
        self.pred2 = nn.Conv2d(feature_dims[0], out_channels, kernel_size=1)

        self.up4 = nn.ConvTranspose2d(feature_dims[0], feature_dims[0] // 2, kernel_size=2, stride=2)
        self.conv4 = nn.Conv2d(feature_dims[0] // 2, feature_dims[0] // 2, kernel_size=3, padding=1)
        self.pred1 = nn.Conv2d(feature_dims[0] // 2, out_channels, kernel_size=1)

        self.out_conv = nn.Conv2d(feature_dims[0] // 2, out_channels, kernel_size=1)

    def forward(self, features):
        f1, f2, f3, f4 = features

        x = self.up1(f4)
        x = self.conv1(torch.cat([x, f3], dim=1))
        pred4 = nn.functional.interpolate(self.pred4(x), size=self.output_size, mode='bilinear', align_corners=False)

        x = self.up2(x)
        x = self.conv2(torch.cat([x, f2], dim=1))
        pred3 = nn.functional.interpolate(self.pred3(x), size=self.output_size, mode='bilinear', align_corners=False)

        x = self.up3(x)
        x = self.conv3(torch.cat([x, f1], dim=1))
        pred2 = nn.functional.interpolate(self.pred2(x), size=self.output_size, mode='bilinear', align_corners=False)

        x = self.up4(x)
        x = self.conv4(x)
        pred1 = nn.functional.interpolate(self.pred1(x), size=self.output_size, mode='bilinear', align_corners=False)

        final = nn.functional.interpolate(self.out_conv(x), size=self.output_size, mode='bilinear', align_corners=False)

        return [final, pred1, pred2, pred3, pred4]

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = ResNet(img_channels=2, num_layers=18, block=BasicBlock, num_classes=1)
        self.embed = MLP(512, 4)
        self.decoder = ResNetDecoder(feature_dims=[64, 128, 256, 512], out_channels=1, output_size=(256, 256))

    def forward(self, x, reco=False):
        out = self.encoder(x)
        feat = out[-1]
        embed = torch.nn.functional.normalize(self.embed(feat).flatten(1), p=2, dim=1)

        if reco:
            rec = self.decoder(out)
            return embed, rec

        return embed