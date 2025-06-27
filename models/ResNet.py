# Ghiotto Andrea   2118418

import torch
import torch.nn as nn

class Bottleneck(nn.Module):

    expansion = 4

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Bottleneck, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)
        
        self.conv3 = nn.Conv2d(out_channels, out_channels*self.expansion, kernel_size=1, stride=1, padding=0)
        self.batch_norm3 = nn.BatchNorm2d(out_channels*self.expansion)
        
        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()
        
    def forward(self, x):
        identity = x.clone()
        x = self.relu(self.batch_norm1(self.conv1(x)))
        
        x = self.relu(self.batch_norm2(self.conv2(x)))
        
        x = self.conv3(x)
        x = self.batch_norm3(x)

        if self.i_downsample is not None:
            identity = self.i_downsample(identity)

        x+=identity
        x=self.relu(x)
        
        return x

class Block(nn.Module):

    expansion = 1

    def __init__(self, in_channels, out_channels, i_downsample=None, stride=1):
        super(Block, self).__init__()
       

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        self.i_downsample = i_downsample
        self.stride = stride
        self.relu = nn.ReLU()

    def forward(self, x):
      identity = x.clone()

      x = self.relu(self.batch_norm2(self.conv1(x)))
      x = self.batch_norm2(self.conv2(x))

      if self.i_downsample is not None:
          identity = self.i_downsample(identity)
      print(x.shape)
      print(identity.shape)
      x += identity
      x = self.relu(x)
      return x

class ResNet(nn.Module):
    
    def __init__(self, ResBlock=Bottleneck, layer_list=[3,8,36,3], num_channels=2):
        super().__init__()
        self.in_channels = 8
        
        self.conv1 = nn.Conv2d(num_channels, 8, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch_norm1 = nn.BatchNorm2d(8)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size = 3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(ResBlock, layer_list[0], planes=8)
        self.layer2 = self._make_layer(ResBlock, layer_list[1], planes=16, stride=2)
        self.layer3 = self._make_layer(ResBlock, layer_list[2], planes=32, stride=2)
        self.layer4 = self._make_layer(ResBlock, layer_list[3], planes=64, stride=2)

    def _make_layer(self, ResBlock, blocks, planes, stride=1):
        ii_downsample = None
        layers = []
        
        if stride != 1 or self.in_channels != planes*ResBlock.expansion:
            ii_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes*ResBlock.expansion, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes*ResBlock.expansion)
            )
            
        layers.append(ResBlock(self.in_channels, planes, i_downsample=ii_downsample, stride=stride))
        self.in_channels = planes*ResBlock.expansion
        
        for i in range(blocks-1):
            layers.append(ResBlock(self.in_channels, planes))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        out0 = self.relu(self.batch_norm1(self.conv1(x)))
        out1 = self.max_pool(out0)
        out2 = self.layer1(out1)
        out3 = self.layer2(out2)
        out4 = self.layer3(out3)
        out5 = self.layer4(out4)
        return [out2, out3, out4, out5]

class MLP(nn.Module):

    def __init__(self, input_dim=2048, embed_dim=768, bn=8):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.norm = nn.BatchNorm1d(bn*bn)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        b, c, h, w = x.shape
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
        self.encoder = ResNet()
        self.embed = MLP(256, 4)
        self.decoder = ResNetDecoder(feature_dims=[32, 64, 128, 256], out_channels=1, output_size=(256, 256))

    def forward(self, x, reco=False):
        #print(f"input shape: {x.shape}")
        out = self.encoder(x)
        feat = out[-1]
        #print(f"shape feature map dopo encoder: {feat.shape}")
        emb = self.embed(feat)
        #print(f"shape output mlp prima normalizzazione: {emb.shape}")
        embed = torch.nn.functional.normalize(emb.flatten(1), p=2, dim=1)
        #print(f"shape output mlp dopo normalizzazione: {embed.shape}")
        #embed = torch.nn.functional.normalize(self.embed(feat).flatten(1), p=2, dim=1)

        if reco:
            rec = self.decoder(out)
            return embed, rec

        return embed