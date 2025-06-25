# Ghiotto Andrea   2118418

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

class Block(nn.Module):

    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = input + self.drop_path(x)
        return x

class ConvNext(nn.Module):

    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], dims=[96, 192, 384, 768], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        features = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            features.append(x)
        return features

    def forward(self, x):
        return self.forward_features(x)

class LayerNorm(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

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

class Decoder(nn.Module):
    def __init__(self, feature_dims=[32, 64, 128, 256], out_channels=1, output_size=(256, 256)):
        super().__init__()
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
        pred4 = F.interpolate(self.pred4(x), size=self.output_size, mode='bilinear', align_corners=False)

        x = self.up2(x)
        x = self.conv2(torch.cat([x, f2], dim=1))
        pred3 = F.interpolate(self.pred3(x), size=self.output_size, mode='bilinear', align_corners=False)

        x = self.up3(x)
        x = self.conv3(torch.cat([x, f1], dim=1))
        pred2 = F.interpolate(self.pred2(x), size=self.output_size, mode='bilinear', align_corners=False)

        x = self.up4(x)
        x = self.conv4(x)
        pred1 = F.interpolate(self.pred1(x), size=self.output_size, mode='bilinear', align_corners=False)

        final = F.interpolate(self.out_conv(x), size=self.output_size, mode='bilinear', align_corners=False)

        return [final, pred1, pred2, pred3, pred4]
    
class WrapDecoder(nn.Module):

    def __init__(self, decoder, return_list=False):
        super().__init__()
        self.decoder = decoder
        self.return_list = return_list

    def forward(self, x):
        x = self.decoder(x)
        if self.return_list:
            return [x]
        else:
            return x

class Model(nn.Module):
    
    def __init__(self):
        super().__init__()
        channels = [32, 64, 128, 256]
        self.encoder = ConvNext(in_chans=2, depths=[2, 2, 2, 2], dims=channels)
        self.embed = MLP(256, 8)
        self.decoder = WrapDecoder(
            Decoder(feature_dims=channels, out_channels=1),
            return_list=False
        )

    def forward(self, x, reco=False):
        out = self.encoder.forward_features(x)
        feat = out[-1]
        embed = torch.nn.functional.normalize(self.embed(feat).flatten(1), p=2, dim=1)

        if reco:
            rec = self.decoder(out)
            return embed, rec

        return embed