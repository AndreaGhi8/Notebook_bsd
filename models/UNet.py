# Ghiotto Andrea   2118418

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNetEncoder(nn.Module):

    def __init__(self, n_channels, bilinear=False):
        super(UNetEncoder, self).__init__()
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1
        self.down4 = Down(512, 1024 // factor)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        return x1, x2, x3, x4, x5
    
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

class UNetDecoder(nn.Module):

    def __init__(self, n_classes, bilinear=False, output_size=(256, 256)):
        super(UNetDecoder, self).__init__()
        self.output_size = output_size

        factor = 2 if bilinear else 1
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        self.outc = OutConv(64, n_classes)

        self.pred1 = nn.Conv2d(64, n_classes, kernel_size=1)
        self.pred2 = nn.Conv2d(128, n_classes, kernel_size=1)
        self.pred3 = nn.Conv2d(256, n_classes, kernel_size=1)
        self.pred4 = nn.Conv2d(512, n_classes, kernel_size=1)

    def forward(self, *args):
        if len(args) == 1 and isinstance(args[0], (tuple, list)):
            out = args[0]
            if len(out) == 5:
                if out[0].shape[1] == 64:
                    x1, x2, x3, x4, x5 = out
                else:
                    x5, x4, x3, x2, x1 = out
        else:
            x5, x4, x3, x2, x1 = args

        x = self.up1(x5, x4)
        pred4 = nn.functional.interpolate(self.pred4(x), size=self.output_size, mode='bilinear', align_corners=False)
        x = self.up2(x, x3)
        pred3 = nn.functional.interpolate(self.pred3(x), size=self.output_size, mode='bilinear', align_corners=False)
        x = self.up3(x, x2)
        pred2 = nn.functional.interpolate(self.pred2(x), size=self.output_size, mode='bilinear', align_corners=False)
        x = self.up4(x, x1)
        pred1 = nn.functional.interpolate(self.pred1(x), size=self.output_size, mode='bilinear', align_corners=False)
        final = self.outc(x)
        final = nn.functional.interpolate(final, size=self.output_size, mode='bilinear', align_corners=False)

        return [final, pred1, pred2, pred3, pred4]

class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = UNetEncoder(n_channels=2)
        self.embed = self.embed = MLP(1024, 4, 16)
        self.decoder = UNetDecoder(n_classes=1, bilinear=False)

    def forward(self, x, reco=False):
        out = self.encoder(x)
        feat = out[-1]
        emb = self.embed(feat)
        embed = torch.nn.functional.normalize(emb.flatten(1), p=2, dim=1)

        if reco:
            rec = self.decoder(out)
            return embed, rec

        return embed