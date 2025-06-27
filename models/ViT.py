# Ghiotto Andrea   2118418

import torch
from torch import nn
import torch.nn.functional as F

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):

    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):

    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

class ViT(nn.Module):

    def __init__(self, *, image_size=256, patch_size=32, num_classes=1, dim=256, depth=6, heads=8, mlp_dim=1024, pool = 'cls', channels = 2, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
            nn.LayerNorm(dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Linear(dim, num_classes)

        self.decoder = nn.Sequential(
            nn.Linear(dim, 1024),
            nn.GELU(),
            nn.Linear(1024, 256 * 256),
            nn.Sigmoid(),
            Rearrange('b (h w) -> b 1 h w', h=256, w=256)
        )

    def forward(self, img, reco=False):
        x = self.to_patch_embedding(img)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)
        x = self.transformer(x)

        if reco:
            embed = x[:, 1:]
            embed_flat = embed.view(b, -1)
            embed_mean = embed.mean(dim=1)
            recon_image = self.decoder(embed_mean)
            rec_list = [recon_image for _ in range(5)]
            return embed_flat, rec_list

        embed = x[:, 1:]
        embed_flat = embed.view(b, -1)
        spatial_dim = int(embed_flat.shape[1] // 256)
        spatial_size = int(spatial_dim ** 0.5)
        fake_feat = embed_flat.view(b, 256, spatial_size, spatial_size)

        out = [
            F.interpolate(fake_feat, scale_factor=0.25, mode='bilinear', align_corners=False),
            F.interpolate(fake_feat, scale_factor=0.5, mode='bilinear', align_corners=False),
            F.interpolate(fake_feat, scale_factor=2.0, mode='bilinear', align_corners=False),
            fake_feat
        ]
        return out
    
class MLP(nn.Module):

    def __init__(self, input_dim=2048, embed_dim=768, bn=8):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.act = nn.LeakyReLU()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.proj(x)
        x = self.act(x)
        return x

class SegFormerHead2(nn.Module):
    
    def __init__(self, embedding_dim=128, in_channels_head=[32, 64, 128, 256], num_classes=1, img_size=512):
        super().__init__()
        self.img_size = img_size
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels_head

        self.linear_fuse = nn.Sequential(
            nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(num_features=embedding_dim),
            nn.ReLU(inplace=True)
        )

        self.linear_pred = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.linear_pred_1 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.linear_pred_2 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.linear_pred_3 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.linear_pred_4 = nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        
        self.linear_c4 = nn.ConvTranspose2d(c4_in_channels, embedding_dim, kernel_size=32, stride=32)
        self.linear_c3 = nn.ConvTranspose2d(c3_in_channels, embedding_dim, kernel_size=16, stride=16)
        self.linear_c2 = nn.ConvTranspose2d(c2_in_channels, embedding_dim, kernel_size=8, stride=8)
        self.linear_c1 = nn.ConvTranspose2d(c1_in_channels, embedding_dim, kernel_size=4, stride=4)
    
    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4)
        _c3 = self.linear_c3(c3)
        _c2 = self.linear_c2(c2)
        _c1 = self.linear_c1(c1)

        target_size = _c4.shape[2:]

        _c3 = F.interpolate(_c3, size=target_size, mode='bilinear', align_corners=False)
        _c2 = F.interpolate(_c2, size=target_size, mode='bilinear', align_corners=False)
        _c1 = F.interpolate(_c1, size=target_size, mode='bilinear', align_corners=False)

        x = self.linear_fuse(torch.cat([_c4, _c3, _c2, _c1], dim=1))

        _c4 = self.linear_pred_4(_c4)
        _c3 = self.linear_pred_3(_c3)
        _c2 = self.linear_pred_2(_c2)
        _c1 = self.linear_pred_1(_c1)
        x   = self.linear_pred(x)

        return [x , _c1, _c2, _c3, _c4]
    
class Model(nn.Module):

    def __init__(self):
        super().__init__()
        self.encoder = ViT()
        self.embed = MLP(256, 4)
        self.decoder = SegFormerHead2(in_channels_head=[256, 256, 256, 256], img_size=256, num_classes=1)

    def forward(self, x, reco=False):
        if x.shape[2] != 256 or x.shape[3] != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
        
        out = self.encoder(x)
        feat = out[-1]
        embed = torch.nn.functional.normalize(self.embed(feat).flatten(1), p=2, dim=1)

        if reco:
            rec = self.decoder(out)
            return embed, rec

        return embed