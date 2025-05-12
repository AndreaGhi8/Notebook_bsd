# Ghiotto Andrea   2118418

from Classes_and_functions import imports

def save_state(epoch, model, path):
    imports.os.makedirs(imports.os.path.dirname(path), exist_ok=True)

    imports.torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, path)
    
def load_state(model, path):
    checkpoint = imports.torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    return model

class Mlp(imports.nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=imports.nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = imports.nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = imports.nn.Linear(hidden_features, out_features)
        self.drop = imports.nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = imports.nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = imports.nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = imports.nn.Dropout(attn_drop)
        self.proj = imports.nn.Linear(dim, dim)
        self.proj_drop = imports.nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = imports.nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = imports.nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x
    
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=imports.nn.GELU, norm_layer=imports.nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        self.drop_path = imports.DropPath(drop_path) if drop_path > 0. else imports.nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = imports.to_2tuple(img_size)
        patch_size = imports.to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = imports.nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = imports.nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)
    
class PyramidVisionTransformer(imports.nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=imports.nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, F4=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.F4 = F4
        self.num_stages = num_stages

        dpr = [x.item() for x in imports.torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = imports.nn.Parameter(imports.torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = imports.nn.Dropout(p=drop_rate)

            block = imports.nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

            imports.trunc_normal_(pos_embed, std=.02)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, imports.nn.Linear):
            imports.trunc_normal_(m.weight, std=.02)
            if isinstance(m, imports.nn.Linear) and m.bias is not None:
                imports.nn.init.constant_(m.bias, 0)
        elif isinstance(m, imports.nn.LayerNorm):
            imports.nn.init.constant_(m.bias, 0)
            imports.nn.init.constant_(m.weight, 1.0)

    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    def forward_features(self, x):
        outs = []
        B = x.shape[0]

        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            x, (H, W) = patch_embed(x)
            if i == self.num_stages - 1:
                pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)

            x = pos_drop(x + pos_embed)
            for blk in block:
                x = blk(x, H, W)
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)

        if self.F4:
            x = x[3:4]

        return x

    def _conv_filter(state_dict, patch_size=16):
        out_dict = {}
        for k, v in state_dict.items():
            if 'patch_embed.proj.weight' in k:
                v = v.reshape((v.shape[0], 3, patch_size, patch_size))
            out_dict[k] = v

        return out_dict

class MLP(imports.nn.Module):
    """
    Linear Embedding
    """
    def __init__(self, input_dim=2048, embed_dim=768, bn=8):
        super().__init__()
        self.proj = imports.nn.Linear(input_dim, embed_dim)
        self.norm = imports.nn.BatchNorm1d(bn*bn)
        self.act = imports.nn.LeakyReLU()

    def forward(self, x):
        b, c, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        x = self.proj(x)
        x = self.act(x)
        return x

class SegFormerHead2(imports.nn.Module):
    def __init__(self, embedding_dim=128, in_channels_head=[32, 64, 128, 256], num_classes=1, img_size=512):
        super().__init__()
        self.img_size = img_size
        c1_in_channels, c2_in_channels, c3_in_channels, c4_in_channels = in_channels_head

        self.linear_fuse = imports.nn.Sequential(
            imports.nn.Conv2d(in_channels=embedding_dim*4, out_channels=embedding_dim, kernel_size=1, bias=False),
            imports.nn.BatchNorm2d(num_features=embedding_dim),
            imports.nn.ReLU(inplace=True)
        )

        self.linear_pred = imports.nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.linear_pred_1 = imports.nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.linear_pred_2 = imports.nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.linear_pred_3 = imports.nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        self.linear_pred_4 = imports.nn.Conv2d(embedding_dim, num_classes, kernel_size=1)
        
        self.linear_c4 = imports.nn.ConvTranspose2d(c4_in_channels, embedding_dim, kernel_size=32, stride=32)
        self.linear_c3 = imports.nn.ConvTranspose2d(c3_in_channels, embedding_dim, kernel_size=16, stride=16)
        self.linear_c2 = imports.nn.ConvTranspose2d(c2_in_channels, embedding_dim, kernel_size=8, stride=8)
        self.linear_c1 = imports.nn.ConvTranspose2d(c1_in_channels, embedding_dim, kernel_size=4, stride=4)
        

    def forward(self, inputs):
        c1, c2, c3, c4 = inputs
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4)
        _c3 = self.linear_c3(c3)
        _c2 = self.linear_c2(c2)
        _c1 = self.linear_c1(c1)
        x = self.linear_fuse(imports.torch.cat([_c4, _c3, _c2, _c1], dim=1))

        _c4 = self.linear_pred_4(_c4)
        _c3 = self.linear_pred_3(_c3)
        _c2 = self.linear_pred_2(_c2)
        _c1 = self.linear_pred_1(_c1)
        x   = self.linear_pred(x)

        return [x , _c1, _c2, _c3, _c4]
    
class Model(imports.nn.Module):
    def __init__(self):
        super().__init__()
        channels = [16, 32, 64, 128]
        self.encoder = PyramidVisionTransformer(in_chans=2, img_size=256, sr_ratios=[8, 4, 2, 1], patch_size=4, embed_dims=channels)
        self.embed = MLP(128, 4)
        self.decoder = SegFormerHead2(in_channels_head=channels, img_size=256, num_classes=1)
        
    def forward(self, x, reco=False):
        out = self.encoder(x)
        embed= imports.torch.nn.functional.normalize(self.embed(out[-1]).flatten(1), p=2, dim=1)
        
        if reco:
            rec = self.decoder(out)
            return embed, rec
            
        return embed