# Ghiotto Andrea   2118418

import os, cv2, glob, math, time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm
from scipy.spatial.transform import Rotation as R
from timm.models.layers import DropPath, to_2tuple, trunc_normal_

## For visualization
from IPython import display

from sklearn.neighbors import KDTree
import random

from torch.utils.tensorboard import SummaryWriter



### DATASET
### DATASET
### DATASET



class Pose:
    def __init__(self, label_path):
        with open(label_path, "r") as file:
            line = file.readline()[:-1].split()
            self.x, self.y, self.z, self.r, self.p,  self.yaw, = float(line[0]), float(line[1]), float(line[2]), float(line[3]), float(line[4]), float(line[5])
            self.yaw = -self.yaw * 180 / math.pi
    def __call__(self):
        return torch.Tensor([self.x, self.y, self.yaw])
    
class SonarDescriptorRealDataset(Dataset):
    def __init__(self, datapath, database4val=None):
        self.img_source = glob.glob("Datasets/placerec_trieste_updated/imgs/*")
        self.img_labels = glob.glob("Datasets/placerec_trieste_updated/pose/*")

        self.img_source.sort()
        self.img_labels.sort()
        self.img_source = np.array(self.img_source)
        self.img_labels = np.array(self.img_labels)

        self.imgs = self.img_source
        self.pose_paths = self.img_labels
        self.poses = np.zeros((len(self.img_source), 3))

        self.synth = len(self.img_source)

        cont=0
        for i in range(len(self.imgs)):
            lab_path = self.pose_paths[i]
            self.poses[i] = Pose(lab_path)()

        self.pad = nn.ZeroPad2d((0, 0, 28, 28))
        self.img_size = (256, 200)
        self.min_dx, self.min_dy = 335, -458
        self.poses[:, 0]-=self.min_dx
        self.poses[:, 1]-=self.min_dy
        self.poses[:, :2]*=10

        self.poses = torch.Tensor(self.poses)
        
    def __len__(self):
        return len(self.imgs)


    def crop_and_resize_image(self, image:np.ndarray, rotation:float) -> np.ndarray:
        shift = int(1536*rotation / 360)
        image = image[:, (512-shift):(1024-shift)]
        return image

    def __getitem__(self, idx):

        img_path = self.imgs[idx]

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)

        image = cv2.flip(image, 0) ## flip vertically
       
        pose = np.copy(self.poses[idx])
        
        image = cv2.resize(image, self.img_size)

        ## Tensorize, pad and normalize image
        image = self.pad(torch.Tensor(image))
        image = ( image / 255.0 ) - 0.5

        ## Augment image dimensionality using sine and cosine
        image_ = image[None] * np.pi
        sin, cos = torch.sin(image_), torch.cos(image_)
        
        return torch.cat([sin, cos]), torch.Tensor(image)[None], pose, img_path, self.img_labels[idx]

class SonarDescriptorDatasetFull(Dataset):
    def __init__(self, datapath, database4val=None):
        self.img_source = glob.glob(os.path.join(datapath, "imgs", "*"))
        self.img_labels = glob.glob(os.path.join(datapath, "poses", "*"))
        self.img_source.sort()
        self.img_labels.sort()
        self.img_source = np.array(self.img_source)#[7:]
        self.img_labels = np.array(self.img_labels)#[7:]
        
        self.training = database4val is None

        # KEEP by Andrea from the original notebook
        if self.training:
            self.idxs = np.arange(0, len(self.img_source), 1, dtype=int)
            np.random.shuffle(self.idxs)
            self.train_idxs_num = self.idxs.shape[0]
            self.train_idxs = self.idxs[:self.train_idxs_num]
    
            self.img_source = self.img_source[self.train_idxs]
            self.img_labels = self.img_labels[self.train_idxs]
        else:
            self.idxs = np.arange(0, len(self.img_source), 1, dtype=int)
            np.random.shuffle(self.idxs)
            self.valid_idxs_num = self.idxs.shape[0]
            self.valid_idxs = self.idxs[:self.valid_idxs_num]
            
            self.img_source = self.img_source[self.valid_idxs]
            self.img_labels = self.img_labels[self.valid_idxs]

            
        
        if False and self.training:     # REMOVE "False and" WHEN NEED TO TRAIN ALSO REAL
            idxs = np.arange(0, 1700, 1, dtype=int)
            np.random.shuffle(idxs)
            idxs = idxs[:1700]
            
            self.realimg_source = glob.glob("Datasets/placerec_trieste_updated/imgs/*")
            self.realimg_source.sort()
            self.realimg_source = np.array(self.realimg_source)[idxs]

            self.realimg_labels = glob.glob("Datasets/placerec_trieste_updated/pose/*")
            self.realimg_labels.sort()
            self.realimg_labels = np.array(self.realimg_labels)[idxs]
            
            self.imgs       = np.concatenate((self.img_source, self.realimg_source))
            self.pose_paths = np.concatenate((self.img_labels, self.realimg_labels))
            
            self.descriptors=[]
            
            self.poses = np.zeros((len(self.img_source)+len(self.realimg_source), 3))
            
        else:
            #self.poses = np.zeros((len(self.img_source), 3))
            #self.apply_random_rot = True
            #idxs = np.arange(0, len(self.img_source), 1, dtype=int)
            #np.random.shuffle(idxs)
            #idxs = idxs[:600]
            #self.img_source = self.img_source[idxs]
            #self.img_labels = self.img_labels[idxs]
            #self.imgs = self.img_source
            #self.pose_paths = self.img_labels

            # KEEP by Andrea from the original notebook
            self.imgs = self.img_source
            self.poses = np.zeros((len(self.img_source), 3))
            self.pose_paths = self.img_labels

        
        self.synth = len(self.img_source)

        #print("synth", self.synth) # REMOVE THIS PRINT

        if not self.training:
            self.rotations = np.zeros(len(self.img_labels))
        
        cont=0
        for i in range(len(self.imgs)):
            lab_path = self.pose_paths[i]
            pose = Pose(lab_path)()
            self.poses[i] = pose

        self.pad = nn.ZeroPad2d((0, 0, 28, 28))
        # self.img_size = (256, 220)
        self.img_size = (256, 200)
        self.min_dx, self.min_dy = 335, -458
        self.poses[:, 0]-=self.min_dx
        self.poses[:, 1]-=self.min_dy
        self.poses[:, :2]*=10
        
        if self.training:
            self.poses = torch.Tensor(self.poses)
        else:
            self.closest_poses = self.correlate_poses(database4val)
            
        
    def __len__(self):
        return len(self.imgs)

    def computeDescriptors(self, net):
        self.descriptors=[]
        print("computing dataset descriptors")
        net.eval()
        if not self.training:
            self.shifts=np.array([0])
        with torch.no_grad():
            for idx in tqdm(range(self.synth)):
                img_path = self.imgs[idx]
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
                image_ = np.copy(image)
                image_ = self.pad(torch.Tensor(image_))
                image_ = torch.Tensor(image_)
                image_ = ( image_ / 255.0 ) - 0.5
                image_ = image_[None] * np.pi
                sin, cos = torch.sin(image_), torch.cos(image_)
                image_ = torch.cat([sin, cos]).cuda()[None]
                descriptor = net(image_, reco=False)[0, :].detach().cpu().numpy()
                self.descriptors.append(descriptor)
        print("descriptors computed!")

    def correlate_poses(self, database4val):
        self.closest_indices = np.zeros(self.poses.shape[0])
        for idx in range(self.poses.shape[0]):
            self.closest_indices[idx] = database4val.gtquery_synth(self.poses[idx])
        self.closest_indices = self.closest_indices.astype(int)

    def gtquery_synth(self, synthpose):
        x,y,yaw_deg = synthpose
        # yaw_deg = (90 + yaw_deg) % 360        # REMOVE THE # AND CHECK WHEN NEED TO TRAIN ALSO REAL
        #print("synthpose:", x, y, yaw_deg)
        return self.gtquery(x, y, yaw_deg)
    def gtquery_real(self, realpose):
        x,y,yaw_deg = realpose
        #yaw_deg = (90+yaw_deg)%360
        #print("realpose:", x, y, yaw_deg)
        return self.gtquery(x, y, yaw_deg)

    def gtquery(self, x, y, yaw_deg):
        
        dist_matrix = torch.cdist(torch.Tensor([x,y]).unsqueeze(0), self.poses[:self.synth, :2].unsqueeze(0)).squeeze()  # Shape: (N, M)
    
        # closest_index = torch.argmin(dist_matrix, dim=-1)
    
        # print("closest_index in poses", closest_index)
        # print("XY distance:", math.sqrt((x-self.poses[closest_index, 0])**2 + (y-self.poses[closest_index, 1])**2), closest_index.item())

        _, cand_indx = torch.topk(dist_matrix, 5, dim=-1, largest=False, sorted=True)
        #print("pos", dist_matrix[cand_indx])    # REMOVE THIS PRINT

        candidates = self.poses[:self.synth, 2][cand_indx]
        diff_yaw = abs(candidates-yaw_deg)
        #print("diff_yaw", diff_yaw)             # REMOVE THIS PRINT

        min_yaw_idx = torch.argmin(diff_yaw, dim=-1)
        #print("min_yaw_idx", min_yaw_idx)       # REMOVE THIS PRINT

        closest_index = cand_indx[min_yaw_idx]
        closest_index = closest_index.item()
        
        return closest_index
   
    def query(self, query_descriptor):
        self.norms = np.zeros(len(self.descriptors))
        for i in range(len(self.descriptors)):
            self.norms[i] = np.sum((self.descriptors[i] - query_descriptor)**2)
        return self.norms.argmin()

    def crop_and_resize_image(self, image:np.ndarray, rotation:float) -> np.ndarray:
        image=image
        return image #cv2.resize(image, self.img_size)

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
       
        pose = np.copy(self.poses[idx])
        
        image = cv2.resize(image, self.img_size)

        image = self.pad(torch.Tensor(image))
        image = torch.Tensor(image)
        image = (image / 255.0) - 0.5

        image_ = image[None] * np.pi
        sin, cos = torch.sin(image_), torch.cos(image_)
        
        return torch.cat([sin, cos]), torch.Tensor(image)[None], pose, img_path, self.img_labels[idx] if idx<self.synth else "aaa", 1 if idx<self.synth else 0

def start_plot(train_data, sonar_radius=50, figsize = (15,10)):
    global plt
    plt.figure(figsize=figsize)   

    ax = plt.gca()
    ax.set_xlim([0, train_data.poses[:, 0].max()+sonar_radius])
    ax.set_ylim([0, train_data.poses[:, 1].max()+sonar_radius])

def plot_synth_poses_train(td, color="blue"):
    global plt
    plt.scatter(td.poses[:td.synth, 0], td.poses[:td.synth, 1], c=color, marker='o', linestyle='None', s =1)

def plot_synth_poses_val(vd, color="red"):
    global plt
    plt.scatter(vd.poses[:, 0], vd.poses[:, 1], c=color, marker='o', linestyle='None', s = 1)

def parse_pose(pose):
    x, y, Y_deg = np.array(pose, copy=True)
    # Y_deg = 90 + Y_deg
    Y_deg %= 360
    Y = Y_deg * math.pi / 180
    return x, y, Y, Y_deg

def scatter_point(x, y, color, label=None):
    global plt
    if label is None:
        plt.scatter(x, y, c=color, s = 20.51)
    else:
        plt.scatter(x, y, c=color, s = 20.51, label=label)
        
def scatter_orientation(x, y, Y, color, rad=50):
    global plt
    dy, dx = rad*math.cos(Y), rad*math.sin(Y)
    plt.arrow(x, y, dx, dy, color=color)

def sector_mask(shape,centre,radius, Y_deg):
    """
    Return a boolean mask for a circular sector. The start/stop angles in  
    `angle_range` should be given in clockwise order.
    """
    angle_range = (Y_deg-60, Y_deg+60)
    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)

    # ensure stop angle > start angle
    if tmax < tmin:
            tmax += 2*np.pi

    # convert cartesian --> polar coordinates
    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin

    # wrap angles between 0 and 2*pi
    theta %= (2*np.pi)

    # circular mask
    circmask = r2 <= radius*radius

    # angular mask
    anglemask = theta <= (tmax-tmin)

    return (circmask*anglemask).astype(int)

def generate_interference_mask(x1, y1, Y1, Y1_deg, x2, y2, Y2, Y2_deg, sonar_map_size = (325+50, 295+50)):
    
    mask1 = sector_mask(sonar_map_size,np.array([x1, y1]).astype(int),50,Y1_deg).T
    mask2 = sector_mask(sonar_map_size,np.array([x2, y2]).astype(int),50,Y2_deg).T

    mask_and = np.logical_and(mask1, mask2) ## intersection
    mask_xor = np.logical_xor(mask1, mask2) ## U/intersection 
    
    R3 = 2*abs(math.cos((Y1-Y2)/2))
    mask_and = mask_and*R3
    
    mask3 = mask_and + mask_xor
    
    union = mask1.sum() + mask2.sum()
    intersection = mask_and.sum()
    iou = intersection/union if union>0 else 0

    return mask3, iou

def generate_interference_mask_transparent(x1, y1, Y1, Y1_deg, x2, y2, Y2, Y2_deg, sonar_map_size = (325+50, 295+50)):
    
    mask1 = sector_mask(sonar_map_size,np.array([x1, y1]).astype(int),50,Y1_deg).T
    mask2 = sector_mask(sonar_map_size,np.array([x2, y2]).astype(int),50,Y2_deg).T

    mask_and = np.logical_and(mask1, mask2) ## intersection
    mask_xor = np.logical_xor(mask1, mask2) ## U/intersection 
    
    R3 = 2*abs(math.cos((Y1-Y2)/2))
    mask_and = mask_and*R3
    
    mask3 = mask_and + mask_xor
    
    union = mask1.sum() + mask2.sum()
    intersection = mask_and.sum()
    iou = intersection/union if union>0 else 0

    return mask3*0, iou
        
def scatter_real_orientation(x, y, Y, color, rad=50):
    global plt
    dx, dy = rad*math.cos(Y), rad*math.sin(Y)
    plt.arrow(x, y, dy, dx, color=color)

def plot_real_poses(rd, color="pink"):
    global plt
    plt.scatter(rd.poses[:, 0], rd.poses[:, 1], c=color, marker='o', linestyle='None', s =1)
    for i in range(0, rd.poses.shape[0], 5):
    # for i in range(0, 100, 1):
        q_x, q_y, q_Y_deg = rd.poses[i, :]
        scatter_real_orientation(q_x, q_y, (q_Y_deg*np.pi/180) % np.pi, "mediumturquoise")

def gtquery(database, x, y, yaw_deg):
    dist_matrix = torch.cdist(torch.Tensor([x,y]).unsqueeze(0), database.poses[:database.synth, :2].unsqueeze(0)).squeeze()  # Shape: (N, M)

    # closest_index = torch.argmin(dist_matrix, dim=-1)

    # print("closest_index in poses", closest_index)
    # print("XY distance:", math.sqrt((x-self.poses[closest_index, 0])**2 + (y-self.poses[closest_index, 1])**2), closest_index.item())

    _, cand_indx = torch.topk(dist_matrix, 5, dim=-1, largest=False, sorted=True)
    # print("pos", dist_matrix[cand_indx])    # REMOVE THIS PRINT

    candidates = database.poses[:database.synth, 2][cand_indx]
    diff_yaw = abs(candidates-yaw_deg)%360
    # print("diff_yaw", diff_yaw)             # REMOVE THIS PRINT

    min_yaw_idx = torch.argmin(diff_yaw, dim=-1)
    # print("min_yaw_idx", min_yaw_idx)       # REMOVE THIS PRINT

    closest_index = cand_indx[min_yaw_idx]
    closest_index = closest_index.item()
    
    return closest_index



### MODEL
### MODEL
### MODEL



def save_state(epoch, model, path):

    # ADD by Andrea
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            }, path)
    
def load_state(model, path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    epoch = checkpoint['epoch']
    return model

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

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

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

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
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
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
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape

        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)


class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, F4=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.F4 = F4
        self.num_stages = num_stages

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        for i in range(num_stages):
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size=patch_size if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            cur += depths[i]

            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

            trunc_normal_(pos_embed, std=.02)

        # init weights
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

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
        """ convert patch embedding weight from manual patchify + linear proj to conv"""
        out_dict = {}
        for k, v in state_dict.items():
            if 'patch_embed.proj.weight' in k:
                v = v.reshape((v.shape[0], 3, patch_size, patch_size))
            out_dict[k] = v

        return out_dict

class MLP(nn.Module):
    """
    Linear Embedding
    """
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


class SegFormerHead2(nn.Module):
    """
    SegFormer: Simple and Efficient Design for Semantic Segmentation with Transformers
    """
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

        ############## MLP decoder on C1-C4 ###########
        n, _, h, w = c4.shape

        _c4 = self.linear_c4(c4)#.reshape(n, -1, c4.shape[2], c4.shape[3])
        # print("c4", c4.shape, "_c4", _c4.shape)
        _c3 = self.linear_c3(c3)#.reshape(n, -1, c3.shape[2], c3.shape[3])
        # print("c3", c3.shape, "_c3", _c3.shape)
        _c2 = self.linear_c2(c2)#.reshape(n, -1, c2.shape[2], c2.shape[3])
        # print("c2", c2.shape, "_c2", _c2.shape)
        _c1 = self.linear_c1(c1)#.reshape(n, -1, c1.shape[2], c1.shape[3])
        # print("c1", c1.shape, "_c1", _c1.shape)
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
        channels = [16, 32, 64, 128]
        self.encoder = PyramidVisionTransformer(in_chans=2, img_size=256, sr_ratios=[8, 4, 2, 1], patch_size=4, embed_dims=channels)
        self.embed = MLP(128, 4)
        self.decoder = SegFormerHead2(in_channels_head=channels, img_size=256, num_classes=1)
        
    def forward(self, x, reco=False):
        out = self.encoder(x)
        embed= torch.nn.functional.normalize(self.embed(out[-1]).flatten(1), p=2, dim=1)
        
        if reco:
            rec = self.decoder(out)
            return embed, rec
            
        return embed



### TRAINING PIPELINE
### TRAINING PIPELINE
### TRAINING PIPELINE



def calcEmbedMatrix(vectors):
    # Compute differences between vectors
    diff = vectors.unsqueeze(1) - vectors.unsqueeze(0)  # Shape: (N, N, C)

    # Compute squared distances
    squared_dist = torch.sum(diff ** 2, dim=2)  # Shape: (N, N)
    squared_dist = torch.clamp(squared_dist, 1e-8, squared_dist.max().item()) ### the derivative at sqrt(0) is infinity!!! must clamp
    return torch.sqrt(squared_dist) #euclidean_dist

def generate_sonar_map(pose, flag):
    rad = 150
    if not flag:
        # return np.zeros((325, 295)).T
        return np.zeros((325+rad*2, 295+rad*2)).T
    x,y,Y_deg = pose[:3].clone()
    x+=rad
    y+=rad
    Y_deg = 90 - Y_deg
    Y_deg = 90 - Y_deg
    center = np.array([x, y]).astype(int)
    # mask = sector_mask((325, 295),center,rad,(Y_deg-60,Y_deg+60)).T
    mask = sector_mask((325+rad*2, 295+rad*2),center,rad,Y_deg).T
    return mask

def sonar_overlap_distance_matrix(gtposes, mode):
    iou_matrix = np.zeros((gtposes.shape[0], gtposes.shape[0]))
    sonar_images = [generate_sonar_map(gtposes[i], mode[i]) for i in range(gtposes.shape[0])]

    ## the matrix will be symmetric! Thus we only compute
    ## the upper triangular matrix and then fill the other
    for i in range(gtposes.shape[0]):
        for j in range(i+1, gtposes.shape[0]):
            ##### INTERFERENCE
            mask_and = np.logical_and(sonar_images[i], sonar_images[j]) ## intersection
            mask_xor = np.logical_xor(sonar_images[i], sonar_images[j]) ## U/intersection
            Y1 = (90-gtposes[i][2])*np.pi/180
            Y2 = (90-gtposes[j][2])*np.pi/180
            
            R3 = 2*abs(math.cos((Y1-Y2)/2))
            mask_and = mask_and*R3
            
            #mask3 = mask_and + mask_xor
            
            union = sonar_images[i].sum() + sonar_images[j].sum()
            intersection = mask_and.sum()
            iou = intersection/union if union >0 else 0
            iou_matrix[i][j] = iou

    ## fill the lower triangular matrix
    iou_matrix = np.maximum( iou_matrix, iou_matrix.transpose() )
    
    ## we didn't compute the diagonal (trivially, it's always maximum, thus one)
    np.fill_diagonal(iou_matrix, 1.0)

    ## we want it to be minimum (0) where most similar
    return torch.Tensor(1-iou_matrix)

def correlate_poses_topk(matrix1, matrix2, k=1):
    # Compute Euclidean distance between each pair of poses
    dist_matrix = torch.cdist(matrix1, matrix2)  # Shape: (N, M)

    # Find the index of the closest pose in matrix2 for each pose in matrix1
    closest_indices = torch.argsort(dist_matrix, dim=1)[:, :k]

    return closest_indices

def computeAverageMetricError(pred_indices, gt_indices, train_data, k=1):
    prp = train_data.poses[pred_indices, :2]   ## pred_retrieved_poses
    grp = train_data.poses[gt_indices, :2]     ## gt_retrieved_poses
    # print("prp", prp.shape, "grp", grp.shape)
    if k==1:
        diff = prp.unsqueeze(1) - grp.unsqueeze(1)
    else:
        # print(grp.unsqueeze(1).expand((-1, k, 2).shape))
        diff = prp - grp.unsqueeze(1).expand((-1, k, 2))
    
    squared_dist = torch.sqrt(torch.sum(diff ** 2, dim=2))
    # print(squared_dist.max())

    if k>1:
        squared_dist = torch.min(squared_dist, dim=1)[0]
    
    return squared_dist.mean()



### VISUALIZE TRAINING RESULTS
### VISUALIZE TRAINING RESULTS
### VISUALIZE TRAINING RESULTS


def process(q_idx, net, train_data, val_data, plot=True):

    if plot:
        start_plot(train_data)

    # plot_real_poses(train_data, "pink")
    # plot_synth_poses_train(train_data, "blue")
    # plot_synth_poses_val(val_data, "red")
    
    ## Randomly select an "input index" from the validation dataset
    # query_idx = 497
    
    ## compute query image descriptor and reconstruction by the network
    q_image_a, q_image, q_pose, _, _, _ = val_data[q_idx]
    q_image_a = q_image_a[None].cuda()
    if plot:
        q_desc, (q_image_r, _, _, _, _)  = net(q_image_a, reco=True)
    else:
        q_desc = net(q_image_a, reco=False)[0, :]#.detach().cpu().numpy()
    q_desc = q_desc.detach().cpu().numpy()
    
    q_x, q_y, q_Y, q_Y_deg = parse_pose(q_pose)
    if plot:
        scatter_point(q_x, q_y, 'magenta', label="val pose (query)")
        scatter_orientation(q_x, q_y, q_Y, "magenta")
    
    ## compute
    # colors=[[0, 0, 1] for _ in train_data.img_source]

    # plot the dots (positions of train and val data)
    if plot:
        plt.scatter(train_data.poses[:train_data.synth, 0], train_data.poses[:train_data.synth, 1], c="blue", marker='o', linestyle='None', s =1, label="training set positions")
        plt.scatter(val_data.poses[:, 0], val_data.poses[:, 1], c="red", marker='o', linestyle='None', s =1, label="validation set positions")
        # plot_synth_poses_train(train_data, colors)
    
    ##### BLUE POINT, PREDICTION FROM THE DATABASE #####
    train_data.apply_random_rot = False
    minidx = train_data.query(q_desc)
    min_pose = train_data[minidx][2]
    min_x, min_y, min_Y, min_Y_deg = parse_pose(min_pose)
    if plot:
        scatter_point(min_x, min_y, 'gold', label="predicted pose")
        scatter_orientation(min_x, min_y, min_Y, "gold")
    
    ##### GREEN POINT, GROUND TRUTH CLOSEST POINT IN THE DATABASE #####
    # gt_pose = train_data[val_data.closest_indices[q_idx]][2]

    
    gt_pose_idx = gtquery(train_data, q_x, q_y, q_Y_deg)
    gt_pose = train_data[gt_pose_idx][2]

    gt_x, gt_y, gt_Y, gt_Y_deg = parse_pose(gt_pose)
    if plot:
                   
        scatter_orientation(gt_x, gt_y, gt_Y, "green")
        scatter_point(gt_x, gt_y, "green", label="database gt closest pose")
    
    ##### INTERFERENCE
    mask3, iou = generate_interference_mask(min_x, min_y, min_Y, min_Y_deg, q_x, q_y, q_Y, q_Y_deg)
    # mask3, iou = generate_interference_mask(gt_x, gt_y, gt_Y, gt_Y_deg, q_x, q_y, q_Y, q_Y_deg)

    loca_error=np.linalg.norm(gt_pose[:2]-min_pose[:2], ord=2)/10

    orie_error = gt_Y_deg - min_Y_deg.item()
    orie_error = np.abs((orie_error + 180) % 360 - 180)

    gt_closest_image = train_data[val_data.closest_indices[q_idx]][1]

    # Print yaw difference --- TO REMOVE
    # print(f'GT degree: {gt_Y_deg}, query degree: {q_Y_deg}')
    
    if plot:
        plt.imshow(mask3, cmap="gray")
        print("iou:", iou)
        plt.legend(loc="lower right")
    
        plt.figure()
        
        f, axarr = plt.subplots(1, 3, figsize=(15, 15))
        axarr[0].set_title("query image")
        axarr[1].set_title("closest image from database")
        axarr[2].set_title("reconstructed query image")
        
        axarr[0].imshow(q_image.detach().cpu().numpy()[0, :, :], cmap='gray')
        # axarr[1].imshow(min_img.numpy()[0, :, :], cmap='gray')
        axarr[1].imshow(gt_closest_image[0, :, :], cmap='gray')
        axarr[2].imshow(q_image.detach().cpu().numpy()[0, :, :], cmap='gray')   # it was [0, 0, :, :] MODIFIED BY ANDREA to avoid this error: IndexError: too many indices for array: array is 3-dimensional, but 4 were indexed
        
    return loca_error, orie_error



### PROCESS REAL IMAGES
### PROCESS REAL IMAGES
### PROCESS REAL IMAGES



def parse_real_pose(pose):
    x, y, Y = np.array(pose, copy=True)
    Y %= np.pi
    Y_deg = Y * 180 / math.pi
    Y_deg = Y_deg #Y * 180 / math.pi#
    Y_deg -= 180
    # Y_deg %= 360
    Y = Y_deg * np.pi / 180
    Y %= np.pi
    # print(Y, Y_deg)
    return x, y, Y, Y_deg

def process_real(q_idx, net, train_data, real_data):
       
    start_plot(train_data)

    plot_synth_poses_train(train_data, "blue")
    
    ## compute query image descriptor and reconstruction by the network
    q_image_a, q_image, q_pose, _, _ = real_data[q_idx]
        
    q_image_a = q_image_a[None].cuda()
    q_desc, (q_image_r, _, _, _, _)  = net(q_image_a, reco=True)
    q_desc = q_desc.detach().cpu().numpy()
    
    q_x, q_y, q_Y, q_Y_deg = parse_real_pose(q_pose)
    scatter_point(q_x, q_y, 'pink', label="val pose (query)")
    scatter_orientation(q_x, q_y, q_Y, "pink")

    print(q_x, q_y, q_Y, q_Y_deg)
    
    ## compute
    train_data.apply_random_rot = False
    minidx = train_data.query(q_desc)
    _, min_img, min_pose, min_img_path, min_lab_path, _ = train_data[minidx]
    
    rad = 50
    
    ##### BLUE POINT, PREDICTION FROM THE DATABASE #####
    min_x, min_y, min_Y, min_Y_deg = parse_pose(min_pose)
    scatter_point(min_x, min_y, 'blue', label="predicted pose")
    scatter_orientation(min_x, min_y, min_Y, "blue")
    
    ##### GREEN POINT, GROUND TRUTH CLOSEST POINT IN THE DATABASE #####
    gt_pose_idx = train_data.gtquery_real(q_pose)
    gt_pose     = train_data[gt_pose_idx][2]
    gt_img_path = train_data.imgs[gt_pose_idx//len(train_data)]
    gt_image = cv2.cvtColor(cv2.imread(gt_img_path), cv2.COLOR_BGR2GRAY)
    gt_x, gt_y, gt_Y, gt_Y_deg = parse_pose(gt_pose)
    scatter_orientation(gt_x, gt_y, gt_Y, "green")
    # scatter_point(x2, y2, "green", label="database predicted closest pose")
    
    # center = np.array([x, y]).astype(int)
    # mask2 = sector_mask((325+50, 295+50),center,50,(Y2_deg-60,Y2_deg+60)).T
    
    ##### INTERFERENCE
    mask3, iou = generate_interference_mask(min_x, min_y, min_Y, min_Y_deg, gt_x, gt_y, gt_Y, gt_Y_deg)
    plt.imshow(mask3, cmap="gray")
    print("iou:", iou)
    plt.legend(loc="lower right")
    plot_real_poses(real_data, "pink")
    
    plt.figure()
    
    f, axarr = plt.subplots(1, 3, figsize=(15, 15))
    axarr[0].set_title("query image")
    axarr[1].set_title("closest image")
    axarr[2].set_title("ground truth 360 closest synthetic image")
    
    axarr[0].imshow(q_image.detach().cpu().numpy()[0, :, :], cmap='gray')
    axarr[1].imshow(min_img.numpy()[0, :, :], cmap='gray')
    axarr[2].imshow(gt_image, cmap='gray')
    
    print("localization error Upper: ", np.linalg.norm(q_pose[:2]-min_pose[:2], ord=2)/10, "meters")
    # print("localization error Norma: ", np.linalg.norm(min_pose[:2]-gt_pose[:2], ord=2)/10, "meters")



### PROCESS ONLY REAL IMAGES
### PROCESS ONLY REAL IMAGES
### PROCESS ONLY REAL IMAGES



class SonarDescriptorOnlyRealDataset(Dataset):
    def __init__(self, database4val=None):
        self.training = database4val is None
        
        self.img_source = glob.glob("Datasets/placerec_trieste_updated/imgs/*")
        self.img_labels = glob.glob("Datasets/placerec_trieste_updated/pose/*")

        self.img_source.sort()
        self.img_labels.sort()
        if self.training:
            self.img_source = np.array(self.img_source)[:710]
            self.img_labels = np.array(self.img_labels)[:710]
        else:
            self.img_source = np.array(self.img_source)[715:1500]
            self.img_labels = np.array(self.img_labels)[715:1500]

        self.imgs = self.img_source
        self.pose_paths = self.img_labels
        self.poses = np.zeros((len(self.img_source), 3))

        # ADD by Andrea
        self.synth = len(self.img_source)

        cont=0
        for i in range(len(self.imgs)):
            lab_path = self.pose_paths[i]
            self.poses[i] = Pose(lab_path)()

        self.pad = nn.ZeroPad2d((0, 0, 28, 28))
        self.img_size = (256, 200)
        self.min_dx, self.min_dy = 335, -458
        self.poses[:, 0]-=self.min_dx
        self.poses[:, 1]-=self.min_dy
        self.poses[:, :2]*=10

        self.poses = torch.Tensor(self.poses)

        if not self.training:
            self.closest_poses = self.correlate_poses(database4val)
    
        
    def __len__(self):
        return len(self.imgs)

    def correlate_poses(self, database4val):
        self.closest_indices = np.zeros(self.poses.shape[0])
        for idx in range(self.poses.shape[0]):
            self.closest_indices[idx] = database4val.gtquery_real(self.poses[idx])
        self.closest_indices = self.closest_indices.astype(int)


    def crop_and_resize_image(self, image:np.ndarray, rotation:float) -> np.ndarray:
        shift = int(1536*rotation / 360)
        image = image[:, (512-shift):(1024-shift)]
        return image

    def computeDescriptors(self, net):
        self.descriptors=[]
        print("computing dataset descriptors")
        net.eval()
        if not self.training:
            self.shifts=np.array([0])
        with torch.no_grad():
            for idx in tqdm(range(self.synth)):
                img_path = self.imgs[idx]
                image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
                image_ = np.copy(image)
                image_ = self.pad(torch.Tensor(image_))
                image_ = torch.Tensor(image_)
                image_ = ( image_ / 255.0 ) - 0.5
                image_ = image_[None] * np.pi
                sin, cos = torch.sin(image_), torch.cos(image_)
                image_ = torch.cat([sin, cos]).cuda()[None]
                descriptor = net(image_, reco=False)[0, :].detach().cpu().numpy()
                self.descriptors.append(descriptor)
        print("descriptors computed!")

    def gtquery_real(self, realpose):
        x,y,yaw_deg = realpose
        yaw_deg = (90+yaw_deg)%360
        #print("realpose:", x, y, yaw_deg)
        return self.gtquery(x, y, yaw_deg)

    def gtquery(self, x, y, yaw_deg):
        
        dist_matrix = torch.cdist(torch.Tensor([x,y]).unsqueeze(0), self.poses[:self.synth, :2].unsqueeze(0)).squeeze()  # Shape: (N, M)
    
        closest_index = torch.argmin(dist_matrix, dim=-1)
    
        return closest_index
   
    def query(self, query_descriptor):
        self.norms = np.zeros(len(self.descriptors))
        for i in range(len(self.descriptors)):
            self.norms[i] = np.sum((self.descriptors[i] - query_descriptor)**2)
        return self.norms.argmin()

    def __getitem__(self, idx):

        img_path = self.imgs[idx]

        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
       
        pose = np.copy(self.poses[idx])
        
        image = cv2.resize(image, self.img_size)

        ## Tensorize, pad and normalize image
        image = self.pad(torch.Tensor(image))
        image = ( image / 255.0 ) - 0.5

        ## Augment image dimensionality using sine and cosine
        image_ = image[None] * np.pi
        sin, cos = torch.sin(image_), torch.cos(image_)
        
        return torch.cat([sin, cos]), torch.Tensor(image)[None], pose, img_path, self.img_labels[idx]

def processReal(q_idx, net, train_data, val_data, plot=True):

    if plot:
        start_plot(train_data)

        plt.scatter(train_data.poses[:train_data.synth, 0], train_data.poses[:train_data.synth, 1], c="blue", marker='o', linestyle='None', s =1, label="training set positions")
        plt.scatter(val_data.poses[:, 0], val_data.poses[:, 1], c="red", marker='o', linestyle='None', s =1, label="validation set positions")

    # plot_real_poses(train_data, "pink")
    # plot_synth_poses_train(train_data, "blue")
    # plot_synth_poses_val(val_data, "red")
    
    ## Randomly select an "input index" from the validation dataset
    # query_idx = 497
    
    ## compute query image descriptor and reconstruction by the network
    q_image_a, q_image, q_pose, _, _ = val_data[q_idx]
    q_image_a = q_image_a[None].cuda()
    if plot:
        q_desc, (q_image_r, _, _, _, _)  = net(q_image_a, reco=True)
    else:
        q_desc = net(q_image_a, reco=False)[0, :]#.detach().cpu().numpy()
    q_desc = q_desc.detach().cpu().numpy()
    
    #q_x, q_y, q_Y, q_Y_deg = parse_pose(q_pose)
    #q_pose = val_data.poses[realidx].numpy()
    q_x, q_y, q_Y_deg = q_pose
    q_Y_deg = (90+q_Y_deg)%360
    q_Y = q_Y_deg * np.pi/180
    #scatter_orientation(q_x, q_y, q_Y, "orange", rad=50)
    if plot:
        scatter_point(q_x, q_y, 'magenta', label="val pose (query)")
        scatter_orientation(q_x, q_y, q_Y, "magenta")
    
    ##### GOLD POINT, PREDICTION FROM THE DATABASE #####
    #min_x, min_y, min_Y, min_Y_deg = parse_pose(min_pose)
    #q_pose = val_data.poses[realidx].numpy()
    minidx = train_data.query(q_desc)
    _, min_img, min_pose, min_img_path, min_lab_path = train_data[minidx]

    min_x, min_y, min_Y_deg = min_pose
    min_Y_deg = (90+min_Y_deg)%360
    min_Y = min_Y_deg * np.pi/180

    if plot:
        scatter_point(min_x, min_y, 'gold', label="predicted pose")
        scatter_orientation(min_x, min_y, min_Y, "gold")
    
    ##### GREEN POINT, GROUND TRUTH CLOSEST POINT IN THE DATABASE #####
    gt_pose = train_data[val_data.closest_indices[q_idx]][2]

    # gt_pose_idx = gtquery(train_data, q_pose)
    # gt_pose = train_data[gt_pose_idx][2]

    
    #gt_x, gt_y, gt_Y, gt_Y_deg = parse_pose(gt_pose)
    gt_x, gt_y, gt_Y_deg = gt_pose
    gt_Y_deg = (90+gt_Y_deg)%360
    gt_Y = gt_Y_deg * np.pi/180
    if plot:
        # base = (val_data.closest_indices[q_idx]//13)*13
        # cont = 0.2
        # for i in range(base, base+13):
        #     #print(train_data[i][2])
        #     x3,y3, Y3, Y3_deg = parse_pose(train_data[i][2])
        #     scatter_orientation(x3, y3, Y3, [cont, 0, 0])
        #     scatter_point(x3, y3, "red")
        #     cont += 0.035
            
        # scatter_orientation(gt_x, gt_y, gt_Y, "green")
        scatter_point(gt_x, gt_y, "green", label="database gt closest pose")
    
    ##### INTERFERENCE
    mask3, iou = generate_interference_mask(min_x, min_y, min_Y, min_Y_deg, q_x, q_y, q_Y, q_Y_deg)
    # mask3, iou = generate_interference_mask(gt_x, gt_y, gt_Y, gt_Y_deg, q_x, q_y, q_Y, q_Y_deg)

    loca_error=np.linalg.norm(gt_pose[:2]-min_pose[:2], ord=2)/10

    orie_error = gt_Y_deg - min_Y_deg.item()
    orie_error = np.abs((orie_error + 180) % 360 - 180)

    # print(min_x, gt_x)
    # print(min_y, gt_y)
    # print(gt_pose[:2], min_pose[:2])
    
    if plot:
        plt.imshow(mask3, cmap="gray")
        print("iou:", iou)
        plt.legend(loc="lower right")
    
        plt.figure()
        
        f, axarr = plt.subplots(1, 3, figsize=(15, 15))
        axarr[0].set_title("query image")
        axarr[1].set_title("closest image from database")
        axarr[2].set_title("reconstructed query image")
        
        axarr[0].imshow(q_image.detach().cpu().numpy()[0, :, :], cmap='gray')
        axarr[1].imshow(min_img.numpy()[0, :, :], cmap='gray')
        axarr[2].imshow(q_image_r.detach().cpu().numpy()[0, 0, :, :], cmap='gray')
        
    return loca_error, orie_error