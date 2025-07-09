# Ghiotto Andrea   2118418

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import glob, os, cv2
from natsort import natsorted

from datasets import pose as load_poses
from utils import visualizer as parser

class SonarDescriptorDatasetFull(Dataset):
    def __init__(self, datapath, database4val=None):
        self.img_source = glob.glob(os.path.join(datapath, "imgs", "*"))
        self.img_labels = glob.glob(os.path.join(datapath, "poses", "*"))
        self.img_source.sort()
        self.img_labels.sort()
        self.img_source = np.array(self.img_source)
        self.img_labels = np.array(self.img_labels)
        
        self.training = database4val is None
        self.tags = []

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

        if self.training:     # ADD "False and" when need to train without real
            idxs = np.arange(0, 1500, 1, dtype=int)
            np.random.shuffle(idxs)
            idxs = idxs[:1500]
            
            self.realimg_source = glob.glob("Datasets/placerec_trieste_updated/imgs/*")
            self.realimg_source = natsorted(glob.glob("Datasets/placerec_trieste_updated/imgs/*"))
            self.realimg_source = np.array(self.realimg_source)[idxs]

            self.realimg_labels = glob.glob("Datasets/placerec_trieste_updated/pose/*")
            self.realimg_labels = natsorted(glob.glob("Datasets/placerec_trieste_updated/pose/*"))
            self.realimg_labels = np.array(self.realimg_labels)[idxs]
            
            self.imgs       = np.concatenate((self.img_source, self.realimg_source))
            self.pose_paths = np.concatenate((self.img_labels, self.realimg_labels))

            for i in range(len(self.realimg_source)):
                self.tags.append(f"real")
            
            self.descriptors=[]
            
            self.poses = np.zeros((len(self.img_source)+len(self.realimg_source), 3))
            
        else:
            self.imgs = self.img_source
            self.poses = np.zeros((len(self.img_source), 3))
            self.pose_paths = self.img_labels
        
        self.synth = len(self.img_source)

        if not self.training:
            self.rotations = np.zeros(len(self.img_labels))
        
        cont=0
        for i in range(len(self.imgs)):
            lab_path = self.pose_paths[i]
            pose = load_poses.Pose(lab_path)()
            self.poses[i] = pose

        self.pad_synth = nn.ZeroPad2d((0, 0, 28, 28))
        self.pad_real = nn.ZeroPad2d((0, 0, 18, 18))
        self.img_size_synth = (256, 200)
        self.img_size_real = (256, 220)
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
                if image_.shape == (200, 256):
                    self.pad = nn.ZeroPad2d((0, 0, 28, 28))
                if image_.shape == (220, 256):
                    self.pad = nn.ZeroPad2d((0, 0, 18, 18))
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
        # yaw_deg = (90 + yaw_deg) % 360
        #print("synthpose:", x, y, yaw_deg)
        return self.gtquery(x, y, yaw_deg)
    
    def gtquery_real(self, realpose):
        x,y,yaw_deg = realpose
        #yaw_deg = (90+yaw_deg)%360
        #print("realpose:", x, y, yaw_deg)
        return self.gtquery(x, y, yaw_deg) 

    def gtquery(self, x, y, yaw_deg):
        
        dist_matrix = torch.cdist(torch.Tensor([x,y]).unsqueeze(0), self.poses[:self.synth, :2].unsqueeze(0)).squeeze()

        _, cand_indx = torch.topk(dist_matrix, 5, dim=-1, largest=False, sorted=True)
        
        candidates = self.poses[:self.synth, 2][cand_indx]
        candidates = torch.Tensor([parser.parse_pose([0,0,cand])[3] for cand in candidates])

        diff_yaw = torch.min(abs(candidates-yaw_deg), abs(360-abs(candidates-yaw_deg)))

        min_yaw_idx = torch.argmin(diff_yaw, dim=-1)

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
        return image

    def __getitem__(self, idx):
        img_path = self.imgs[idx]
        image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2GRAY)
        
        real_idx = self.synth - idx
        if real_idx < len(self.tags):
            image = cv2.flip(image, 0)
            self.pad = self.pad_real
            self.img_size = self.img_size_real
        else:
            self.pad = self.pad_synth
            self.img_size = self.img_size_synth
       
        pose = np.copy(self.poses[idx])
        
        image = cv2.resize(image, self.img_size)

        image = self.pad(torch.Tensor(image))
        image = torch.Tensor(image)
        image = (image / 255.0) - 0.5

        image_ = image[None] * np.pi
        sin, cos = torch.sin(image_), torch.cos(image_)

        label = self.img_labels[idx]
        if real_idx < len(self.tags):
            label = "aaa"
        
        mode = 1
        if real_idx < len(self.tags):
            mode = 0
        
        return torch.cat([sin, cos]), torch.Tensor(image)[None], pose, img_path, label, mode