# Ghiotto Andrea   2118418

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from tqdm import tqdm
import glob, cv2

from datasets import pose as load_poses

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

        self.synth = len(self.img_source)

        cont=0
        for i in range(len(self.imgs)):
            lab_path = self.pose_paths[i]
            self.poses[i] = load_poses.Pose(lab_path)()

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
        return self.gtquery(x, y, yaw_deg)

    def gtquery(self, x, y, yaw_deg):
        
        dist_matrix = torch.cdist(torch.Tensor([x,y]).unsqueeze(0), self.poses[:self.synth, :2].unsqueeze(0)).squeeze()
    
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

        image = self.pad(torch.Tensor(image))
        image = ( image / 255.0 ) - 0.5

        image_ = image[None] * np.pi
        sin, cos = torch.sin(image_), torch.cos(image_)
        
        return torch.cat([sin, cos]), torch.Tensor(image)[None], pose, img_path, self.img_labels[idx]