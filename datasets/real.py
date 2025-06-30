# Ghiotto Andrea   2118418

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import glob, cv2

from datasets import pose as load_poses

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

        for i in range(len(self.imgs)):
            lab_path = self.pose_paths[i]
            self.poses[i] = load_poses.Pose(lab_path)()

        self.pad = nn.ZeroPad2d((0, 0, 18, 18))
        self.img_size = (256, 220)
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
        image = cv2.flip(image, 0)
       
        pose = np.copy(self.poses[idx])
        
        image = cv2.resize(image, self.img_size)
        image = self.pad(torch.Tensor(image))
        image = ( image / 255.0 ) - 0.5
        image_ = image[None] * np.pi
        sin, cos = torch.sin(image_), torch.cos(image_)
        
        return torch.cat([sin, cos]), torch.Tensor(image)[None], pose, img_path, self.img_labels[idx]