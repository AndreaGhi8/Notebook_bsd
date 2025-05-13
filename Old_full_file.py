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



### MODEL
### MODEL
### MODEL



### TRAINING PIPELINE
### TRAINING PIPELINE
### TRAINING PIPELINE



### VISUALIZE TRAINING RESULTS
### VISUALIZE TRAINING RESULTS
### VISUALIZE TRAINING RESULTS



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