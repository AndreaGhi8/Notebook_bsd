# Ghiotto Andrea   2118418

import matplotlib.pyplot as plt
import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from Classes_and_functions.Training import visualize_results

def start_plot(train_data, sonar_radius=50, figsize = (15,10)):
    plt.figure(figsize=figsize)

    ax = plt.gca()
    ax.set_xlim([0, train_data.poses[:, 0].max()+sonar_radius])
    ax.set_ylim([0, train_data.poses[:, 1].max()+sonar_radius])

def plot_synth_poses_train(td, color="blue"):
    plt.scatter(td.poses[:td.synth, 0], td.poses[:td.synth, 1], c=color, marker='o', linestyle='None', s =1)

def plot_synth_poses_val(vd, color="red"):
    plt.scatter(vd.poses[:, 0], vd.poses[:, 1], c=color, marker='o', linestyle='None', s = 1)

def parse_pose(pose):
    x, y, Y_deg = np.array(pose, copy=True)
    Y_deg_ = Y_deg + 90
    Y_deg_ %= 360
    Y = Y_deg_ * math.pi / 180
    return x, y, Y, Y_deg_

def scatter_point(x, y, color, label=None):
    if label is None:
        plt.scatter(x, y, c=color, s = 20.51)
    else:
        plt.scatter(x, y, c=color, s = 20.51, label=label)
        
def scatter_orientation(x, y, Y_r, color, rad=50):
    dy, dx = rad*math.cos(Y_r), rad*math.sin(Y_r)
    plt.arrow(x, y, dx, dy, color=color)

def sector_mask(shape,centre,radius, Y_deg):
    angle_range = (Y_deg-60, Y_deg+60)
    x,y = np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = np.deg2rad(angle_range)
    if tmax < tmin:
            tmax += 2*np.pi

    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = np.arctan2(x-cx,y-cy) - tmin
    theta %= (2*np.pi)
    circmask = r2 <= radius*radius
    anglemask = theta <= (tmax-tmin)

    return (circmask*anglemask).astype(int)

def generate_interference_mask(x1, y1, Y1, Y1_deg, x2, y2, Y2, Y2_deg, sonar_map_size = (325+50, 295+50)):
    
    mask1 = sector_mask(sonar_map_size,np.array([x1, y1]).astype(int),50,Y1_deg).T
    mask2 = sector_mask(sonar_map_size,np.array([x2, y2]).astype(int),50,Y2_deg).T

    mask_and = np.logical_and(mask1, mask2)
    mask_xor = np.logical_xor(mask1, mask2)
    
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

    mask_and = np.logical_and(mask1, mask2)
    mask_xor = np.logical_xor(mask1, mask2)
    
    R3 = 2*abs(math.cos((Y1-Y2)/2))
    mask_and = mask_and*R3
    
    mask3 = mask_and + mask_xor
    
    union = mask1.sum() + mask2.sum()
    intersection = mask_and.sum()
    iou = intersection/union if union>0 else 0

    return mask3*0, iou
        
def scatter_real_orientation(x, y, Y, color, rad=50):
    dx, dy = rad*math.cos(Y), rad*math.sin(Y)
    plt.arrow(x, y, dy, dx, color=color)

def plot_real_poses(rd, color="pink"):
    plt.scatter(rd.poses[:, 0], rd.poses[:, 1], c=color, marker='o', linestyle='None', s =1)
    for i in range(0, rd.poses.shape[0], 5):
        q_x, q_y, q_Y_deg = rd.poses[i, :]
        scatter_real_orientation(q_x, q_y, (q_Y_deg*np.pi/180) % np.pi, "mediumturquoise")

def gtquery_process(database, x, y, yaw_deg):
    dist_matrix = torch.cdist(torch.Tensor([x,y]).unsqueeze(0), database.poses[:database.synth, :2].unsqueeze(0)).squeeze()

    _, cand_indx = torch.topk(dist_matrix, 5, dim=-1, largest=False, sorted=True)
    print("dist matrix", dist_matrix[cand_indx])

    candidates = database.poses[:database.synth, 2][cand_indx]
    candidates = torch.Tensor([parse_pose([0,0,cand])[3] for cand in candidates])
    print("cand", candidates)
    print("yaw_deg", yaw_deg)

    diff_yaw = torch.min(abs(candidates-yaw_deg), abs(360-abs(candidates-yaw_deg)))
    print("diff yaw", diff_yaw)

    min_yaw_idx = torch.argmin(diff_yaw, dim=-1)

    closest_index = cand_indx[min_yaw_idx]
    closest_index = closest_index.item()
    
    return closest_index

def gtquery_process_check(database, x, y, yaw_deg):
    dist_matrix = torch.cdist(torch.Tensor([x,y]).unsqueeze(0), database.poses[:database.synth, :2].unsqueeze(0)).squeeze()

    _, cand_indx = torch.topk(dist_matrix, 5, dim=-1, largest=False, sorted=True)

    candidates = database.poses[:database.synth, 2][cand_indx]
    candidates = torch.Tensor([parse_pose([0,0,cand])[3] for cand in candidates])

    diff_yaw = torch.min(abs(candidates-yaw_deg), abs(360-abs(candidates-yaw_deg)))
    min_diff_yaw = diff_yaw.min()

    min_yaw_idx = torch.argmin(diff_yaw, dim=-1)

    closest_index = cand_indx[min_yaw_idx]
    closest_index = closest_index.item()
    
    return closest_index, min_diff_yaw

def plot_train_data(data):
    plt.scatter(data.poses[:, 0], data.poses[:, 1], c="pink", marker='o', linestyle='None', s =1)
    for i in range(0, data.poses.shape[0], 20):
        q_x, q_y, q_Y_deg = data.poses[i, :]
        q_Y = (q_Y_deg+90)*np.pi/180
        q_Y %= 2*np.pi
        scatter_real_orientation(q_x, q_y, q_Y, "mediumturquoise", rad=10)

def plot_data(data):
    plt.scatter(data.poses[:, 0], data.poses[:, 1], c="pink", marker='o', linestyle='None', s =1)
    for i in range(0, data.poses.shape[0], 5):
        q_x, q_y, q_Y_deg = data.poses[i, :]
        q_Y = (q_Y_deg+90)*np.pi/180
        q_Y %= 2*np.pi
        scatter_real_orientation(q_x, q_y, q_Y, "mediumturquoise", rad=10)

def filter_data(train_data, data_to_filter):
    train_poses = []
    for pose_file in train_data.pose_paths:
        pose = np.loadtxt(pose_file)[:3]
        train_poses.append(pose)
    train_poses = np.array(train_poses)

    val_poses = []
    for pose_file in data_to_filter.pose_paths:
        pose = np.loadtxt(pose_file)[:3]
        val_poses.append(pose)
    val_poses = np.array(val_poses)

    keep_poses = []
    for i in tqdm(range(len(val_poses)), desc="Filtering validation poses"):
        dists = np.linalg.norm(train_poses - val_poses[i], axis=1)
        if np.min(dists) <= 0.5:
            keep_poses.append(i)

    keep_poses = np.array(keep_poses)

    data_to_filter.imgs = data_to_filter.imgs[keep_poses]
    data_to_filter.pose_paths = data_to_filter.pose_paths[keep_poses]
    data_to_filter.poses = data_to_filter.poses[keep_poses]
    data_to_filter.synth = len(data_to_filter.imgs)

    new_val_poses = []
    for pose_file in data_to_filter.pose_paths:
        pose = np.loadtxt(pose_file)[:3]
        new_val_poses.append(pose)
    new_val_poses = np.array(new_val_poses)

    closest_indices = []
    for val_pose in new_val_poses:
        dists = np.linalg.norm(train_poses[:, :2] - val_pose[:2], axis=1)
        closest_idx = np.argmin(dists)
        closest_indices.append(closest_idx)

    data_to_filter.closest_indices = np.array(closest_indices)

def localization(train_data, val_data, real_data):
    start_plot(train_data)
    train_data.apply_random_rot = False

    plt.scatter(real_data.poses[:, 0], real_data.poses[:, 1], c="pink", marker='o', linestyle='None', s =1)
    for i in range(0, 2000, 5):
        q_pose = real_data.poses[i]
        q_x, q_y, q_Y_deg = q_pose
        q_Y_deg = (q_Y_deg+90)%360
        q_Y = q_Y_deg * np.pi/180
        q_pose = np.array([q_x, q_y, q_Y_deg])
        scatter_orientation(q_x, q_y, q_Y, "mediumturquoise", rad=10)

    plot_synth_poses_train(train_data, "blue")
    plot_synth_poses_val(val_data, "red")

    train_data.apply_random_rot = False

    realidx = random.randint(0, real_data.poses.shape[0])
    q_x, q_y, q_Y_deg = q_pose = real_data.poses[realidx]

    q_Y_deg = (q_Y_deg+90)%360
    q_Y = q_Y_deg * np.pi/180
    q_pose = np.array([q_x, q_y, q_Y_deg])
    scatter_orientation(q_x, q_y, q_Y, "orange", rad=50)

    q_pose2 = np.array([q_x, q_y, (q_Y_deg)%360])
    gt_pose_idx = gtquery_process(train_data, q_x, q_y, q_pose2[2])
    train_closest = train_data[gt_pose_idx][2]

    x2,y2, Y2, Y2_deg = parse_pose(train_closest)
    scatter_orientation(x2, y2, Y2, "green")
    scatter_point(x2, y2, "green")
 
    mask3, iou = generate_interference_mask(x2, y2, Y2, Y2_deg, q_x, q_y, q_Y, q_Y_deg)
    print("iou:", iou)
    print("yaw difference", abs(Y2_deg-q_Y_deg), "deg")
    print("localization error: ", np.linalg.norm(train_closest[:2]-q_pose2[:2], ord=2)/10, "meters")

    plt.imshow(mask3, cmap="gray")
    plt.figure()
        
    f, axarr = plt.subplots(1, 2, figsize=(15, 15))
    axarr[0].set_title("real, query image")
    axarr[1].set_title("closest image from synthetic database")

    axarr[0].imshow(real_data[realidx][1].numpy()[0, :, :], cmap='gray')
    axarr[1].imshow(train_data[gt_pose_idx][1].numpy()[0, :, :], cmap='gray')

def check_gt(train_data, dataset):
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
    index = 0
    indices_to_remove = []
    for batch in dataloader:
        _, _, gtpose, _, _, _ = batch
        gt_pose = gtpose[0]
        visualize_results.check_process(gt_pose, index, indices_to_remove, train_data, dataset, plot=False)
        index += 1
    dataset = remove_data_at_indices(dataset, indices_to_remove)
    return dataset

def remove_data_at_indices(dataset, indices):

    dataset.imgs = np.delete(dataset.imgs, indices, axis=0)
    
    if isinstance(dataset.poses, torch.Tensor):
        poses_np = dataset.poses.numpy()
        poses_np = np.delete(poses_np, indices, axis=0)
        dataset.poses = torch.tensor(poses_np)
    else:
        dataset.poses = np.delete(dataset.poses, indices, axis=0)

    dataset.pose_paths = np.delete(dataset.pose_paths, indices, axis=0)
    dataset.synth = len(dataset.imgs)

    return dataset