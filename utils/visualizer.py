# Ghiotto Andrea   2118418

import matplotlib.pyplot as plt
import math, cv2
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random

from utils import plotter

def parse_pose(pose):
    x, y, Y_deg = np.array(pose, copy=True)
    Y_deg_ = Y_deg + 90
    Y_deg_ %= 360
    Y = Y_deg_ * math.pi / 180
    return x, y, Y, Y_deg_

def parse_real_pose(pose):
    x, y, Y = np.array(pose, copy=True)
    Y %= np.pi
    Y_deg = Y * 180 / math.pi
    Y_deg = Y_deg
    Y_deg -= 180
    Y = Y_deg * np.pi / 180
    Y %= np.pi
    return x, y, Y, Y_deg

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

def gtquery_process(database, x, y, yaw_deg):
    dist_matrix = torch.cdist(torch.Tensor([x,y]).unsqueeze(0), database.poses[:database.synth, :2].unsqueeze(0)).squeeze()

    _, cand_indx = torch.topk(dist_matrix, 5, dim=-1, largest=False, sorted=True)

    candidates = database.poses[:database.synth, 2][cand_indx]
    candidates = torch.Tensor([parse_pose([0,0,cand])[3] for cand in candidates])

    diff_yaw = torch.min(abs(candidates-yaw_deg), abs(360-abs(candidates-yaw_deg)))

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

def filter_train_data(data_to_filter, target_pose=np.array([354.449, -440.277, -4.26713])):
    poses = []
    for pose_file in data_to_filter.pose_paths:
        pose = np.loadtxt(pose_file)[:3]
        poses.append(pose)
    poses = np.array(poses)

    keep_poses = []
    for i in tqdm(range(len(poses)), desc="Filtering poses"):
        dist = np.linalg.norm(poses[i] - target_pose)
        if dist >= 5.0:
            keep_poses.append(i)

    keep_poses = np.array(keep_poses)

    data_to_filter.imgs = data_to_filter.imgs[keep_poses]
    data_to_filter.pose_paths = data_to_filter.pose_paths[keep_poses]
    data_to_filter.poses = data_to_filter.poses[keep_poses]
    data_to_filter.synth = len(data_to_filter.imgs)

    new_poses = []
    for pose_file in data_to_filter.pose_paths:
        pose = np.loadtxt(pose_file)[:3]
        new_poses.append(pose)
    new_poses = np.array(new_poses)

    closest_indices = []
    for pose in new_poses:
        dists = np.linalg.norm(poses[:, :2] - pose[:2], axis=1)
        closest_idx = np.argmin(dists)
        closest_indices.append(closest_idx)

    data_to_filter.closest_indices = np.array(closest_indices)

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
    for i in tqdm(range(len(val_poses)), desc="Filtering poses"):
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
    plotter.start_plot(train_data)
    train_data.apply_random_rot = False

    plt.scatter(real_data.poses[:, 0], real_data.poses[:, 1], c="pink", marker='o', linestyle='None', s =1)
    for i in range(0, 2000, 5):
        q_pose = real_data.poses[i]
        q_x, q_y, q_Y_deg = q_pose
        q_Y_deg = (q_Y_deg+90)%360
        q_Y = q_Y_deg * np.pi/180
        q_pose = np.array([q_x, q_y, q_Y_deg])
        plotter.scatter_orientation(q_x, q_y, q_Y, "mediumturquoise", rad=10)

    plotter.plot_synth_poses_train(train_data, "blue")
    plotter.plot_synth_poses_val(val_data, "red")

    train_data.apply_random_rot = False

    realidx = random.randint(0, real_data.poses.shape[0])
    q_x, q_y, q_Y_deg = q_pose = real_data.poses[realidx]

    q_Y_deg = (q_Y_deg+90)%360
    q_Y = q_Y_deg * np.pi/180
    q_pose = np.array([q_x, q_y, q_Y_deg])
    plotter.scatter_orientation(q_x, q_y, q_Y, "orange", rad=50)

    q_pose2 = np.array([q_x, q_y, (q_Y_deg)%360])
    gt_pose_idx = gtquery_process(train_data, q_x, q_y, q_pose2[2])
    train_closest = train_data[gt_pose_idx][2]

    x2,y2, Y2, Y2_deg = parse_pose(train_closest)
    plotter.scatter_orientation(x2, y2, Y2, "green")
    plotter.scatter_point(x2, y2, "green")
 
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
        check_process(gt_pose, index, indices_to_remove, train_data, dataset, plot=False)
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
    dataset.closest_indices = np.delete(dataset.closest_indices, indices)
    dataset.synth = len(dataset.imgs)

    return dataset

def process(q_idx, net, train_data, val_data, plot=True):

    if plot:
        plotter.start_plot(train_data)

    q_image_a, q_image, q_pose, _, _, _ = val_data[q_idx]
    q_image_a = q_image_a[None].cuda()
    if plot:
        q_desc, (q_image_r, _, _, _, _)  = net(q_image_a, reco=True)
    else:
        q_desc = net(q_image_a, reco=False)[0, :]
    q_desc = q_desc.detach().cpu().numpy()
    
    q_x, q_y, q_Y, q_Y_deg = parse_pose(q_pose)
    if plot:
        plotter.scatter_point(q_x, q_y, 'magenta', label="val pose (query)")
        plotter.scatter_orientation(q_x, q_y, q_Y, "magenta")

    if plot:
        plt.scatter(train_data.poses[:train_data.synth, 0], train_data.poses[:train_data.synth, 1], c="blue", marker='o', linestyle='None', s =1, label="training set positions")
        plt.scatter(val_data.poses[:, 0], val_data.poses[:, 1], c="red", marker='o', linestyle='None', s =1, label="validation set positions")

    train_data.apply_random_rot = False
    minidx = train_data.query(q_desc)
    min_img = train_data[minidx][1]
    min_pose = train_data[minidx][2]
    min_x, min_y, min_Y, min_Y_deg = parse_pose(min_pose)
    if plot:
        plotter.scatter_point(min_x, min_y, 'gold', label="predicted pose")
        plotter.scatter_orientation(min_x, min_y, min_Y, "gold")
    
    gt_pose_idx = gtquery_process(train_data, q_x, q_y, q_Y_deg)
    gt_pose = train_data[gt_pose_idx][2]

    gt_x, gt_y, gt_Y, gt_Y_deg = parse_pose(gt_pose)
    if plot:
                   
        plotter.scatter_orientation(gt_x, gt_y, gt_Y, "green")
        plotter.scatter_point(gt_x, gt_y, "green", label="database gt closest pose")
    
    mask3, iou = generate_interference_mask(min_x, min_y, min_Y, min_Y_deg, q_x, q_y, q_Y, q_Y_deg)

    loca_error=np.linalg.norm(gt_pose[:2]-min_pose[:2], ord=2)/10

    orie_error = gt_Y_deg - min_Y_deg.item()
    orie_error = np.abs((orie_error + 180) % 360 - 180)

    gt_closest_image = train_data[val_data.closest_indices[q_idx]][1]

    if plot:
        plt.imshow(mask3, cmap="gray")
        print("iou:", iou)
        plt.legend(loc="lower right")
    
        plt.figure()
        
        f, axarr = plt.subplots(1, 3, figsize=(15, 15))
        axarr[0].set_title("query image")
        axarr[1].set_title("predicted image")
        axarr[2].set_title("reconstructed query image")
        
        axarr[0].imshow(q_image.detach().cpu().numpy()[0, :, :], cmap='gray')
        axarr[1].imshow(min_img[0, :, :], cmap='gray')
        axarr[2].imshow(q_image_r.detach().cpu().numpy()[0, 0, :, :], cmap='gray')
        
    return loca_error, orie_error

def check_process(gt_pose, index, indices_to_remove, train_data, dataset, plot=True):

    if plot:
        plotter.start_plot(train_data)

    gt_x, gt_y, gt_Y, gt_Y_deg = parse_pose(gt_pose)
    if plot:
                   
        plotter.scatter_orientation(gt_x, gt_y, gt_Y, "green")
        plotter.scatter_point(gt_x, gt_y, "green", label="database gt closest pose")
    
    if plot:
        plt.scatter(train_data.poses[:train_data.synth, 0], train_data.poses[:train_data.synth, 1], c="blue", marker='o', linestyle='None', s =1, label="training set positions")
        plt.scatter(dataset.poses[:, 0], dataset.poses[:, 1], c="red", marker='o', linestyle='None', s =1, label="validation set positions")

    q_pose_idx, min_diff_yaw = gtquery_process_check(train_data, gt_x, gt_y, gt_Y_deg)

    if min_diff_yaw > 7.5:
        indices_to_remove.append(index)
    else:
        q_pose = train_data[q_pose_idx][2]

        q_x, q_y, q_Y, q_Y_deg = parse_pose(q_pose)
        if plot:
            plotter.scatter_point(q_x, q_y, 'magenta', label="val pose (query)")
            plotter.scatter_orientation(q_x, q_y, q_Y, "magenta")
        
        mask3, iou = generate_interference_mask(gt_x, gt_y, gt_Y, gt_Y_deg, q_x, q_y, q_Y, q_Y_deg)

        if plot:
            plt.imshow(mask3, cmap="gray")
            plt.legend(loc="lower right")
            plt.figure()

def analyze_feature_robustness(train_data, net):
    q_idx = 200
    q_image_a, q_image, q_pose, _, _, _ = train_data[q_idx]
    q_image_a = q_image_a[None].cuda()

    out = net.encoder(q_image_a)
    print(out[0].shape, out[0].min(), out[0].max())
    print(out[1].shape, out[1].min(), out[1].max())
    print(out[2].shape, out[2].min(), out[2].max())
    print(out[3].shape, out[3].min(), out[3].max())
    out[2][0, :, :, :] = out[2][0, :, :, :] + torch.normal(0, 3, size=out[2][:, :, :, :].shape).cuda()
    out[2][0, :, :, :] = out[2][0, :, :, :] + torch.normal(0, 3, size=out[2][:, :, :, :].shape).cuda()
    out[3][0, :, :, :] = 0
    q_desc = torch.nn.functional.normalize(net.embed(out[-1]).flatten(1), p=2, dim=1)
    q_image_r = net.decoder(out)[0]

    print(q_image_r[0].min(), q_image_r[0].max())

    if len(q_image_r.shape) == 3 and q_image_r.shape[0] == 1 and q_image_r.shape[1] == 256 and q_image_r.shape[2] == 256:
        q_image_r = q_image_r.unsqueeze(1)

    f, axarr = plt.subplots(1, 2, figsize=(15, 15))
    axarr[0].set_title("query image")
    axarr[1].set_title("reco image")

    axarr[0].imshow(q_image.detach().cpu().numpy()[0, :, :], cmap='gray')
    axarr[1].imshow(q_image_r.detach().cpu().numpy()[0, 0, :, :], cmap='gray')

def process_real(q_idx, net, train_data, real_data):
       
    plotter.start_plot(train_data)
    plotter.plot_synth_poses_train(train_data, "blue")

    q_image_a, q_image, q_pose, _, _ = real_data[q_idx]
        
    q_image_a = q_image_a[None].cuda()
    q_desc, (q_image_r, _, _, _, _)  = net(q_image_a, reco=True)
    q_desc = q_desc.detach().cpu().numpy()
    
    q_x, q_y, q_Y, q_Y_deg = parse_real_pose(q_pose)
    plotter.scatter_point(q_x, q_y, 'pink', label="val pose (query)")
    plotter.scatter_real_orientation(q_x, q_y, q_Y, "pink")

    print(q_x, q_y, q_Y, q_Y_deg)

    train_data.apply_random_rot = False
    minidx = train_data.query(q_desc)
    _, min_img, min_pose, min_img_path, min_lab_path, _ = train_data[minidx]
    
    rad = 50

    min_x, min_y, min_Y, min_Y_deg = parse_pose(min_pose)
    plotter.scatter_point(min_x, min_y, 'blue', label="predicted pose")
    plotter.scatter_real_orientation(min_x, min_y, min_Y, "blue")
    
    gt_pose_idx = train_data.gtquery_real(q_pose)
    gt_pose     = train_data[gt_pose_idx][2]
    gt_img_path = train_data.imgs[gt_pose_idx//len(train_data)]
    gt_image = cv2.cvtColor(cv2.imread(gt_img_path), cv2.COLOR_BGR2GRAY)
    gt_x, gt_y, gt_Y, gt_Y_deg = parse_pose(gt_pose)
    plotter.scatter_real_orientation(gt_x, gt_y, gt_Y, "green")

    mask3, iou = generate_interference_mask(min_x, min_y, min_Y, min_Y_deg, gt_x, gt_y, gt_Y, gt_Y_deg)
    plt.imshow(mask3, cmap="gray")
    print("iou:", iou)
    plt.legend(loc="lower right")
    plotter.plot_real_poses(real_data, "pink")
    
    plt.figure()
    
    f, axarr = plt.subplots(1, 3, figsize=(15, 15))
    axarr[0].set_title("query image")
    axarr[1].set_title("closest image")
    axarr[2].set_title("ground truth 360 closest synthetic image")
    
    axarr[0].imshow(q_image.detach().cpu().numpy()[0, :, :], cmap='gray')
    axarr[1].imshow(min_img.numpy()[0, :, :], cmap='gray')
    axarr[2].imshow(gt_image, cmap='gray')
    
    print("localization error Upper: ", np.linalg.norm(q_pose[:2]-min_pose[:2], ord=2)/10, "meters")

def process_only_real(q_idx, net, train_data, val_data, plot=True):

    if plot:
        plotter.start_plot(train_data)

        plt.scatter(train_data.poses[:train_data.synth, 0], train_data.poses[:train_data.synth, 1], c="blue", marker='o', linestyle='None', s =1, label="training set positions")
        plt.scatter(val_data.poses[:, 0], val_data.poses[:, 1], c="red", marker='o', linestyle='None', s =1, label="validation set positions")

    q_image_a, q_image, q_pose, _, _ = val_data[q_idx]
    q_image_a = q_image_a[None].cuda()
    if plot:
        q_desc, (q_image_r, _, _, _, _)  = net(q_image_a, reco=True)
    else:
        q_desc = net(q_image_a, reco=False)[0, :]
    q_desc = q_desc.detach().cpu().numpy()
    
    q_x, q_y, q_Y_deg = q_pose
    q_Y_deg = (90+q_Y_deg)%360
    q_Y = q_Y_deg * np.pi/180

    if plot:
        plotter.scatter_point(q_x, q_y, 'magenta', label="val pose (query)")
        plotter.scatter_real_orientation(q_x, q_y, q_Y, "magenta")
    
    minidx = train_data.query(q_desc)
    _, min_img, min_pose, min_img_path, min_lab_path = train_data[minidx]

    min_x, min_y, min_Y_deg = min_pose
    min_Y_deg = (90+min_Y_deg)%360
    min_Y = min_Y_deg * np.pi/180

    if plot:
        plotter.scatter_point(min_x, min_y, 'gold', label="predicted pose")
        plotter.scatter_real_orientation(min_x, min_y, min_Y, "gold")
    
    gt_pose = train_data[val_data.closest_indices[q_idx]][2]

    gt_x, gt_y, gt_Y_deg = gt_pose
    gt_Y_deg = (90+gt_Y_deg)%360
    gt_Y = gt_Y_deg * np.pi/180
    if plot:
        plotter.scatter_point(gt_x, gt_y, "green", label="database gt closest pose")
    
    mask3, iou = generate_interference_mask(min_x, min_y, min_Y, min_Y_deg, q_x, q_y, q_Y, q_Y_deg)

    loca_error=np.linalg.norm(gt_pose[:2]-min_pose[:2], ord=2)/10

    orie_error = gt_Y_deg - min_Y_deg.item()
    orie_error = np.abs((orie_error + 180) % 360 - 180)
    
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

def visualize_real(train_data, real_data):
    plotter.start_plot(train_data)
    train_data.apply_random_rot = False

    for i in range(0, 300, 5):
        q_pose = real_data.poses[i]
        q_x, q_y, q_Y_deg = q_pose
        q_Y_deg = (90+q_Y_deg)%360
        q_Y = q_Y_deg * np.pi/180
        q_pose = np.array([q_x, q_y, q_Y_deg])
        plotter.scatter_real_orientation(q_x, q_y, q_Y, "mediumturquoise", rad=10)
        plt.scatter(real_data.poses[i, 0], real_data.poses[i, 1], c="red", marker='o', linestyle='None', s =1)

    for i in range(300, 710, 2):
        q_pose = real_data.poses[i]
        q_x, q_y, q_Y_deg = q_pose
        q_Y_deg = (90+q_Y_deg)%360
        q_Y = q_Y_deg * np.pi/180
        q_pose = np.array([q_x, q_y, q_Y_deg])
        plotter.scatter_real_orientation(q_x, q_y, q_Y, "mediumturquoise", rad=10)
        plt.scatter(real_data.poses[i, 0], real_data.poses[i, 1], c="red", marker='o', linestyle='None', s =1) 

    for i in range(710, 900, 2):
        q_pose = real_data.poses[i]
        q_x, q_y, q_Y_deg = q_pose
        q_Y_deg = (90+q_Y_deg)%360
        q_Y = q_Y_deg * np.pi/180
        q_pose = np.array([q_x, q_y, q_Y_deg])
        plotter.scatter_real_orientation(q_x, q_y, q_Y, "mediumturquoise", rad=10)
        plt.scatter(real_data.poses[i, 0], real_data.poses[i, 1], c="blue", marker='o', linestyle='None', s =1)

    for i in range(900, 1200, 5):
        q_pose = real_data.poses[i]
        q_x, q_y, q_Y_deg = q_pose
        q_Y_deg = (90+q_Y_deg)%360
        q_Y = q_Y_deg * np.pi/180
        q_pose = np.array([q_x, q_y, q_Y_deg])
        plotter.scatter_real_orientation(q_x, q_y, q_Y, "mediumturquoise", rad=10)
        plt.scatter(real_data.poses[i, 0], real_data.poses[i, 1], c="blue", marker='o', linestyle='None', s =1)  

    for i in range(1200, 1500, 5):
        q_pose = real_data.poses[i]
        q_x, q_y, q_Y_deg = q_pose
        q_Y_deg = (90+q_Y_deg)%360
        q_Y = q_Y_deg * np.pi/180
        q_pose = np.array([q_x, q_y, q_Y_deg])
        plotter.scatter_real_orientation(q_x, q_y, q_Y, "mediumturquoise", rad=10)
        plt.scatter(real_data.poses[i, 0], real_data.poses[i, 1], c="blue", marker='o', linestyle='None', s =1)