# Ghiotto Andrea   2118418

import math
import numpy as np
import torch
from tqdm import tqdm
import time
import csv
import os

from utils import visualizer

def calcEmbedMatrix(vectors):
    diff = vectors.unsqueeze(1) - vectors.unsqueeze(0)

    squared_dist = torch.sum(diff ** 2, dim=2)
    squared_dist = torch.clamp(squared_dist, 1e-8, squared_dist.max().item())
    return torch.sqrt(squared_dist)

def generate_sonar_map(pose, flag):
    rad = 150
    if not flag:
        return np.zeros((325+rad*2, 295+rad*2)).T
    x,y,Y_deg = pose[:3].clone()
    x+=rad
    y+=rad
    Y_deg = 90 - Y_deg
    Y_deg = 90 - Y_deg
    center = np.array([x, y]).astype(int)
    mask = visualizer.sector_mask((325+rad*2, 295+rad*2),center,rad,Y_deg).T
    return mask

def sonar_overlap_distance_matrix(gtposes, mode):
    iou_matrix = np.zeros((gtposes.shape[0], gtposes.shape[0]))
    sonar_images = [generate_sonar_map(gtposes[i], mode[i]) for i in range(gtposes.shape[0])]

    for i in range(gtposes.shape[0]):
        for j in range(i+1, gtposes.shape[0]):
            mask_and = np.logical_and(sonar_images[i], sonar_images[j])
            mask_xor = np.logical_xor(sonar_images[i], sonar_images[j])
            Y1 = (90-gtposes[i][2])*np.pi/180
            Y2 = (90-gtposes[j][2])*np.pi/180
            
            R3 = 2*abs(math.cos((Y1-Y2)/2))
            mask_and = mask_and*R3
            
            union = sonar_images[i].sum() + sonar_images[j].sum()
            intersection = mask_and.sum()
            iou = intersection/union if union >0 else 0
            iou_matrix[i][j] = iou

    iou_matrix = np.maximum( iou_matrix, iou_matrix.transpose() )
    np.fill_diagonal(iou_matrix, 1.0)

    return torch.Tensor(1-iou_matrix)

def correlate_poses_topk(matrix1, matrix2, k=1):
    dist_matrix = torch.cdist(matrix1, matrix2)
    closest_indices = torch.argsort(dist_matrix, dim=1)[:, :k]

    return closest_indices

def computeAverageMetricError(pred_indices, gt_indices, train_data, k=1):
    prp = train_data.poses[pred_indices, :2]
    grp = train_data.poses[gt_indices, :2]
    if k==1:
        diff = prp.unsqueeze(1) - grp.unsqueeze(1)
    else:
        diff = prp - grp.unsqueeze(1).expand((-1, k, 2))
    
    squared_dist = torch.sqrt(torch.sum(diff ** 2, dim=2))

    if k>1:
        squared_dist = torch.min(squared_dist, dim=1)[0]
    
    return squared_dist.mean()

def get_descriptors(train_data, val_data, net):
    with torch.no_grad():

        #emb_size = 256
        sample_descriptor = net(train_data[0][0][None].cuda(), reco=False)[0, :].detach().cpu()
        emb_size = sample_descriptor.shape[0]
        train_pred_embeds = torch.zeros((train_data.synth, emb_size))
        
        for idx in tqdm(range(train_data.synth)):
            image, _, _, _, _, _ = train_data[idx]
            image = image[None].cuda()
            descriptor = net(image, reco=False)[0, :].detach().cpu()
            train_pred_embeds[idx, :] = descriptor
        train_data.computeDescriptors(net)
        train_pred_embeds = torch.Tensor(train_data.descriptors)
        
        val_data.computeDescriptors(net)
        val_pred_embeds = torch.Tensor(val_data.descriptors)
        
        gt_indices = val_data.closest_indices
        
        pred_indices      = correlate_poses_topk(val_pred_embeds, train_pred_embeds, k=1).squeeze()
        pred_indices_top5 = correlate_poses_topk(val_pred_embeds, train_pred_embeds, k=5)
        
        print(pred_indices.shape, pred_indices_top5.shape)
        
        avg_metric_e      = computeAverageMetricError(pred_indices,      gt_indices, train_data, k=1)
        avg_metric_e_top5 = computeAverageMetricError(pred_indices_top5, gt_indices, train_data, k=5)
        
        print("avg_metric_e     :", avg_metric_e)
        print("avg_metric_e_top5:", avg_metric_e_top5)

def count_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())

    return total_params

def inference_time(net, train_dataloader):
    net.eval()
    times = []
    for i, sample in tqdm(enumerate(train_dataloader)):
        with torch.no_grad():
            sample = sample[0].cuda()
            torch.cuda.synchronize()
            start_inf = time.time()
            _ = net(sample)
            torch.cuda.synchronize()
            end_inf = time.time()

        inference_time_per_image = (end_inf - start_inf)
        times.append(inference_time_per_image)
        if i>110:
            break
    inference_time_per_image = np.array(times[10:]).mean()
    
    return inference_time_per_image

def inference_memory(net, input_tensor):
    net.eval()
    with torch.no_grad():
        torch.cuda.reset_peak_memory_stats()
        _ = net(input_tensor)
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2

    return peak_mem

def save_results(model_name, total_params, training_time, inference_time_per_img, inference_memory_per_batch, ale_t, aoe_t, ale_r, aoe_r, file_path):
    
    header = [
        "Model",
        "Total Parameters",
        "Training Time (s)",
        "Inference Time per Image (s)",
        "Inference Memory per Batch (MB)",
        "Average Localization Error in Test (m)",
        "Average Orientation Error in Test (°)",
        "Average Localization Error in Real(m)",
        "Average Orientation Error in Real (°)"
    ]

    rows = []
    if os.path.isfile(file_path):
        with open(file_path, mode='r', newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = list(reader)

    data_rows = []
    if rows and rows[0] == header:
        data_rows = rows[1:]
    elif rows:
        data_rows = rows
    else:
        rows = [header]

    new_row = [
        model_name,
        total_params,
        f"{training_time:.4f}",
        f"{inference_time_per_img:.4f}",
        f"{inference_memory_per_batch:.4f}",
        f"{ale_t:.4f}",
        f"{aoe_t:.4f}",
        f"{ale_r:.4f}",
        f"{aoe_r:.4f}"
    ]

    updated = False
    for i, row in enumerate(data_rows):
        if row and row[0] == model_name:
            data_rows[i] = new_row
            updated = True
            break

    if not updated:
        data_rows.append(new_row)

    with open(file_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        writer.writerows(data_rows)