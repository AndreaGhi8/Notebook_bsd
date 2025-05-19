# Ghiotto Andrea   2118418

import math
import numpy as np
import torch
from tqdm import tqdm

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

        emb_size = 256
        
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