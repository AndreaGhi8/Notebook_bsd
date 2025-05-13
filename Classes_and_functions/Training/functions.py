# Ghiotto Andrea   2118418

from Classes_and_functions import imports
from Classes_and_functions.Dataloader import functions

def calcEmbedMatrix(vectors):
    diff = vectors.unsqueeze(1) - vectors.unsqueeze(0)

    squared_dist = imports.torch.sum(diff ** 2, dim=2)
    squared_dist = imports.torch.clamp(squared_dist, 1e-8, squared_dist.max().item())
    return imports.torch.sqrt(squared_dist)

def generate_sonar_map(pose, flag):
    rad = 150
    if not flag:
        return imports.np.zeros((325+rad*2, 295+rad*2)).T
    x,y,Y_deg = pose[:3].clone()
    x+=rad
    y+=rad
    Y_deg = 90 - Y_deg
    Y_deg = 90 - Y_deg
    center = imports.np.array([x, y]).astype(int)
    mask = functions.sector_mask((325+rad*2, 295+rad*2),center,rad,Y_deg).T
    return mask

def sonar_overlap_distance_matrix(gtposes, mode):
    iou_matrix = imports.np.zeros((gtposes.shape[0], gtposes.shape[0]))
    sonar_images = [generate_sonar_map(gtposes[i], mode[i]) for i in range(gtposes.shape[0])]

    for i in range(gtposes.shape[0]):
        for j in range(i+1, gtposes.shape[0]):
            mask_and = imports.np.logical_and(sonar_images[i], sonar_images[j])
            mask_xor = imports.np.logical_xor(sonar_images[i], sonar_images[j])
            Y1 = (90-gtposes[i][2])*imports.np.pi/180
            Y2 = (90-gtposes[j][2])*imports.np.pi/180
            
            R3 = 2*abs(imports.math.cos((Y1-Y2)/2))
            mask_and = mask_and*R3
            
            union = sonar_images[i].sum() + sonar_images[j].sum()
            intersection = mask_and.sum()
            iou = intersection/union if union >0 else 0
            iou_matrix[i][j] = iou

    iou_matrix = imports.np.maximum( iou_matrix, iou_matrix.transpose() )
    imports.np.fill_diagonal(iou_matrix, 1.0)

    return imports.torch.Tensor(1-iou_matrix)

def correlate_poses_topk(matrix1, matrix2, k=1):
    dist_matrix = imports.torch.cdist(matrix1, matrix2)
    closest_indices = imports.torch.argsort(dist_matrix, dim=1)[:, :k]

    return closest_indices

def computeAverageMetricError(pred_indices, gt_indices, train_data, k=1):
    prp = train_data.poses[pred_indices, :2]
    grp = train_data.poses[gt_indices, :2]
    if k==1:
        diff = prp.unsqueeze(1) - grp.unsqueeze(1)
    else:
        diff = prp - grp.unsqueeze(1).expand((-1, k, 2))
    
    squared_dist = imports.torch.sqrt(imports.torch.sum(diff ** 2, dim=2))

    if k>1:
        squared_dist = imports.torch.min(squared_dist, dim=1)[0]
    
    return squared_dist.mean()