# Ghiotto Andrea   2118418

from Classes_and_functions import imports

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
    x, y, Y_deg = imports.np.array(pose, copy=True)
    # Y_deg = 90 + Y_deg
    Y_deg %= 360
    Y = Y_deg * imports.math.pi / 180
    return x, y, Y, Y_deg

def scatter_point(x, y, color, label=None):
    global plt
    if label is None:
        plt.scatter(x, y, c=color, s = 20.51)
    else:
        plt.scatter(x, y, c=color, s = 20.51, label=label)
        
def scatter_orientation(x, y, Y, color, rad=50):
    global plt
    dy, dx = rad*imports.math.cos(Y), rad*imports.math.sin(Y)
    plt.arrow(x, y, dx, dy, color=color)

def sector_mask(shape,centre,radius, Y_deg):
    angle_range = (Y_deg-60, Y_deg+60)
    x,y = imports.np.ogrid[:shape[0],:shape[1]]
    cx,cy = centre
    tmin,tmax = imports.np.deg2rad(angle_range)
    if tmax < tmin:
            tmax += 2*imports.np.pi

    r2 = (x-cx)*(x-cx) + (y-cy)*(y-cy)
    theta = imports.np.arctan2(x-cx,y-cy) - tmin
    theta %= (2*imports.np.pi)
    circmask = r2 <= radius*radius
    anglemask = theta <= (tmax-tmin)

    return (circmask*anglemask).astype(int)

def generate_interference_mask(x1, y1, Y1, Y1_deg, x2, y2, Y2, Y2_deg, sonar_map_size = (325+50, 295+50)):
    
    mask1 = sector_mask(sonar_map_size,imports.np.array([x1, y1]).astype(int),50,Y1_deg).T
    mask2 = sector_mask(sonar_map_size,imports.np.array([x2, y2]).astype(int),50,Y2_deg).T

    mask_and = imports.np.logical_and(mask1, mask2)
    mask_xor = imports.np.logical_xor(mask1, mask2)
    
    R3 = 2*abs(imports.math.cos((Y1-Y2)/2))
    mask_and = mask_and*R3
    
    mask3 = mask_and + mask_xor
    
    union = mask1.sum() + mask2.sum()
    intersection = mask_and.sum()
    iou = intersection/union if union>0 else 0

    return mask3, iou

def generate_interference_mask_transparent(x1, y1, Y1, Y1_deg, x2, y2, Y2, Y2_deg, sonar_map_size = (325+50, 295+50)):
    
    mask1 = sector_mask(sonar_map_size,imports.np.array([x1, y1]).astype(int),50,Y1_deg).T
    mask2 = sector_mask(sonar_map_size,imports.np.array([x2, y2]).astype(int),50,Y2_deg).T

    mask_and = imports.np.logical_and(mask1, mask2)
    mask_xor = imports.np.logical_xor(mask1, mask2)
    
    R3 = 2*abs(imports.math.cos((Y1-Y2)/2))
    mask_and = mask_and*R3
    
    mask3 = mask_and + mask_xor
    
    union = mask1.sum() + mask2.sum()
    intersection = mask_and.sum()
    iou = intersection/union if union>0 else 0

    return mask3*0, iou
        
def scatter_real_orientation(x, y, Y, color, rad=50):
    global plt
    dx, dy = rad*imports.math.cos(Y), rad*imports.math.sin(Y)
    plt.arrow(x, y, dy, dx, color=color)

def plot_real_poses(rd, color="pink"):
    global plt
    plt.scatter(rd.poses[:, 0], rd.poses[:, 1], c=color, marker='o', linestyle='None', s =1)
    for i in range(0, rd.poses.shape[0], 5):
        q_x, q_y, q_Y_deg = rd.poses[i, :]
        scatter_real_orientation(q_x, q_y, (q_Y_deg*imports.np.pi/180) % imports.np.pi, "mediumturquoise")

def gtquery(database, x, y, yaw_deg):
    dist_matrix = imports.torch.cdist(imports.torch.Tensor([x,y]).unsqueeze(0), database.poses[:database.synth, :2].unsqueeze(0)).squeeze()

    _, cand_indx = imports.torch.topk(dist_matrix, 5, dim=-1, largest=False, sorted=True)

    candidates = database.poses[:database.synth, 2][cand_indx]
    diff_yaw = abs(candidates-yaw_deg)%360

    min_yaw_idx = imports.torch.argmin(diff_yaw, dim=-1)

    closest_index = cand_indx[min_yaw_idx]
    closest_index = closest_index.item()
    
    return closest_index