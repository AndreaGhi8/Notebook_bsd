# Ghiotto Andrea   2118418

from Classes_and_functions import imports
from Classes_and_functions.Dataloader import functions

def parse_real_pose(pose):
    x, y, Y = imports.np.array(pose, copy=True)
    Y %= imports.np.pi
    Y_deg = Y * 180 / imports.math.pi
    Y_deg = Y_deg
    Y_deg -= 180
    Y = Y_deg * imports.np.pi / 180
    Y %= imports.np.pi
    return x, y, Y, Y_deg

def scatter_point(x, y, color, label=None):
    if label is None:
        imports.plt.scatter(x, y, c=color, s = 20.51)
    else:
        imports.plt.scatter(x, y, c=color, s = 20.51, label=label)
        
def scatter_real_orientation(x, y, Y, color, rad=50):
    dx, dy = rad*imports.math.cos(Y), rad*imports.math.sin(Y)
    imports.plt.arrow(x, y, dx, dy, color=color)

def process_real(q_idx, net, train_data, real_data):
       
    functions.start_plot(train_data)
    functions.plot_synth_poses_train(train_data, "blue")

    q_image_a, q_image, q_pose, _, _ = real_data[q_idx]
        
    q_image_a = q_image_a[None].cuda()
    q_desc, (q_image_r, _, _, _, _)  = net(q_image_a, reco=True)
    q_desc = q_desc.detach().cpu().numpy()
    
    q_x, q_y, q_Y, q_Y_deg = parse_real_pose(q_pose)
    scatter_point(q_x, q_y, 'pink', label="val pose (query)")
    scatter_real_orientation(q_x, q_y, q_Y, "pink")

    print(q_x, q_y, q_Y, q_Y_deg)

    train_data.apply_random_rot = False
    minidx = train_data.query(q_desc)
    _, min_img, min_pose, min_img_path, min_lab_path, _ = train_data[minidx]
    
    rad = 50

    min_x, min_y, min_Y, min_Y_deg = functions.parse_pose(min_pose)
    scatter_point(min_x, min_y, 'blue', label="predicted pose")
    scatter_real_orientation(min_x, min_y, min_Y, "blue")
    
    gt_pose_idx = train_data.gtquery_real(q_pose)
    gt_pose     = train_data[gt_pose_idx][2]
    gt_img_path = train_data.imgs[gt_pose_idx//len(train_data)]
    gt_image = imports.cv2.cvtColor(imports.cv2.imread(gt_img_path), imports.cv2.COLOR_BGR2GRAY)
    gt_x, gt_y, gt_Y, gt_Y_deg = functions.parse_pose(gt_pose)
    scatter_real_orientation(gt_x, gt_y, gt_Y, "green")

    mask3, iou = functions.generate_interference_mask(min_x, min_y, min_Y, min_Y_deg, gt_x, gt_y, gt_Y, gt_Y_deg)
    imports.plt.imshow(mask3, cmap="gray")
    print("iou:", iou)
    imports.plt.legend(loc="lower right")
    functions.plot_real_poses(real_data, "pink")
    
    imports.plt.figure()
    
    f, axarr = imports.plt.subplots(1, 3, figsize=(15, 15))
    axarr[0].set_title("query image")
    axarr[1].set_title("closest image")
    axarr[2].set_title("ground truth 360 closest synthetic image")
    
    axarr[0].imshow(q_image.detach().cpu().numpy()[0, :, :], cmap='gray')
    axarr[1].imshow(min_img.numpy()[0, :, :], cmap='gray')
    axarr[2].imshow(gt_image, cmap='gray')
    
    print("localization error Upper: ", imports.np.linalg.norm(q_pose[:2]-min_pose[:2], ord=2)/10, "meters")

def process_only_real(q_idx, net, train_data, val_data, plot=True):

    if plot:
        functions.start_plot(train_data)

        imports.plt.scatter(train_data.poses[:train_data.synth, 0], train_data.poses[:train_data.synth, 1], c="blue", marker='o', linestyle='None', s =1, label="training set positions")
        imports.plt.scatter(val_data.poses[:, 0], val_data.poses[:, 1], c="red", marker='o', linestyle='None', s =1, label="validation set positions")

    q_image_a, q_image, q_pose, _, _ = val_data[q_idx]
    q_image_a = q_image_a[None].cuda()
    if plot:
        q_desc, (q_image_r, _, _, _, _)  = net(q_image_a, reco=True)
    else:
        q_desc = net(q_image_a, reco=False)[0, :]
    q_desc = q_desc.detach().cpu().numpy()
    
    q_x, q_y, q_Y_deg = q_pose
    q_Y_deg = (90+q_Y_deg)%360
    q_Y = q_Y_deg * imports.np.pi/180

    if plot:
        scatter_point(q_x, q_y, 'magenta', label="val pose (query)")
        scatter_real_orientation(q_x, q_y, q_Y, "magenta")
    
    minidx = train_data.query(q_desc)
    _, min_img, min_pose, min_img_path, min_lab_path = train_data[minidx]

    min_x, min_y, min_Y_deg = min_pose
    min_Y_deg = (90+min_Y_deg)%360
    min_Y = min_Y_deg * imports.np.pi/180

    if plot:
        scatter_point(min_x, min_y, 'gold', label="predicted pose")
        scatter_real_orientation(min_x, min_y, min_Y, "gold")
    
    gt_pose = train_data[val_data.closest_indices[q_idx]][2]

    gt_x, gt_y, gt_Y_deg = gt_pose
    gt_Y_deg = (90+gt_Y_deg)%360
    gt_Y = gt_Y_deg * imports.np.pi/180
    if plot:
        scatter_point(gt_x, gt_y, "green", label="database gt closest pose")
    
    mask3, iou = functions.generate_interference_mask(min_x, min_y, min_Y, min_Y_deg, q_x, q_y, q_Y, q_Y_deg)

    loca_error=imports.np.linalg.norm(gt_pose[:2]-min_pose[:2], ord=2)/10

    orie_error = gt_Y_deg - min_Y_deg.item()
    orie_error = imports.np.abs((orie_error + 180) % 360 - 180)
    
    if plot:
        imports.plt.imshow(mask3, cmap="gray")
        print("iou:", iou)
        imports.plt.legend(loc="lower right")
    
        imports.plt.figure()
        
        f, axarr = imports.plt.subplots(1, 3, figsize=(15, 15))
        axarr[0].set_title("query image")
        axarr[1].set_title("closest image from database")
        axarr[2].set_title("reconstructed query image")
        
        axarr[0].imshow(q_image.detach().cpu().numpy()[0, :, :], cmap='gray')
        axarr[1].imshow(min_img.numpy()[0, :, :], cmap='gray')
        axarr[2].imshow(q_image_r.detach().cpu().numpy()[0, 0, :, :], cmap='gray')
        
    return loca_error, orie_error

def visualize_real(train_data, real_data):
    functions.start_plot(train_data)
    train_data.apply_random_rot = False

    for i in range(0, 300, 5):
        q_pose = real_data.poses[i]
        q_x, q_y, q_Y_deg = q_pose
        q_Y_deg = (90+q_Y_deg)%360
        q_Y = q_Y_deg * imports.np.pi/180
        q_pose = imports.np.array([q_x, q_y, q_Y_deg])
        scatter_real_orientation(q_x, q_y, q_Y, "mediumturquoise", rad=10)
        imports.plt.scatter(real_data.poses[i, 0], real_data.poses[i, 1], c="red", marker='o', linestyle='None', s =1)

    for i in range(300, 710, 2):
        q_pose = real_data.poses[i]
        q_x, q_y, q_Y_deg = q_pose
        q_Y_deg = (90+q_Y_deg)%360
        q_Y = q_Y_deg * imports.np.pi/180
        q_pose = imports.np.array([q_x, q_y, q_Y_deg])
        scatter_real_orientation(q_x, q_y, q_Y, "mediumturquoise", rad=10)
        imports.plt.scatter(real_data.poses[i, 0], real_data.poses[i, 1], c="red", marker='o', linestyle='None', s =1) 

    for i in range(710, 900, 2):
        q_pose = real_data.poses[i]
        q_x, q_y, q_Y_deg = q_pose
        q_Y_deg = (90+q_Y_deg)%360
        q_Y = q_Y_deg * imports.np.pi/180
        q_pose = imports.np.array([q_x, q_y, q_Y_deg])
        scatter_real_orientation(q_x, q_y, q_Y, "mediumturquoise", rad=10)
        imports.plt.scatter(real_data.poses[i, 0], real_data.poses[i, 1], c="blue", marker='o', linestyle='None', s =1)

    for i in range(900, 1200, 5):
        q_pose = real_data.poses[i]
        q_x, q_y, q_Y_deg = q_pose
        q_Y_deg = (90+q_Y_deg)%360
        q_Y = q_Y_deg * imports.np.pi/180
        q_pose = imports.np.array([q_x, q_y, q_Y_deg])
        scatter_real_orientation(q_x, q_y, q_Y, "mediumturquoise", rad=10)
        imports.plt.scatter(real_data.poses[i, 0], real_data.poses[i, 1], c="blue", marker='o', linestyle='None', s =1)  

    for i in range(1200, 1500, 5):
        q_pose = real_data.poses[i]
        q_x, q_y, q_Y_deg = q_pose
        q_Y_deg = (90+q_Y_deg)%360
        q_Y = q_Y_deg * imports.np.pi/180
        q_pose = imports.np.array([q_x, q_y, q_Y_deg])
        scatter_real_orientation(q_x, q_y, q_Y, "mediumturquoise", rad=10)
        imports.plt.scatter(real_data.poses[i, 0], real_data.poses[i, 1], c="blue", marker='o', linestyle='None', s =1)