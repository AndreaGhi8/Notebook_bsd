# Ghiotto Andrea   2118418

from Classes_and_functions import imports
from Classes_and_functions. Dataloader import functions

def process(q_idx, net, train_data, val_data, plot=True):

    if plot:
        functions.start_plot(train_data)
    print("its me")

    q_image_a, q_image, q_pose, _, _, _ = val_data[q_idx]
    q_image_a = q_image_a[None].cuda()
    if plot:
        q_desc, (q_image_r, _, _, _, _)  = net(q_image_a, reco=True)
    else:
        q_desc = net(q_image_a, reco=False)[0, :]
    q_desc = q_desc.detach().cpu().numpy()
    
    q_x, q_y, q_Y, q_Y_deg = functions.parse_pose(q_pose)
    if plot:
        functions.scatter_point(q_x, q_y, 'magenta', label="val pose (query)")
        functions.scatter_orientation(q_x, q_y, q_Y, "magenta")

    if plot:
        imports.plt.scatter(train_data.poses[:train_data.synth, 0], train_data.poses[:train_data.synth, 1], c="blue", marker='o', linestyle='None', s =1, label="training set positions")
        imports.plt.scatter(val_data.poses[:, 0], val_data.poses[:, 1], c="red", marker='o', linestyle='None', s =1, label="validation set positions")

    train_data.apply_random_rot = False
    minidx = train_data.query(q_desc)
    min_pose = train_data[minidx][2]
    min_x, min_y, min_Y, min_Y_deg = functions.parse_pose(min_pose)
    if plot:
        functions.scatter_point(min_x, min_y, 'gold', label="predicted pose")
        functions.scatter_orientation(min_x, min_y, min_Y, "gold")
    
    gt_pose_idx = functions.gtquery_process(train_data, q_x, q_y, q_Y_deg)
    gt_pose = train_data[gt_pose_idx][2]

    gt_x, gt_y, gt_Y, gt_Y_deg = functions.parse_pose(gt_pose)
    if plot:
                   
        functions.scatter_orientation(gt_x, gt_y, gt_Y, "green")
        functions.scatter_point(gt_x, gt_y, "green", label="database gt closest pose")
    
    mask3, iou = functions.generate_interference_mask(min_x, min_y, min_Y, min_Y_deg, q_x, q_y, q_Y, q_Y_deg)

    loca_error=imports.np.linalg.norm(gt_pose[:2]-min_pose[:2], ord=2)/10

    orie_error = gt_Y_deg - min_Y_deg.item()
    orie_error = imports.np.abs((orie_error + 180) % 360 - 180)

    gt_closest_image = train_data[val_data.closest_indices[q_idx]][1]

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
        axarr[1].imshow(gt_closest_image[0, :, :], cmap='gray')
        axarr[2].imshow(q_image.detach().cpu().numpy()[0, :, :], cmap='gray')
        
    return loca_error, orie_error

def check_process(gt_pose, train_data, val_data, plot=True):

    if plot:
        functions.start_plot(train_data)

    gt_x, gt_y, gt_Y, gt_Y_deg = functions.parse_pose(gt_pose)
    if plot:
                   
        functions.scatter_orientation(gt_x, gt_y, gt_Y, "green")
        functions.scatter_point(gt_x, gt_y, "green", label="database gt closest pose")
    
    if plot:
        imports.plt.scatter(train_data.poses[:train_data.synth, 0], train_data.poses[:train_data.synth, 1], c="blue", marker='o', linestyle='None', s =1, label="training set positions")
        imports.plt.scatter(val_data.poses[:, 0], val_data.poses[:, 1], c="red", marker='o', linestyle='None', s =1, label="validation set positions")

    q_pose_idx, min_diff_yaw = functions.gtquery_process_check(train_data, gt_x, gt_y, gt_Y_deg)
    if min_diff_yaw > 7.5:
        print("ERROR!! min_diff_yaw > 7.5")
    #else:
        #print("min_diff_yaw: ", min_diff_yaw)
    q_pose = train_data[q_pose_idx][2]

    q_x, q_y, q_Y, q_Y_deg = functions.parse_pose(q_pose)
    if plot:
        functions.scatter_point(q_x, q_y, 'magenta', label="val pose (query)")
        functions.scatter_orientation(q_x, q_y, q_Y, "magenta")
    
    mask3, iou = functions.generate_interference_mask(gt_x, gt_y, gt_Y, gt_Y_deg, q_x, q_y, q_Y, q_Y_deg)

    if plot:
        imports.plt.imshow(mask3, cmap="gray")
        imports.plt.legend(loc="lower right")
        imports.plt.figure()

def analyze_feature_robustness(train_data, net):
    q_idx = 200
    q_image_a, q_image, q_pose, _, _, _ = train_data[q_idx]
    q_image_a = q_image_a[None].cuda()

    out = net.encoder(q_image_a)
    print(out[0].shape, out[0].min(), out[0].max())
    print(out[1].shape, out[1].min(), out[1].max())
    print(out[2].shape, out[2].min(), out[2].max())
    print(out[3].shape, out[3].min(), out[3].max())
    out[2][0, :, :, :] = out[2][0, :, :, :] + imports.torch.normal(0, 3, size=out[2][:, :, :, :].shape).cuda()
    out[2][0, :, :, :] = out[2][0, :, :, :] + imports.torch.normal(0, 3, size=out[2][:, :, :, :].shape).cuda()
    out[3][0, :, :, :] = 0
    q_desc = imports.torch.nn.functional.normalize(net.embed(out[-1]).flatten(1), p=2, dim=1)
    q_image_r = net.decoder(out)[0]

    print(q_image_r[0].min(), q_image_r[0].max())

    f, axarr = imports.plt.subplots(1, 2, figsize=(15, 15))
    axarr[0].set_title("query image")
    axarr[1].set_title("reco image")

    axarr[0].imshow(q_image.detach().cpu().numpy()[0, :, :], cmap='gray')
    axarr[1].imshow(q_image_r.detach().cpu().numpy()[0, 0, :, :], cmap='gray')