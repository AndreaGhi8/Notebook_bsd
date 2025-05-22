# Ghiotto Andrea   2118418

import numpy as np
import torch
from tqdm import tqdm
import pickle
import gc

import model, metrics
from utils import visualizer

def train(writer, train_data, train_dataloader, val_data, net, optimizer, scheduler, drop, recocriterion, locacriterion):
    best_loca_error = float("inf")
    best_model_path = None
    
    for epoch in range(1, 25):
        print("epoch: ", epoch)
        train_epochs(epoch, writer, train_data, train_dataloader, val_data, net, optimizer, scheduler, drop, recocriterion, locacriterion)
        best_model_path = validate(epoch, train_data, val_data, best_loca_error, best_model_path)
    
    return best_model_path

def train_epochs(epoch, writer, train_data, train_dataloader, val_data, net, optimizer, scheduler, drop, recocriterion, locacriterion):
    train_data.apply_random_rot = True
    net.train()

    train_losses = []

    for idx, (image, gtimage, gtpose, _, _, mode) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
        image = drop(image.cuda())
        gtimage = gtimage.cuda()
        mode = mode[:, None].cuda()
            
        embed, rec = net(image, reco=True)

        distmat  = torch.clamp(metrics.sonar_overlap_distance_matrix(gtpose, mode), 1e-4, 1).cuda()
        embedmat = torch.clamp(metrics.calcEmbedMatrix(embed), 0, 1)

        distmat, embedmat = mode*distmat, mode*embedmat
            
        loss_reco = recocriterion(rec[0], gtimage) + 0.125*recocriterion(rec[3], gtimage) + 0.25*recocriterion(rec[4], gtimage)
        loss_loca = locacriterion(distmat, embedmat)
        loss = loss_reco + loss_loca
        writer.add_scalar(f"Loss/recotrain_{str(epoch).zfill(2)}", loss_reco.item(), idx)
        writer.add_scalar(f"Loss/locatrain_{str(epoch).zfill(2)}", loss_loca.item(), idx)
        writer.add_scalar(f"Loss/losstrain_{str(epoch).zfill(2)}", loss.item(), idx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_losses.append(loss.item())

        torch.cuda.empty_cache()

    print("train loss mean:", np.array(train_losses).mean())
        
    model.save_state(epoch, net, f"correct_model_3/epoch_{str(epoch).zfill(2)}.pth")

def validate(epoch, train_data, val_data, best_loca_error, best_model_path):
    checkpoint_path = f"correct_model_3/epoch_{str(epoch).zfill(2)}.pth"
    net = model.Model()
    model.load_state(net, checkpoint_path)
    net = net.cuda()
    net.eval()

    train_data.apply_random_rot = False
    train_data.computeDescriptors(net)

    val_data.computeDescriptors(net)

    with open("train_data.pickle", "wb") as handle:
        pickle.dump(train_data, handle)

    q_idx = 200
    q_image_a, q_image, q_pose, _, _, _ = train_data[q_idx]
    q_image_a = q_image_a[None].cuda()

    out = net.encoder(q_image_a)
    out[2][0, :, :, :] = out[2][0, :, :, :] + torch.normal(0, 3, size=out[2][:, :, :, :].shape).cuda()
    out[2][0, :, :, :] = out[2][0, :, :, :] + torch.normal(0, 3, size=out[2][:, :, :, :].shape).cuda()
    out[3][0, :, :, :] = 0
    q_desc = torch.nn.functional.normalize(net.embed(out[-1]).flatten(1), p=2, dim=1)
    q_image_r = net.decoder(out)[0]

    loca_errors, orie_errors = [], []

    print("computing metrics")
    for query_idx in tqdm(range(0, len(val_data))):
        loca_error, orie_error = visualizer.process(query_idx, net, train_data, val_data, plot=False)
        loca_errors.append(loca_error)
        orie_errors.append(orie_error)

    avg_loca_error = np.array(loca_errors).mean()
    avg_orie_error = np.array(orie_errors).mean()

    print(f"average localization error: {avg_loca_error:6.4f} meters")
    print(f"average orientation error : {avg_orie_error:6.4f} degrees")

    if avg_loca_error < best_loca_error:
        best_loca_error = avg_loca_error
        best_model_path = checkpoint_path
    
    del net
    del q_desc, q_image_a, q_image_r, out
    del loca_errors, orie_errors

    torch.cuda.empty_cache()
    gc.collect()

    return best_model_path