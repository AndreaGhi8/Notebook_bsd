# Ghiotto Andrea   2118418

import numpy as np
import torch
from tqdm import tqdm

from Classes_and_functions. Model import model_classes
from Classes_and_functions.Training import functions

def train_epochs(writer, train_data, train_dataloader, net, optimizer, scheduler, drop, recocriterion, locacriterion):
    for epoch in range(1, 25):

        train_data.apply_random_rot = True
        net.train()

        train_losses = []

        for idx, (image, gtimage, gtpose, _, _, mode) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            image = drop(image.cuda())
            gtimage = gtimage.cuda()
            mode = mode[:, None].cuda()
            
            embed, rec = net(image, reco=True)

            distmat  = torch.clamp(functions.sonar_overlap_distance_matrix(gtpose, mode), 1e-4, 1).cuda()
            embedmat = torch.clamp(functions.calcEmbedMatrix(embed), 0, 1)

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
        
        model_classes.save_state(epoch, net, f"correct_model_3/epoch_{str(epoch).zfill(2)}.pth")