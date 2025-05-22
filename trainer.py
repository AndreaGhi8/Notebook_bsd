# Ghiotto Andrea   2118418

import numpy as np
import torch
from tqdm import tqdm
import pickle
import gc

import model, metrics
from utils import visualizer

class Trainer:
    def __init__(self, writer, train_data, train_dataloader, val_data, net, optimizer, scheduler, drop, recocriterion, locacriterion):
        self.writer = writer
        self.train_data = train_data
        self.train_dataloader = train_dataloader
        self.val_data = val_data
        self.net = net
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.drop = drop
        self.recocriterion = recocriterion
        self.locacriterion = locacriterion
        self.best_loca_error = float("inf")
        self.best_model_path = None

    def train(self, num_epochs=24):
        for epoch in range(1, num_epochs + 1):
            print("epoch:", epoch)
            self.train_epoch(epoch)
            self.validate(epoch)
        return self.best_model_path

    def train_epoch(self, epoch):
        self.train_data.apply_random_rot = True
        self.net.train()
        train_losses = []

        for idx, (image, gtimage, gtpose, _, _, mode) in tqdm(enumerate(self.train_dataloader), total=len(self.train_dataloader)):
            image = self.drop(image.cuda())
            gtimage = gtimage.cuda()
            mode = mode[:, None].cuda()

            embed, rec = self.net(image, reco=True)

            distmat = torch.clamp(metrics.sonar_overlap_distance_matrix(gtpose, mode), 1e-4, 1).cuda()
            embedmat = torch.clamp(metrics.calcEmbedMatrix(embed), 0, 1)
            distmat, embedmat = mode * distmat, mode * embedmat

            loss_reco = self.recocriterion(rec[0], gtimage) + \
                        0.125 * self.recocriterion(rec[3], gtimage) + \
                        0.25 * self.recocriterion(rec[4], gtimage)
            loss_loca = self.locacriterion(distmat, embedmat)
            loss = loss_reco + loss_loca

            self.writer.add_scalar(f"Loss/recotrain_{str(epoch).zfill(2)}", loss_reco.item(), idx)
            self.writer.add_scalar(f"Loss/locatrain_{str(epoch).zfill(2)}", loss_loca.item(), idx)
            self.writer.add_scalar(f"Loss/losstrain_{str(epoch).zfill(2)}", loss.item(), idx)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            train_losses.append(loss.item())

            torch.cuda.empty_cache()

        print("train loss mean:", np.mean(train_losses))
        model.save_state(epoch, self.net, f"correct_model_3/epoch_{str(epoch).zfill(2)}.pth")

    def validate(self, epoch):
        self.net.eval()
        self.train_data.apply_random_rot = False
        self.train_data.computeDescriptors(self.net)
        self.val_data.computeDescriptors(self.net)

        with open("train_data.pickle", "wb") as handle:
            pickle.dump(self.train_data, handle)

        q_idx = 200
        q_image_a, q_image, q_pose, _, _, _ = self.train_data[q_idx]
        q_image_a = q_image_a[None].cuda()

        out = self.net.encoder(q_image_a)
        out[2][0, :, :, :] += torch.normal(0, 3, size=out[2].shape).cuda()
        out[2][0, :, :, :] += torch.normal(0, 3, size=out[2].shape).cuda()
        out[3][0, :, :, :] = 0

        q_desc = torch.nn.functional.normalize(self.net.embed(out[-1]).flatten(1), p=2, dim=1)
        q_image_r = self.net.decoder(out)[0]

        loca_errors, orie_errors = [], []

        print("computing metrics")
        for query_idx in tqdm(range(0, len(self.val_data))):
            loca_error, orie_error = visualizer.process(query_idx, self.net, self.train_data, self.val_data, plot=False)
            loca_errors.append(loca_error)
            orie_errors.append(orie_error)

        avg_loca_error = np.mean(loca_errors)
        avg_orie_error = np.mean(orie_errors)

        print(f"average localization error: {avg_loca_error:6.4f} meters")
        print(f"average orientation error : {avg_orie_error:6.4f} degrees")

        if avg_loca_error < self.best_loca_error:
            self.best_loca_error = avg_loca_error
            self.best_model_path = f"correct_model_3/epoch_{str(epoch).zfill(2)}.pth"

        del q_desc, q_image_a, q_image_r, out
        del loca_errors, orie_errors
        torch.cuda.empty_cache()
        gc.collect()