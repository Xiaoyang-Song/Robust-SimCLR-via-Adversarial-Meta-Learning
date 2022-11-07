import logging
import numpy as np
import os
import sys
import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.distributed as dist
from icecream import ic
from data.dataset import *

# Credit: the PairwiseSimilarity class is modified based on these two sources:
# 1. https://medium.com/the-owl/simclr-in-pytorch-5f290cb11dd7
# 2. https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules/nt_xent.py


class PairwiseSimilarity(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)

        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        # assertion check
        assert len(z_i.shape) == 2
        assert len(z_j.shape) == 2

        N = 2 * self.batch_size

        z = torch.cat((z_i, z_j), dim=0)
        # ic(z.shape)
        # ic(z)
        sim = self.similarity_f(z.unsqueeze(
            1), z.unsqueeze(0)) / self.temperature
        # ic(sim)
        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)
        # ic(sim_i_j)
        # ic(sim_j_i)
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)
        # ic(positive_samples)
        # ic(negative_samples)
        # SIMCLR pairwise loss
        labels = torch.from_numpy(
            np.array([0]*N)).reshape(-1).to(positive_samples.device).long()  # .float()

        logits = torch.cat((positive_samples, negative_samples), dim=1)
        ic(logits)
        ic(labels)
        loss = self.criterion(logits, labels)
        loss /= N

        return loss


class RBSimCLRLoss(nn.Module):
    def __init__(self, batch_size, temperature):
        super().__init__()
        self.bsz = batch_size
        self.temperature = temperature
        # Define pairwise similarity
        self.SIM = PairwiseSimilarity(self.bsz, self.temperature)

    def forward(self, z1, z2, z3):
        # By default z3 is the adversarial branch
        # TODO: (Xiaoyang) modify this to enable weighting
        return self.SIM(z1, z2) + self.SIM(z1, z3) + self.SIM(z2, z3)


if __name__ == '__main__':
    ic("loss.py")
    # Get dataset for testing
    # dataset = ContrastiveLearningDataset("./datasets")
    # num_views = 2
    # train_dataset = dataset.get_dataset('cifar10', num_views)
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=32, shuffle=True,
    #     num_workers=2, pin_memory=True, drop_last=True)
    # tri_img, tri_lbl = next(iter(train_loader))
    # ic(len(tri_img))
    # ic(tri_img[0].shape)
    # ic(tri_img[1].shape)
    # cast to torch.Tensor: not necesary
    # tri_img = torch.cat(tri_img)
    # ic(tri_img.shape)

    # Test pairwise similarity & RBSimCLRLoss
    z_i = torch.tensor([[1, 2, 3], [4, 5, 6], [4, 5, 6]], dtype=torch.float32)
    z_j = torch.tensor([[2, 3, 4], [4, 5, 6], [4, 5, 6]], dtype=torch.float32)
    z_k = torch.tensor([[2, 3, 4], [4, 5, 6], [4, 5, 6]], dtype=torch.float32)
    SIM = PairwiseSimilarity(3, 1)
    loss1 = SIM(z_i, z_j)
    loss2 = SIM(z_i, z_k)
    loss3 = SIM(z_k, z_j)
    ic(loss1)
    ic(loss2)
    ic(loss3)
    RBSIM = RBSimCLRLoss(3, 1)
    ic(RBSIM(z_i, z_j, z_k))
