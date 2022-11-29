import math
from random import random
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import markers
import torch
import numpy as np
import os
from icecream import ic

# Device Auto-Configuration (Compatible with GPU training)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def checkpoint(model, bsz, optimizer, scheduler, current_epoch, logger, filename):
    out = os.path.join('./checkpoint/', filename.format(current_epoch))
    ic(f"Checkpoint {current_epoch} saved!")
    torch.save({'bsz': bsz,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'logger': logger
                }, out)


def sample_batch(dset, bsz):
    # num_pts = len(dset)
    # assert num_pts >= bsz
    # random_idx = np.random.choice(num_pts, bsz, replace=False)
    # ic("Starting sample")

    # Slow
    loader = torch.utils.data.DataLoader(
        dset, batch_size=bsz, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)
    # ic(iter(loader)[0])
    # ic(len(iter(loader)))
    return next(iter(loader))[0]


class Logger():
    # Customized logger class
    # TODO: (Xiaoyang) Discard this class and use TensorBoard later
    def __init__(self):
        super().__init__()
        # epochs-level logging
        self.tr_loss_epoch = []
        self.val_loss_epoch = []
        # steps-level logging: only useful for training loss
        self.tr_loss_steps = []
        # self.val_loss_steps = []
        # lr tracker
        self.lr = []
        # attack check
        self.attack = []

    def log_train_step(self, loss):
        self.tr_loss_steps.append(loss)

    def log_eval_epoch(self, loss):
        self.val_loss_epoch.append(loss)

    def log_train_epoch(self, loss):
        self.tr_loss_epoch.append(loss)

    def log_lr_epoch(self, lr):
        self.lr.append(lr)

    def log_attack_epoch(self, atk):
        self.attack.append(atk)


if __name__ == '__main__':
    ic("utils")
