import math
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import markers
import torch
import numpy as np
import os

# Device Auto-Configuration (Compatible with GPU training)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def checkpoint(model, optimizer, scheduler, current_epoch, logger, filename):
    out = os.path.join('/checkpoint/', filename.format(current_epoch))

    torch.save({'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'logger': logger
                }, out)


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

    def log_train_step(self, loss):
        self.tr_loss_steps.append(loss)

    def log_eval_epoch(self, loss):
        self.val_loss_epoch.append(loss)

    def log_train_epoch(self, loss):
        self.tr_loss_epoch.append(loss)

    def log_lr_epoch(self, lr):
        self.lr.append(lr)
