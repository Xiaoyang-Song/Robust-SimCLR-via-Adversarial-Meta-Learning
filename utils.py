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
    def __init__(self):
        super().__init__()

    def log_train_step():
        pass

    def log_eval():
        pass

    def log_epoch():
        pass
