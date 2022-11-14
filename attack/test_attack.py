from data.dataset import CIFAR10
from models.resnet import get_resnet
from data.dataset import ContrastiveLearningDataset
import models
import torch
import matplotlib.pyplot as plt
import numpy as np
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from icecream import ic
from models.projector import Projector
from collections import OrderedDict
from attack import PGDAttack, FGSMAttack
from utils import *
from PIL import Image
import PIL

import data
use_cuda = torch.cuda.is_available()
if use_cuda:
    ngpus_per_node = torch.cuda.device_count()
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

model= get_resnet("resnet50", True)
Linear = nn.Sequential(nn.Linear(512 , 10))


dataset = ContrastiveLearningDataset("./datasets")
num_views = 2
train_dataset = dataset.get_dataset('cifar10_tri', num_views)
train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=2, shuffle=True)
x_i, x_j, x= next(iter(train_loader))[0]
img_view = x_i
#img_view = torch.stack(train_dataset[0][0], train_dataset[0][0])

criterion = nn.CrossEntropyLoss()
use_cuda = torch.cuda.is_available()
if use_cuda:
    ngpus_per_node = torch.cuda.device_count()
    model.cuda()
    Linear.cuda()
    cudnn.benchmark = True

#attacker = PGDAttack(model, Linear, epsilon=0.0314, alpha=0.007, min_val=0.0, max_val=1.0, max_iters=10, _type="linf")
def test_singleimgpgd(img,idx, losstype = "mse"):
    #print(img.shape)
    base_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    projector = Projector(1)
    name = "pgd"
    attacker = PGDAttack(epsilon=0.0314, alpha=0.007,
                         min_val=0.0, max_val=1.0, max_iters=10, optimizer=base_optimizer, batch_size=2,
                         temperature=0.5, loss_type=losstype
                         )

    adv_inputs, _ = attacker.perturb(model, img, img)
    plt.imshow(np.array(img[idx].reshape(224, 224, 3)))
    fn1 = name + losstype + str(idx) + ".png"
    plt.savefig(fn1)

    #plt.show()
    plt.imshow(np.array(adv_inputs[idx].reshape(224, 224, 3)))
    fn2 = name + losstype+"_adv" + str(idx) + ".png"
    plt.savefig(fn2)
    #plt.show()


def test_singleimgfgsm(img,idx):



    attacker = FGSMAttack( epsilon=0.0314, alpha=0.007,
                         min_val=0.0, max_val=1.0, max_iters=10)

    name = "fgsm"
    adv_inputs, _ = attacker.perturb(model, img, img)
    plt.imshow(np.array(img[idx].reshape(224, 224, 3)))
    fn1 = name  + str(idx) + ".png"
    plt.savefig(fn1)

    #plt.show()
    plt.imshow(np.array(adv_inputs[idx].reshape(224, 224, 3)))
    fn2 = name +"_adv" + str(idx) + ".png"
    plt.savefig(fn2)
    #plt.show()


test_singleimgpgd(img_view,0, "mse")
test_singleimgpgd(img_view,0, "l1")
test_singleimgpgd(img_view,0, "cos")


test_singleimgfgsm(img_view, 0)
test_singleimgpgd(img_view,0, "sim")
