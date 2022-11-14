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
    attacker = PGDAttack(model=model, ori=img, target=img, epsilon=0.0314, alpha=0.007,
                         min_val=0.0, max_val=1.0, max_iters=10, optimizer=base_optimizer, batch_size=2,
                         temperature=0.5, loss_type=losstype
                         )

    adv_inputs, _ = attacker.perturb()
    plt.imshow(np.array(img[idx].reshape(224, 224, 3)))
    fn1 = name + losstype + str(idx) + ".png"
    plt.savefig(fn1)

    #plt.show()
    plt.imshow(np.array(adv_inputs[idx].reshape(224, 224, 3)))
    fn2 = name + losstype+"_adv" + str(idx) + ".png"
    plt.savefig(fn2)
    #plt.show()


def test_singleimgfgsm(img,idx):

    base_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

    name = "pgd"
    attacker = FGSMAttack(model=model, linear = 'None', original_images=img, labels=img,  epsilon=0.0314, alpha=0.007,
                         min_val=0.0, max_val=1.0, max_iters=10)

    # def __init__(self, model, linear, original_images, labels, epsilon, alpha, min_val, max_val, max_iters,
    #              _type='linf', reduction4loss='mean', random_start=True):

    name = "fgsm"
    adv_inputs, _ = attacker.perturb()
    plt.imshow(np.array(img[idx].reshape(224, 224, 3)))
    fn1 = name  + str(idx) + ".png"
    plt.savefig(fn1)

    #plt.show()
    plt.imshow(np.array(adv_inputs[idx].reshape(224, 224, 3)))
    fn2 = name +"_adv" + str(idx) + ".png"
    plt.savefig(fn2)
    #plt.show()


# test_singleimgpgd("pgd", img_view,0, "mse")
# test_singleimgpgd("pgd", img_view,0, "l1")
# test_singleimgpgd("pgd", img_view,0, "cos")


#test_singleimgfgsm(img_view, 0)
test_singleimgpgd(img_view,0, "sim")
# def test(name, losstype="mse" ):
#     global best_acc
#
#     model.eval()
#     Linear.eval()
#
#     test_clean_loss = 0
#     test_adv_loss = 0
#     clean_correct = 0
#     adv_correct = 0
#     clean_acc = 0
#     total = 0
#     base_optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
#
#     for idx, (image, label) in enumerate(val_loader):
#
#
#         img = image
#         y = label
#         print(y)
#         print(img.shape)
#         total += y.size(0)
#         # ic(img.shape)
#         # ic(model(img).shape)
#         # out = Linear(model(img))
#         # _, predx = torch.max(out.data, 1)
#         # clean_loss = criterion(out, y)
#         #
#         # clean_correct += predx.eq(y.data).cpu().sum().item()
#         #
#         # clean_acc = 100. * clean_correct / total
#         #
#         # test_clean_loss += clean_loss.data
#         projector = Projector(1)
#
#         if name == "pgd":
#             attacker = PGDAttack(model = model, ori = image, target = image, projector = projector, epsilon=0.0314, alpha=0.007,
#                                  min_val=0.0, max_val=1.0, max_iters=10,optimizer= base_optimizer, batch_size= 2,
#                                  temperature=0.5, loss_type = losstype
#                                 )
#
#         # self, model, ori, target, projector, epsilon, alpha, min_val, max_val,
#         # max_iters, optimizer, batch_size, temperature,
#         # random_start = True, _type = 'linf', loss_type = 'sim', regularize = 'original'
#         adv_inputs, _= attacker.perturb()
#         print("shape of adv_Input", adv_inputs.shape)
#         out = model(adv_inputs)
#         # _, index = torch.max(out, 1)
#         # percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
#         print("out in the followinf ")
#         print(out.shape) #todo why 2*1000? cuz pretrained
#         # print(percentage)
#         # print(y[index[0]], percentage[index[0]].item())
#         #print(out.shape)
#         #ic(len(out))
#         #print(adv_inputs.shape)
#         # plt.imshow(np.array(cifar_tri[0][0].permute(1, 2, 0)))
#         _, predx = torch.max(out.data, 1)
#         adv_loss = criterion(out, y)
#
#         adv_correct += predx.eq(y.data).cpu().sum().item()
#         adv_acc = 100. * adv_correct / total
#
#         test_adv_loss += adv_loss.data
#         print('hi')
#         break
#
#
#     print("Test accuracy: {0}/{1}".format(clean_acc, adv_acc))
#
#     return (clean_acc, adv_acc)

