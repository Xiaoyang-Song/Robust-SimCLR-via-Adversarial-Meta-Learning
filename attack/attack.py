from enum import Enum
import torch
import torch.nn.functional as F
from loss import RBSimCLRLoss
import torch.nn as nn
from icecream import ic
from utils import *
def project(x, original_x, epsilon, _type='linf'):
    if _type == 'linf':
        max_x = original_x + epsilon
        min_x = original_x - epsilon

        x = torch.max(torch.min(x, max_x), min_x)
    else:
        raise NotImplementedError

    return x
class TypeAttack(Enum):
    FGSM = 1
    PGD = 2
    FreeAT = 3


class FGSMAttack(nn.Module):
    """
        Fast gradient sign untargeted adversarial attack, minimizes the initial class activation
        with iterative grad sign updates
    """

    def __init__(self, epsilon=0.0314, alpha=0.07, min_val=0.0, max_val=1.0, max_iters=10, _type='linf',  random_start=True):

        # Model
        super(FGSMAttack, self).__init__()
        #self.model = model
        #self.linear = linear
        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        self._type = _type

        self.random_start = random_start
        #self.original_images = original_images
        #self.labels = labels

    def perturb(self, model, original_images, target):
        # original_images: values are within self.min_val and self.max_val
        # The adversaries created from random close points to the original data


        if self.random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            #rand_perturb = rand_perturb.cuda()
            x = original_images.clone() + rand_perturb
            x = torch.clamp(x, self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True

        model.eval()
        # if not self.linear == 'None':
        #     self.linear.eval()
        #todo: why severl iteraaions?


        model.zero_grad()
        outputs = model(x)

        # if not self.linear == 'None':
        #     self.linear.zero_grad()

        # if self.linear == 'None':
        #     outputs = model(x)
        # else:
        #     outputs = self.linear(model(x))

        loss = F.mse_loss(outputs, model(target))

        grad_outputs = None
        grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, only_inputs=True, retain_graph=False)[0]

        if self._type == 'linf':
            scaled_g = torch.sign(grads.data)

        x.data += self.alpha * scaled_g

        x = torch.clamp(x, self.min_val, self.max_val)

        #x = project(x, self.original_images, self.epsilon, self._type)
        #not sure why here and just check the loss
        model.train()

        #self.projector.train()
        #self.optimizer.zero_grad()
        outputs = model(x)
        # if self.linear == 'None':
        #     outputs = model(x)
        # else:
        #     outputs = self.linear(model(x))
        #loss = F.cross_entropy(outputs, self.labels, reduction=self.reduction4loss)
        loss = F.mse_loss(outputs, model(target))


        return x.detach(), loss
class PGDAttack(nn.Module):
    def __init__(self, epsilon, alpha, min_val, max_val,
                 max_iters,optimizer, batch_size, temperature,
                 random_start = True, _type='linf', loss_type='sim'):
        # Constructor for
        super(PGDAttack, self).__init__()
        # self.model = model
        # self.ori = ori
        # self.target = target
        #self.projector = projector
        # Maximum perturbation
        self.epsilon = epsilon
        # Movement multiplier per iteration
        self.alpha = alpha
        # Minimum value of the pixels
        self.min_val = min_val
        # Maximum value of the pixels
        self.max_val = max_val
        # Maximum numbers of iteration to generated adversaries
        self.max_iters = max_iters
        # The perturbation of epsilon
        #Todo: Jing l2 distance
        self._type = _type
        # loss type
        self.loss_type = loss_type
        self.type_attack = TypeAttack.PGD
        self.optimizer = optimizer
        self.batch_size =batch_size
        self.temperature = temperature
        self.random_start = random_start

    def perturb(self,model, original_images, target):
        if self.random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            if torch.cuda.is_available():
                rand_perturb = rand_perturb.float().to(DEVICE)
            x = original_images.float().clone() + rand_perturb
            x = torch.clamp(x, self.min_val, self.max_val)
            print("shape of x", x.shape)
        else:
            x = original_images.clone()

        x.requires_grad = True

        model.eval()
        #self.projector.eval()
        batch_size = self.batch_size
        criteria = RBSimCLRLoss(self.batch_size, self.temperature)
        with torch.enable_grad():
            for _iter in range(self.max_iters):

                model.zero_grad()
                #self.projector.zero_grad()

                if self.loss_type == 'mse':
                    loss = F.mse_loss(model(x), model(target))
                elif self.loss_type == 'sim':


                    ic(model(x).shape)
                    print(model(original_images).shape)
                    loss = criteria.forward(z1 = model(x), z2 = model(original_images), z3 = None)


                elif self.loss_type == 'l1':
                    loss = F.l1_loss(model(x), model(target))

                elif self.loss_type == 'cos':
                    loss = 1 - F.cosine_similarity(model(x),
                                                   model(target)).mean()

                grads = torch.autograd.grad(loss, x, grad_outputs=None, only_inputs=True, retain_graph=False)[0]

                if self._type == 'linf':
                    scaled_g = torch.sign(grads.data)

                x.data += self.alpha * scaled_g

                x = torch.clamp(x, self.min_val, self.max_val)
                x = project(x, original_images, self.epsilon, self._type)

        model.train()
        #self.projector.train()
        self.optimizer.zero_grad()

        if self.loss_type == 'mse':
            loss = F.mse_loss(model(x), model(target)) * (1.0 / batch_size)
        elif self.loss_type == 'sim':


            loss = criteria.forward(z1=model(x), z2=model(original_images), z3=None)
        elif self.loss_type == 'l1':
            loss = F.l1_loss(model(x), model(target)) * (1.0 / batch_size)
        elif self.loss_type == 'l2':
            loss = F.l2_loss(model(x), model(target))* (1.0 / batch_size)
        elif self.loss_type == 'cos':
            loss = 1 - F.cosine_similarity(model(x), model(target)).sum() * (
                        1.0 / batch_size)

        return x.detach(), loss


#         return x.detach(), loss

