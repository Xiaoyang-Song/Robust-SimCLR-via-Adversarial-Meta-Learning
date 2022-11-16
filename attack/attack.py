from enum import Enum
import torch
import torch.nn.functional as F
from loss import RBSimCLRLoss
import torch.nn as nn
from icecream import ic
from utils import *
from copy import deepcopy
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

    def perturb(self, input_model, original_images, target):
        # original_images: values are within self.min_val and self.max_val
        # The adversaries created from random close points to the original data
        model = deepcopy(input_model)

        if self.random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.to(DEVICE)
            x = original_images.clone() + rand_perturb
            x = torch.clamp(x, self.min_val, self.max_val)
        else:
            x = original_images.clone()

        x.requires_grad = True

        model.eval()


        #todo: why severl iteraaions?


        #model.zero_grad()
        _, _, z_i, z_j = model(x, target)


        loss = F.mse_loss(z_i, z_j)

        grad_outputs = None
        grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, only_inputs=True, retain_graph=False)[0]

        if self._type == 'linf':
            scaled_g = torch.sign(grads.data)

        x.data += self.alpha * scaled_g

        x = torch.clamp(x, self.min_val, self.max_val)

        #model.train()


        # outputs = model(x)
        #
        # loss = F.mse_loss(outputs, model(target))


        return x.detach()
class PGDAttack(nn.Module):
    def __init__(self, batch_size, loss_type, epsilon=0.0314, alpha=0.07, min_val=0.0, max_val=0.7,
                 max_iters=10,  temperature=0.5, random_start = True, _type='linf'):
        super(PGDAttack, self).__init__()

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
        # loss type
        self.loss_type = loss_type
        self.type_attack = TypeAttack.PGD
        #self.optimizer = optimizer
        self.batch_size =batch_size
        self.temperature = temperature
        self.random_start = random_start

    def perturb(self,input_model, original_images, target):

        model = deepcopy(input_model)
        if self.random_start:
            rand_perturb = torch.FloatTensor(original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            if torch.cuda.is_available():
                rand_perturb = rand_perturb.float().to(DEVICE)
            x = original_images.float().clone() + rand_perturb
            x = torch.clamp(x, self.min_val, self.max_val)

        else:
            x = original_images.clone()

        x.requires_grad = True

        model.eval()

        #batch_size = self.batch_size
        criteria = RBSimCLRLoss(self.batch_size, self.temperature)
        with torch.enable_grad():
            for _iter in range(self.max_iters):

                #model.zero_grad()

                _, _, z_i, z_j = model(x, target)


                if self.loss_type == 'mse':
                    loss = F.mse_loss(z_i, z_j)
                elif self.loss_type == 'sim':

                    loss = criteria.forward(z1 = z_i, z2 = z_j, z3 = None)


                elif self.loss_type == 'l1':
                    loss = F.l1_loss(z_i, z_j)

                elif self.loss_type == 'cos':
                    loss = 1 - F.cosine_similarity(z_i, z_j).mean()

                grads = torch.autograd.grad(loss, x, grad_outputs=None, only_inputs=True, retain_graph=False)[0]

                if self._type == 'linf':
                    scaled_g = torch.sign(grads.data)

                x.data += self.alpha * scaled_g

                x = torch.clamp(x, self.min_val, self.max_val)
                x = project(x, original_images, self.epsilon, self._type)

        #model.train()

        #optimizer.zero_grad()

        # if self.loss_type == 'mse':
        #     loss = F.mse_loss(model(x), model(target)) * (1.0 / batch_size)
        # elif self.loss_type == 'sim':
        #
        #
        #     loss = criteria.forward(z1=model(x), z2=model(original_images), z3=None)
        # elif self.loss_type == 'l1':
        #     loss = F.l1_loss(model(x), model(target)) * (1.0 / batch_size)
        # elif self.loss_type == 'l2':
        #     loss = F.l2_loss(model(x), model(target))* (1.0 / batch_size)
        # elif self.loss_type == 'cos':
        #     loss = 1 - F.cosine_similarity(model(x), model(target)).sum() * (
        #                 1.0 / batch_size)
        # #todo: do I need to return loss?
        return x.detach()



