from enum import Enum
import torch
import torch.nn.functional as F
from loss import RBSimCLRLoss
import torch.nn as nn
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

    def __init__(self, model, linear,original_images, labels, epsilon, alpha, min_val, max_val, max_iters, _type='linf', reduction4loss='mean', random_start=True):

        # Model
        self.model = model
        self.linear = linear
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

    def perturb(self  ):
        # original_images: values are within self.min_val and self.max_val
        # The adversaries created from random close points to the original data
        #todo:do we need loss or just provide x
        if self.random_start:
            rand_perturb = torch.FloatTensor(self.original_images.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.cuda()
            x = self.original_images.clone() + rand_perturb
            x = torch.clamp(x, self.min_val, self.max_val)
        else:
            x = self.original_images.clone()

        x.requires_grad = True

        self.model.eval()
        if not self.linear == 'None':
            self.linear.eval()
        #todo: why severl iteraaions?


        self.model.zero_grad()
        if not self.linear == 'None':
            self.linear.zero_grad()

        if self.linear == 'None':
            outputs = self.model(x)
        else:
            outputs = self.linear(self.model(x))

        loss = F.cross_entropy(outputs, self.labels, reduction=self.reduction4loss)

        grad_outputs = None
        grads = torch.autograd.grad(loss, x, grad_outputs=grad_outputs, only_inputs=True, retain_graph=False)[0]

        if self._type == 'linf':
            scaled_g = torch.sign(grads.data)

        x.data += self.alpha * scaled_g

        x = torch.clamp(x, self.min_val, self.max_val)
        x = project(x, self.original_images, self.epsilon, self._type)
        #not sure why here and just check the loss
        self.model.train()
        self.projector.train()
        self.optimizer.zero_grad()
        if self.linear == 'None':
            outputs = self.model(x)
        else:
            outputs = self.linear(self.model(x))
        loss = F.cross_entropy(outputs, self.labels, reduction=self.reduction4loss)


        return x.detach(), loss


class PGDAttack(nn.Module):
    def __init__(self, model, ori, target,  projector, epsilon, alpha, min_val, max_val,
                 max_iters,optimizer, batch_size, temperature,
                 random_start = True, _type='linf', loss_type='sim', regularize='original'):
        # Constructor for
        self.model = model
        self.ori = ori
        self.target = target
        self.projector = projector
        self.regularize = regularize
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

    def get_loss(self):
        if self.random_start:
            rand_perturb = torch.FloatTensor(self.ori.shape).uniform_(
                -self.epsilon, self.epsilon)
            rand_perturb = rand_perturb.float().cuda()
            x = self.ori.float().clone() + rand_perturb
            x = torch.clamp(x, self.min_val, self.max_val)
        else:
            x = self.ori.clone()

        x.requires_grad = True

        self.model.eval()
        self.projector.eval()
        batch_size = self.batch_size

        with torch.enable_grad():
            for _iter in range(self.max_iters):

                self.model.zero_grad()
                self.projector.zero_grad()

                if self.loss_type == 'mse':
                    loss = F.mse_loss(self.projector(self.model(x)), self.projector(self.model(self.target)))
                elif self.loss_type == 'sim':
                    #inputs = torch.cat((x, self.target))
                    #output = self.projector(self.model(inputs))
                    loss = RBSimCLRLoss(self.temperature, self. batch_size )

                elif self.loss_type == 'l1':
                    loss = F.l1_loss(self.projector(self.model(x)), self.projector(self.model(self.target)))
                elif self.loss_type == 'l2':
                    loss = F.l2_loss(self.projector(self.model(x)), self.projector(self.model(self.target)))
                elif self.loss_type == 'cos':
                    loss = 1 - F.cosine_similarity(self.projector(self.model(x)),
                                                   self.projector(self.model(self.target))).mean()

                grads = torch.autograd.grad(loss, x, grad_outputs=None, only_inputs=True, retain_graph=False)[0]

                if self._type == 'linf':
                    scaled_g = torch.sign(grads.data)

                x.data += self.alpha * scaled_g

                x = torch.clamp(x, self.min_val, self.max_val)
                x = project(x, self.ori, self.epsilon, self._type)

        self.model.train()
        self.projector.train()
        self.optimizer.zero_grad()

        if self.loss_type == 'mse':
            loss = F.mse_loss(self.projector(self.model(x)), self.projector(self.model(self.target))) * (1.0 / batch_size)
        elif self.loss_type == 'sim':

            #inputs = torch.cat((x, self.target))
            #output = self.projector(self.model(inputs))
            #similarity, _ = pairwise_similarity(output, temperature=0.5, multi_gpu=False, adv_type='None')
            loss = RBSimCLRLoss(self.temperature, self. batch_size )
        elif self.loss_type == 'l1':
            loss = F.l1_loss(self.projector(self.model(x)), self.projector(self.model(self.target))) * (1.0 / batch_size)
        elif self.loss_type == 'l2':
            loss = F.l2_loss(self.projector(self.model(x)), self.projector(self.model(self.target)))* (1.0 / batch_size)
        elif self.loss_type == 'cos':
            loss = 1 - F.cosine_similarity(self.projector(self.model(x)), self.projector(self.model(self.target))).sum() * (
                        1.0 / batch_size)

        return x.detach(), loss

