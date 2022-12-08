import torch.nn as nn
import torchvision
# from simclr.modules.resnet_hacks import modify_resnet_model
from models.identity import Identity
from models.resnet import *

# Credit: the RBSimCLR class (including resnet.py & identity.py) is MODIFIED based on this repo:
# 1. https://github.com/Spijkervet/SimCLR/blob/master/simclr/modules


class RBSimCLR(nn.Module):
    # TODO: (Xiaoyang) test this implementation
    def __init__(self, projection_dim, adversarial=True, encoder=None, n_features=None):
        super(RBSimCLR, self).__init__()

        self.adversarial = adversarial

        # Handle default backbone
        if encoder is None:
            self.encoder = get_resnet('resnet18')
            # self.encoder = get_resnet('resnet50')
        else:
            self.encoder = encoder
        # Handle default n_features
        if n_features is not None:
            self.n_features = n_features
        else:
            self.n_features = self.encoder.fc.in_features

        # Replace the fc layer with an Identity function
        self.encoder.fc = Identity()

        # We use a MLP with one hidden layer to obtain z_i = g(h_i) = W(2)σ(W(1)h_i) where σ is a ReLU non-linearity.
        self.projector = nn.Sequential(
            nn.Linear(self.n_features, self.n_features, bias=False),
            nn.ReLU(),
            nn.Linear(self.n_features, projection_dim, bias=False),
        )

    def forward(self, x_i, x_j, x_adv=None):
        # Augmented branches
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)
        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        # adversarial branch
        # if self.adversarial:
        if x_adv is not None:
            #assert x_adv is not None
            h_adv = self.encoder(x_adv)
            z_adv = self.projector(h_adv)
            return h_i, h_j, h_adv, z_i, z_j, z_adv
        return h_i, h_j, z_i, z_j


if __name__ == '__main__':
    model = RBSimCLR(256)
    ic(model.encoder)
