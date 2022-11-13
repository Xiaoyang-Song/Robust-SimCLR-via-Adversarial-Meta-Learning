from random import sample
import torch
import torchvision
# Customized import
from models.identity import *
from models.rbsimclr import *
from data.dataset import *
from loss import PairwiseSimilarity, RBSimCLRLoss
import time
from utils import *


class MetaRBSimCLR(nn.Module):
    def __init__(self, model_config, meta_config, tri_args, device=DEVICE):
        self.device = device
        # Extract model configuration
        self.projection_dim = model_config['projection_dim']
        self.rbsimclr_isadv = model_config['adversarial']
        self.encoder_type = model_config['encoder']
        self.n_features = model_config['n_features']
        # Declare local & meta model
        self.local_model = RBSimCLR(self.projection_dim, self.rbsimclr_isadv,
                                    self.encoder_type, self.n_features)
        self.meta_model = RBSimCLR(self.projection_dim, self.rbsimclr_isadv,
                                   self.encoder_type, self.n_features)
        # Training params
        self.alpha = meta_config['alpha']
        self.beta = meta_config['beta']
        self.num_local_updates = meta_config['num_local_updates']
        self.max_epoch = meta_config['max_epoch']
        self.n_epoch_checkpoint = meta_config['n_epoch_checkpoint']
        # Loss & Optimizer
        self.criterion = RBSimCLRLoss(
            batch_size=self.sample_bsz, temperature=0.5)
        self.local_optimizer = None

        # Hyperparams
        self.num_atks_per_ep = tri_args['num_atks_per_ep']
        self.sample_bsz = tri_args['sample_bsz']

        # Buffer
        self.local_model_params = []

    def train(self, dset):
        for epoch in range(self.max_epoch):
            # atks = sample_attacks() TODO: (Xiaoyang) sample attacks
            attacks = None
            self.local_model_params = []
            # Do local updates for each attacks
            meta_model_state = self.meta_model.state_dict()
            for atk in attacks:
                # Load global model states
                self.local_model.load_state_dict(meta_model_state)
                self.local_optimzier = torch.nn.optim.SGD(
                    self.local_model.parameters(), self.alpha, momentum=0.9)
                # Local updates
                for local_steps in range(self.num_local_updates):
                    self.local_optimizer.zero_grad()
                    batch = sample_batch(dset, self.sample_bsz)
                    xi, xj, x = batch
                    x_i = x_i.squeeze().to(self.device).float()
                    x_j = x_j.squeeze().to(self.device).float()
                    # TODO:(Xiaoyang) replace with attacked images
                    x_adv = None
                    # Local forward pass
                    _, _, _, z_i, z_j, z_adv = self.local_model(
                        x_i, x_j, x_adv)
                    # Local loss
                    loss = self.criterion(z_i, z_j, z_adv)
                    loss.backward()
                    self.local_optimizer.step()

                # Save local model params
                self.local_model_params.append(self.local_model.state_dict())

            # Starting global updates
            for idx, atk in enumerate(attacks):
                # Load global model states
                self.local_model.load_state_dict(self.local_model_params[idx])
                # Sample batch of images
                batch = sample_batch(dset, self.sample_bsz)
                xi, xj, x = batch
                x_i = x_i.squeeze().to(self.device).float()
                x_j = x_j.squeeze().to(self.device).float()
                # TODO:(Xiaoyang) replace with attacked images
                x_adv = None
                # Forward pass
                _, _, _, z_i, z_j, z_adv = self.local_model(
                    x_i, x_j, x_adv)
                # Compute loss w.r.t optimal local model
                loss = self.criterion(z_i, z_j, z_adv)
                grad = torch.autograd.grad(loss, self.local_model.parameters())

                # Global update step:
                for layer_grad, param in zip(grad, self.meta_model.parameters()):
                    param.data -= self.beta * layer_grad

            # Checkpointing
            if (epoch+1) % self.n_epoch_checkpoint == 0:
                checkpoint(self.meta_model, None, None, epoch,
                           None, "MetaRBSimCLR_epoch_{}_checkpoint.pt")


if __name__ == '__main__':
    ic("meta_trainer")
    # Test sampling function
    # dataset = ContrastiveLearningDataset("./datasets")
    # num_views = 2
    # train_dataset = dataset.get_dataset('cifar10_tri', num_views)
    # batch = sample_batch(train_dataset, 256)
    # # batch = torch.utils.data.DataLoader(train_dataset, shuffle=True)
    # # ic(batch.shape)
    # ic(len(batch))
    # xi, xj, x = batch
    # ic(xi.shape)
    # ic(xj.shape)
    # ic(x.shape)

    # model_config = dict(
    #     projection_dim=128, adversarial=True, encoder=None, n_features=None
    # )
    # meta_config = dict(
    #     alpha=1e-3, beta=1e-3
    # )

    model = RBSimCLR(128)
    # x = torch.ones(
    #     (2, 3, 224, 224), dtype=torch.float32, requires_grad=True)
    # y = torch.ones(
    #     (2, 3, 224, 224), dtype=torch.float32, requires_grad=True)
    # z = torch.ones(
    #     (2, 3, 224, 224), dtype=torch.float32, requires_grad=True)
    # criterion = RBSimCLRLoss(batch_size=2, temperature=0.5)
    # _, _, _, zx, zy, zz = model(x, y, z)
    # loss = criterion(zx, zy, zz)
    # grad = torch.autograd.grad(loss, model.parameters())
    # for data, param in zip(grad, model.parameters()):
    #     print(data.data.shape)
    #     ic(param.data.shape)
    print(list(model.parameters())[-1])
    for param in model.parameters():
        param.data = torch.zeros(param.data.shape)
    print(list(model.parameters())[-1])
