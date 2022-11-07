import torch
import torchvision
# Customized import
from models.rbsimclr import RBSimCLR
from data.dataset import *
from loss import PairwiseSimilarity, RBSimCLRLoss
import time
from utils import *

model = RBSimCLR(128)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, betas=(0.5, 0.999))

warmupscheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda epoch: (epoch+1)/10.0, verbose=True)

# SCHEDULER FOR COSINE DECAY
mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, 500, eta_min=0.05, last_epoch=-1, verbose=True)

# LOSS FUNCTION
criterion = RBSimCLRLoss(batch_size=128, temperature=0.5)


def RBSimCLR_trainer(model, train_loader, val_loader, optimizer, scheduler, criterion,
                     logger, checkpoint=None, max_epoch=100, device=DEVICE):
    if checkpoint is None:
        # Get SCHEDULER
        warmupscheduler = scheduler['warmupscheduler']
        mainscheduler = scheduler['mainscheduler']
    else:
        # TODO: (Xiaoyang) Load checkpoint
        pass
    # Basic stats
    current_epoch = 0
    for epoch in range(max_epoch):
        print(f"Epoch [{epoch}/{epochs}]\t")
        stime = time.time()
        model.train()
        tr_loss_epoch = 0
        # TODO: (Xiaoyang) Sample attacks here
        # e.g. Attacker = Attack(type)
        for step, (x_i, x_j, x) in enumerate(train_loader):
            optimizer.zero_grad()
            # Get augmented and attacked images
            x_i = x_i.squeeze().to(device).float()
            x_j = x_j.squeeze().to(device).float()
            # x_adv = Attacker(model, x, target)
            # Get latent representation
            z_i = model(x_i)
            z_j = model(x_j)
            # z_adv = model(x_adv)

            # loss = criterion(z_i, z_j, z_adv)
            loss.backward()
            optimizer.step()
        if epoch < 10:
            warmupscheduler.step()
        if epoch >= 10:
            mainscheduler.step()
        # EVALUATION
        model.eval()
        with torch.no_grad():
            val_loss_epoch = 0
            for step, (x_i, x_j, x) in enumerate(val_loader):

                x_i = x_i.squeeze().to(device).float()
                x_j = x_j.squeeze().to(device).float()
                # x_adv = Attacker(x)
                # Get latent representation
                z_i = model(x_i)
                z_j = model(x_j)
                # z_adv = model(x_adv)

                # loss = criterion(z_i, z_j, z_adv)
                val_loss_epoch += loss.item()

        if nr == 0:
            # tr_loss.append(tr_loss_epoch / len(dl))
            # val_loss.append(val_loss_epoch / len(vdl))
            print(
                f"Epoch [{epoch}/{epochs}]\t Training Loss: {tr_loss_epoch / len(dl)}\t lr: {round(lr, 5)}")
            print(
                f"Epoch [{epoch}/{epochs}]\t Validation Loss: {val_loss_epoch / len(vdl)}\t lr: {round(lr, 5)}")
            current_epoch += 1

        # dg.on_epoch_end()

    time_taken = (time.time()-stime)/60
    print(f"Epoch [{epoch}/{epochs}]\t Time Taken: {time_taken} minutes")


nr = 0
current_epoch = 0
epochs = 100
tr_loss = []
val_loss = []

for epoch in range(100):

    print(f"Epoch [{epoch}/{epochs}]\t")
    stime = time.time()

    model.train()
    tr_loss_epoch = 0

    for step, (x_i, x_j) in enumerate(train_loader):
        optimizer.zero_grad()
        x_i = x_i.squeeze().to('cuda:0').float()
        x_j = x_j.squeeze().to('cuda:0').float()

        # positive pair, with encoding
        z_i = model(x_i)
        z_j = model(x_j)

        loss = criterion(z_i, z_j)
        loss.backward()

        optimizer.step()

        if nr == 0 and step % 50 == 0:
            print(
                f"Step [{step}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")

        tr_loss_epoch += loss.item()

    if nr == 0 and epoch < 10:
        warmupscheduler.step()
    if nr == 0 and epoch >= 10:
        mainscheduler.step()

    lr = optimizer.param_groups[0]["lr"]

    # if nr == 0 and (epoch+1) % 50 == 0:
    # save_model(model, optimizer, mainscheduler, current_epoch,
    #            "SimCLR_CIFAR10_RN50_P128_LR0P2_LWup10_Cos500_T0p5_B128_checkpoint_{}_260621.pt")

    model.eval()
    with torch.no_grad():
        val_loss_epoch = 0
        for step, (x_i, x_j) in enumerate(valid_loader):

            x_i = x_i.squeeze().to('cuda:0').float()
            x_j = x_j.squeeze().to('cuda:0').float()

            # positive pair, with encoding
            z_i = model(x_i)
            z_j = model(x_j)

            loss = criterion(z_i, z_j)

            if nr == 0 and step % 50 == 0:
                print(
                    f"Step [{step}/{len(valid_loader)}]\t Loss: {round(loss.item(),5)}")

            val_loss_epoch += loss.item()

    if nr == 0:
        tr_loss.append(tr_loss_epoch / len(dl))
        val_loss.append(val_loss_epoch / len(vdl))
        print(
            f"Epoch [{epoch}/{epochs}]\t Training Loss: {tr_loss_epoch / len(dl)}\t lr: {round(lr, 5)}")
        print(
            f"Epoch [{epoch}/{epochs}]\t Validation Loss: {val_loss_epoch / len(vdl)}\t lr: {round(lr, 5)}")
        current_epoch += 1

    # dg.on_epoch_end()

    time_taken = (time.time()-stime)/60
    print(f"Epoch [{epoch}/{epochs}]\t Time Taken: {time_taken} minutes")

    # if (epoch+1) % 10 == 0:
    #     plot_features(model.pretrained, 10, 2048, 128)

# save_model(model, optimizer, mainscheduler, current_epoch,
#            "SimCLR_CIFAR10_RN50_P128_LR0P2_LWup10_Cos500_T0p5_B128_checkpoint_{}_260621.pt")

if __name__ == '__main__':
    ic("RBSimCLR trainer")
