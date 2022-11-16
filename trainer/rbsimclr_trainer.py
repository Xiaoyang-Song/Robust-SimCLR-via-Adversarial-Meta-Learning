import torch
import torchvision
# Customized import
from models.identity import *
from models.RBSimCLR import *
from data.dataset import *
from loss import PairwiseSimilarity, RBSimCLRLoss
import time
from utils import *
from attack.attack import PGDAttack, FGSMAttack


def RBSimCLR_trainer(model, train_loader, val_loader, optimizer, scheduler, criterion,
                     logger, train_batch_size, test_batch_size, max_epoch=10, n_steps_show=128, n_epoch_checkpoint=1,
                     device=DEVICE):

    attack_sample_list_train = [FGSMAttack(),
                                PGDAttack(batch_size=train_batch_size,
                                          loss_type="mse"),
                                PGDAttack(batch_size=train_batch_size,
                                          loss_type="sim"),
                                PGDAttack(batch_size=train_batch_size,
                                          loss_type="l1"),
                                PGDAttack(batch_size=train_batch_size, loss_type="cos")]
    attack_sample_list_test = [FGSMAttack(),
                               PGDAttack(batch_size=test_batch_size,
                                         loss_type="mse"),
                               PGDAttack(batch_size=test_batch_size,
                                         loss_type="sim"),
                               PGDAttack(batch_size=test_batch_size,
                                         loss_type="l1"),
                               PGDAttack(batch_size=test_batch_size, loss_type="cos")]
    print(f"Device: {device}")
    # TODO: (Xiaoyang) Enable checkpoint loading if necessary
    warmupscheduler = scheduler['warmupscheduler']
    mainscheduler = scheduler['mainscheduler']
    tri_criterion = criterion['tri_criterion']
    val_criterion = criterion['val_criterion']
    checkpoint(model, optimizer, mainscheduler, 0,
               logger, "RBSimCLR_epoch_{}_checkpoint.pt")

    # Basic stats
    current_epoch = 0
    for epoch in range(max_epoch):
        print(f"Epoch [{epoch}/{max_epoch}]\t")
        stime = time.time()

        # TRAINING phase
        model.train()
        tr_loss_epoch = []
        # TODO: Sample attack (for train & test) sample with dict

        rand_idx = np.random.randint(5)
        logger.log_attack_epoch(rand_idx)
        print("rand_idx is", rand_idx)
        attacker_train = attack_sample_list_train[rand_idx]
        attacker_test = attack_sample_list_test[rand_idx]
        # e.g. Attacker = Attack(type, metadata)
        for step, ((x_i, x_j, x), y) in enumerate(train_loader):
            optimizer.zero_grad()
            # Get augmented and attacked images
            x_i = x_i.squeeze().to(device).float()
            x_j = x_j.squeeze().to(device).float()
            x = x.squeeze().to(device).float()

            # get adversarial samples
            # todo: check whether it supports batch process
            x_adv = attacker_train.perturb(input_model=model,
                                           original_images=x,
                                           target=x,
                                           ).to(device)

            # Get latent representation
            h_i, h_j, h_adv, z_i, z_j, z_adv = model(x_i, x_j, x_adv)

            loss = tri_criterion(z_i, z_j, z_adv)
            loss.backward()
            optimizer.step()

            # Logging & Append training loss
            tr_loss_epoch.append(loss.item())
            logger.log_train_step(loss.item())

            # Show stats every n_steps_show updates
            if (step+1) % n_steps_show == 0:
                print(
                    f"Step [{step+1}/{len(train_loader)}]\t Loss: {round(loss.item(), 5)}")

        # Log learning rate
        lr = optimizer.param_groups[0]["lr"]
        logger.log_lr_epoch(lr)

        # SCHEDULER Update
        if epoch < 10:
            warmupscheduler.step()
        if epoch >= 10:
            mainscheduler.step()

        # EVALUATION Phase
        model.eval()
        with torch.no_grad():
            val_loss_epoch = []
            for step, ((x_i, x_j, x), y) in enumerate(val_loader):

                x_i = x_i.squeeze().to(device).float()
                x_j = x_j.squeeze().to(device).float()
                x = x.squeeze().to(device).float()
                # x_adv = Attacker(x)
                # x_adv = x_j.squeeze().to(device).float() # Test
                # TODO: Evaluation -->

                x_adv = attacker_test.perturb(input_model=model,
                                              original_images=x,
                                              target=x).to(device)

                # Get latent representation
                h_i, h_j, h_adv, z_i, z_j, z_adv = model(x_i, x_j, x_adv)

                loss = val_criterion(z_i, z_j, z_adv)

                # Logging & show validation statistics
                val_loss_epoch.append(loss.item())

        # Checkpointing
        if (epoch+1) % n_epoch_checkpoint == 0:
            checkpoint(model, optimizer, mainscheduler, current_epoch,
                       logger, "RBSimCLR_epoch_{}_checkpoint.pt")

        # Logging & Show epoch-level statistics
        print(
            f"Epoch [{epoch+1}/{max_epoch}]\t Training Loss: {np.mean(tr_loss_epoch)}\t lr: {round(lr, 5)}")
        print(
            f"Epoch [{epoch+1}/{max_epoch}]\t Validation Loss: {np.mean(val_loss_epoch)}\t lr: {round(lr, 5)}")
        current_epoch += 1

    # Running time statistics
    time_taken = (time.time()-stime)/60
    print(f"Epoch [{epoch}/{max_epoch}]\t Time Taken: {time_taken} minutes")


if __name__ == '__main__':
    ic("RBSimCLR trainer")
    # Get dataset
    dataset = ContrastiveLearningDataset("./datasets")
    num_views = 2
    print(DEVICE)
    train_dataset = dataset.get_dataset('cifar10_tri', num_views)
    val_dataset = dataset.get_dataset('cifar10_val', num_views)
    # Batch size config
    bsz_tri, bsz_val = 128, 64
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bsz_tri, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bsz_val, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)
    # Model and training config
    model = RBSimCLR(128).to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    warmupscheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda epoch: (epoch+1)/10.0, verbose=True)
    # SCHEDULER FOR COSINE DECAY
    mainscheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 500, eta_min=0.05, last_epoch=-1, verbose=True)
    scheduler = {
        'warmupscheduler': warmupscheduler,
        'mainscheduler': mainscheduler
    }
    # LOSS FUNCTION
    tri_criterion = RBSimCLRLoss(batch_size=bsz_tri, temperature=0.5)
    val_criterion = RBSimCLRLoss(batch_size=bsz_val, temperature=0.5)
    criterion = {
        'tri_criterion': tri_criterion,
        'val_criterion': val_criterion
    }
    # Logger
    logger = Logger()
    RBSimCLR_trainer(model, train_loader, val_loader,
                     optimizer, scheduler, criterion, logger, bsz_tri, bsz_val)
