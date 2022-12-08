from trainer.rbsimclr_trainer import *
from tqdm import tqdm


class LEVAL(nn.Module):
    def __init__(self):
        self.fc = nn.Linear(256, 256)
        self.relu = nn.ReLU()
        self.out = nn.Linear(256, 10)

    def forward(self, x):
        return self.out(self.relu(self.fc(x)))


def clf_trainer(model, cifar_tri_loader, cifar_val_loader, le_writer):
    num_epoch = 10
    iter_count_train = 0
    iter_count_val = 0

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1e-3, betas=(0.5, 0.999))
    criterion = nn.CrossEntropyLoss()

    for epoch in tqdm(range(num_epoch)):
        # Training
        model.train()
        train_loss, train_acc = [], []
        for idx, (img, label) in enumerate(cifar_tri_loader):
            img = img.to(DEVICE)
            label = label.to(DEVICE)
            optimizer.zero_grad()
            logits = model(img)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            # Append training statistics
            acc = (torch.argmax(logits, dim=1) ==
                   label).sum().item() / label.shape[0]
            train_acc.append(acc)
            train_loss.append(loss.detach().item())
            le_writer.add_scalar("Training/Accuracy", acc, iter_count_train)
            le_writer.add_scalar(
                "Training/Loss", loss.detach().item(), iter_count_train)
            iter_count_train += 1

        le_writer.add_scalar("Training/Accuracy (Epoch)",
                             np.mean(train_acc), epoch)
        le_writer.add_scalar("Training/Loss (Epoch)",
                             np.mean(train_loss), epoch)
        print(f"Epoch  # {epoch + 1} | training loss: {np.mean(train_loss)} \
                | training acc: {np.mean(train_acc)}")
        # Evaluation
        model.eval()
        with torch.no_grad():
            val_loss, val_acc = [], []
            for idx, (img, label) in enumerate(cifar_val_loader):
                img, label = img.to(DEVICE), label.to(DEVICE)
                logits = model(img)
                loss = criterion(logits, label)
                acc = (torch.argmax(logits, dim=1) ==
                       label).sum().item() / label.shape[0]
                val_acc.append(acc)
                val_loss.append(loss.detach().item())
                le_writer.add_scalar("Training/Accuracy", acc, iter_count_val)
                le_writer.add_scalar(
                    "Training/Loss", loss.detach().item(), iter_count_val)
                iter_count_val += 1

            le_writer.add_scalar(
                "Training/Accuracy (Epoch)", np.mean(val_acc), epoch)
            le_writer.add_scalar("Training/Loss (Epoch)",
                                 np.mean(val_loss), epoch)
            print(f"Epoch  # {epoch + 1} | validation loss: {np.mean(val_loss)} \
                | validation acc: {np.mean(val_acc)}")
