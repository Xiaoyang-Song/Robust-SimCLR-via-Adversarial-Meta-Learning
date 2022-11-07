import torch
import torchvision
from torchvision.transforms import transforms
from torchvision import transforms, datasets
from icecream import ic
from torchvision.transforms import transforms
from data.blur import GaussianBlur
from torchvision import transforms, datasets
from data.view_generator import ContrastiveLearningViewGenerator
from exceptions.exception import InvalidDatasetSelection
# Customization
from utils import *
from PIL import Image


class ContrastiveLearningDataset:
    def __init__(self, root_folder):
        self.root_folder = root_folder

    @staticmethod
    def get_simclr_pipeline_transform(size, s=1):
        """Return a set of data augmentation transformations as described in the SimCLR paper."""
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        data_transforms = transforms.Compose([transforms.RandomResizedCrop(size=size),
                                              transforms.RandomHorizontalFlip(),
                                              transforms.RandomApply(
                                                  [color_jitter], p=0.8),
                                              transforms.RandomGrayscale(
                                                  p=0.2),
                                              GaussianBlur(
                                                  kernel_size=int(0.1 * size)),
                                              transforms.ToTensor()])
        return data_transforms

    def get_dataset(self, name, n_views):
        valid_datasets = {'cifar10': lambda: datasets.CIFAR10(self.root_folder, train=True,
                                                              transform=ContrastiveLearningViewGenerator(
                                                                  self.get_simclr_pipeline_transform(
                                                                      224),
                                                                  n_views),
                                                              download=True),

                          'stl10': lambda: datasets.STL10(self.root_folder, split='unlabeled',
                                                          transform=ContrastiveLearningViewGenerator(
                                                              self.get_simclr_pipeline_transform(
                                                                  96),
                                                              n_views),
                                                          download=True)}

        try:
            dataset_fn = valid_datasets[name]
        except KeyError:
            raise InvalidDatasetSelection()
        else:
            return dataset_fn()


def CIFAR10(batch_size=128, test_batch_size=128):
    transform = transforms.Compose(
        [transforms.ToTensor()])
    train_dataset = datasets.CIFAR10('./datasets/CIFAR-10', train=True,
                                     download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = datasets.CIFAR10('./datasets/CIFAR-10', train=False, download=True,
                                   transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=True)

    return train_dataset, val_dataset, train_loader, val_loader


if __name__ == '__main__':
    ic("Data Augmentation & Visualization")
    dataset = ContrastiveLearningDataset("./datasets")
    num_views = 3
    train_dataset = dataset.get_dataset('cifar10', num_views)
    ic(len(train_dataset))
    ic(len(train_dataset[0][0]))
    # Visualize example images
    img_view = torch.stack(train_dataset[0][0])
    ic(img_view.shape)
    ic(img_view[0].shape)
    for idx in range(num_views):
        Image.fromarray(np.array(img_view[idx].reshape(224, 224, 3)), 'RGB').save(
            f"data/example/view{idx}.png")
    # View original images
    cifar_tri, _, _, _ = CIFAR10()
    plt.imshow(np.array(cifar_tri[0][0].permute(1, 2, 0)))
    # plt.show()

    # Check dataloader shape
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=32, shuffle=True,
        num_workers=2, pin_memory=True, drop_last=True)
    tri_batch = next(iter(train_loader))
    ic(len(tri_batch))
    ic(len(tri_batch[0]))
    ic(len(tri_batch[1]))
    cat_img = torch.cat(tri_batch[0], dim=0)
    ic(cat_img.shape)
