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
                                                                      32),
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


if __name__ == '__main__':
    ic("Data Augmentation & Visualization")
    dataset = ContrastiveLearningDataset("./datasets")
    train_dataset = dataset.get_dataset('cifar10', 2)
    ic(len(train_dataset))
    ic(len(train_dataset[0][0]))
    # Visualize example images
    img_view = torch.stack(train_dataset[0][0])
    ic(img_view.shape)
    show_images(img_view.data.cpu())
