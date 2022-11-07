import numpy as np
from torchvision.transforms import transforms
np.random.seed(0)


class ContrastiveLearningViewGenerator(object):
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform, n_views=2):
        self.base_transform = base_transform
        self.n_views = n_views
        self.raw_transform = transforms.Compose([transforms.RandomResizedCrop(size=224),
                                                 transforms.ToTensor()])

    def __call__(self, x):
        return [self.base_transform(x) for i in range(self.n_views)] + [self.raw_transform(x)]
