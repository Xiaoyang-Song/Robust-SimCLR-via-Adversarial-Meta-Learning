import torchvision
from models.identity import *
from icecream import ic


def get_resnet(name, pretrained=False):
    # TODO: (Xiaoyang) enable direct access to more versions of ResNet
    resnets = {
        "resnet18": torchvision.models.resnet18(pretrained=pretrained),
        "resnet50": torchvision.models.resnet50(pretrained=pretrained),
    }
    if name not in resnets.keys():
        raise KeyError(f"{name} is not a valid ResNet version")
    return resnets[name]


if __name__ == '__main__':
    ic("ResNet backbone")
    resnet = get_resnet('resnet18')
    resnet.fc = Identity()
    ic(resnet)
