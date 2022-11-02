import math
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib import markers
import torch
import numpy as np

# Helper for visualizing images with augmentation


# def show_images(images):
#     images = torch.reshape(
#         images, [images.shape[0], -1]
#     )  # images reshape to (batch_size, D)
#     sqrtn = int(math.ceil(math.sqrt(images.shape[0])))
#     sqrtimg = int(math.ceil(math.sqrt(images.shape[1])))

#     fig = plt.figure(figsize=(sqrtn, sqrtn))
#     gs = gridspec.GridSpec(sqrtn, sqrtn)
#     gs.update(wspace=0.05, hspace=0.05)

#     for i, img in enumerate(images):
#         ax = plt.subplot(gs[i])
#         plt.axis("off")
#         ax.set_xticklabels([])
#         ax.set_yticklabels([])
#         ax.set_aspect("equal")
#         plt.imshow(img.reshape([sqrtimg, sqrtimg]))
#     return
