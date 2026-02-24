import torchvision.transforms as transforms
import torch
from torch import nn
import numpy as np
import torch.distributions as td

# class ObsTransform:
#     def __init__(self):
#         self.obs_transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))
#         ])


class ObsTransform:
    def __init__(self):
        self.obs_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

    def __call__(self, img):
        return self.obs_transform(img)
    

def ELBO(prediction, target):
    x_hat, z_dist, _, _, _= prediction
    img_loss = nn.MSELoss()(x_hat, target) * np.prod(target.shape[-3:])
    prior = td.Normal(torch.zeros_like(z_dist.mean), torch.ones_like(z_dist.stddev))
    kld = td.kl_divergence(z_dist, prior)
    kld = kld.mean()

    loss = img_loss+kld
    return loss, img_loss, kld