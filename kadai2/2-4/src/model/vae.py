import torch
from torch import nn
from src.model.vision import Encoder,Decoder
import yaml
from box import Box
import torch.distributions as td
import torch.nn.functional as tf
with open('./conf/config.yaml', 'r') as yml:
    cfg = Box(yaml.safe_load(yml))
class VAE(nn.Module):
  def __init__(self):
    super().__init__()
    self.encoder=Encoder(
      cfg.model.encoder.channels,
      cfg.model.encoder.kernels,
      cfg.model.encoder.strides,
      cfg.model.encoder.paddings,
      cfg.model.latent_obs_dim,
      cfg.model.mlp.hidden_dim,
      cfg.model.mlp.layers,
    )
    self.decoder=Decoder(
      cfg.model.decoder.channels,
      cfg.model.decoder.kernels,
      cfg.model.decoder.strides,
      cfg.model.decoder.paddings,
      cfg.model.latent_obs_dim,
      cfg.model.mlp.hidden_dim,
      cfg.model.mlp.layers
    )
  
  def forward(self,x):
    mean_std=self.encoder(x)
    mean,std=torch.chunk(mean_std,2,dim=-1)
    std=tf.softplus(std,beta=0.5)
    z_dist=td.Normal(loc=mean,scale=std)
    z_sample=z_dist.rsample()
    x_hat=self.decoder(z_sample)

    return x_hat,z_dist,mean,std,z_sample

