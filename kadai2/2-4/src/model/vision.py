import torch
from torch import nn
import numpy as np
def get_conved_size(
    obs_shape=None,
    channels=None, 
    kernels=None, 
    strides=None, 
    paddings=None, 
):
    conved_shape = obs_shape
    for i in range(len(channels)):
        conved_shape = conv_out_shape(
            conved_shape, paddings[i], kernels[i], strides[i]
        )
    conved_size = channels[-1] * np.prod(conved_shape).item()
    return conved_size
    
def conv_out_shape(h_in, padding, kernel_size, stride):
    return tuple(conv_out(x, padding, kernel_size, stride) for x in h_in)

def conv_out(h_in, padding, kernel_size, stride):
    return int((h_in + 2.0 * padding - (kernel_size - 1.0) - 1.0) / stride + 1.0)

def _build_mlp(in_dim: int, out_dim: int, hidden_dim: int, n_layers: int):
    layers = []
    layers.append(nn.Flatten())

    if n_layers == 0:
        layers.append(nn.Linear(in_dim, out_dim))
        layers.append(nn.ReLU())
    else:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(nn.ReLU())
        for _ in range(n_layers-1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(hidden_dim, out_dim))
    return nn.Sequential(*layers)
class Encoder(nn.Module):
    def __init__(
        self,
        channels,
        kernels,
        strides,
        paddings,
        latent_obs_dim,
        mlp_hidden_dim,
        n_mlp_layers):
        super().__init__()
        
        obs_size = [28, 28]
        
        self.encoder = self._build_conv_layers(
            channels, kernels, strides, paddings
        )
        self.fc = _build_mlp(
            get_conved_size(
                obs_size,
                channels,
                kernels,
                strides,
                paddings
            ),
            latent_obs_dim*2,
            mlp_hidden_dim,
            n_mlp_layers
        )

    def forward(self, x):
        x = self.encoder(x)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)
        
        return x


    def _build_conv_layers(
        self, 
        channels: tuple,
        kernels: tuple,
        strides: tuple,
        paddings: tuple,
        ):
        layers = []
        for i in range(len(channels)):
            layers.append(
                nn.Conv2d(
                    in_channels=channels[i-1] if i > 0 else 1,
                    out_channels=channels[i],
                    kernel_size=kernels[i],
                    stride=strides[i],
                    padding=paddings[i],
                )
            )
            layers.append(nn.ReLU())
        return nn.Sequential(*layers)