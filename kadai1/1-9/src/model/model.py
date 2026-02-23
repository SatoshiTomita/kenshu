from torch import nn

class FNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, input):
        output = self.net(input)
        return output

def _build_convtranspose_layers(
self, 
channels: tuple,
kernels: tuple,
strides: tuple,
paddings: tuple,
latent_obs_dim: int,
):
  layers = []
  for i in range(len(channels)-1):
      layers.append(
          nn.ConvTranspose2d(
              in_channels=channels[i],
              out_channels=channels[i+1],
              kernel_size=kernels[i],
              stride=strides[i],
              padding=paddings[i],
          )
      )
      if i < len(channels) - 1:
        layers.append(nn.ReLU())
      else:
        layers.append(nn.Sigmoid()) # 出力層の活性化関数は要注意
  return nn.Sequential(*layers)

class Decoder(nn.Module):
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
    
        self.obs_size = [28, 28]
    
        self.decoder = self._build_convtranspose_layers(
            channels, kernels, strides, paddings, latent_obs_dim
        )
        
        self.conved_size = get_conved_size(
                self.obs_size,
                channels,
                kernels,
                strides,
                paddings
            )
        
        self.in_channel = channels[0]
        self.conved_image_size = 7
        
        self.fc = _build_mlp(
            in_dim=latent_obs_dim,
            out_dim=self.in_channel * self.conved_image_size * self.conved_image_size,
            hidden_dim=mlp_hidden_dim,
            n_layers=n_mlp_layers
        )
    
    def forward(self, x):
        x = self.fc(x)
        x = x.reshape([-1, self.in_channel, self.conved_image_size, self.conved_image_size])
        x = self.decoder(x)
        
        return x
    
def _build_convtranspose_layers(
self, 
channels: tuple,
kernels: tuple,
strides: tuple,
paddings: tuple,
latent_obs_dim: int,
):
  layers = []
  for i in range(len(channels)-1):
      layers.append(
          nn.ConvTranspose2d(
              in_channels=channels[i],
              out_channels=channels[i+1],
              kernel_size=kernels[i],
              stride=strides[i],
              padding=paddings[i],
          )
      )
      if i < len(channels) - 1:
        layers.append(nn.ReLU())
      else:
        layers.append(nn.Sigmoid())
  return nn.Sequential(*layers)