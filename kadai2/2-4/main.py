from box import Box
import os
import torch
import yaml
import wandb
from src.dataloader.dataloader import myDataloader
import torchvision
from src.utils.transform import ObsTransform,ELBO
from src.dataloader.dataloader import myDataloader

from src.trainer.trainer import Trainer
from src.model.vae import VAE
with open('./conf/config.yaml', 'r') as yml:
    cfg = Box(yaml.safe_load(yml))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"result/{cfg.wandb.train_name}", exist_ok=True)

# 修正1-9modelのdeviceを確認
device=torch.device("mps" if torch.mps.is_available()else "cpu")


wandb.init(project=cfg.wandb.project_name, config=cfg.wandb.config, name=cfg.wandb.train_name)

obs_transform = ObsTransform()
train_dataset = torchvision.datasets.MNIST(root='./data', train=True,
                                        download=True, transform=obs_transform)
val_dataset = torchvision.datasets.MNIST(root='./data', train=False,
                                        download=True, transform=obs_transform)


dataloader = myDataloader(cfg.train_data.batch_size)
train_dataloader = dataloader.prepare_data(dataset=train_dataset, shuffle=True)
val_dataloader = dataloader.prepare_data(dataset=val_dataset, shuffle=False)

model=VAE().to(device)


optimizer = torch.optim.Adam(
    model.parameters(),
    lr=cfg.optimizer.learning_rate,
    weight_decay=cfg.optimizer.weight_decay
)

trainer = Trainer(
    model=model,
    train_dataloader=train_dataloader,
    val_dataloader=val_dataloader,
    optimizer=optimizer,
    loss_fn=ELBO,
    epoch=cfg.train_data.epoch,
    save_path=f"result/{cfg.wandb.train_name}",
    device=device
)

trainer.train_model()

