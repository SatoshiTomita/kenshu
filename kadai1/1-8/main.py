from box import Box
import numpy as np
import matplotlib.pyplot as plt
import os
import torch
from torch import nn
import yaml
import wandb
import random
from src.dataset.dataset import myDataset
from src.dataloader.dataloader import myDataloader
from src.model.model import FNN
from src.trainer.trainer import Trainer
from src.utils.predict import Predict
seed = 1
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.manual_seed(seed)
with open('./conf/config.yaml', 'r') as yml:
    cfg = Box(yaml.safe_load(yml))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(f"result/{cfg.wandb.train_name}", exist_ok=True)



wandb.init(project=cfg.wandb.project_name, config=cfg.wandb.config, name=cfg.wandb.train_name)

min_value = cfg.gen_data.min
max_value = cfg.gen_data.max * np.pi
points_per_round = cfg.gen_data.points_per_round
num_rounds = cfg.gen_data.num_rounds
theta = np.linspace(min_value, max_value * num_rounds, points_per_round * num_rounds)

x = np.cos(theta)
y = np.sin(theta*2)

data = np.stack([x, y], axis=1)

plt.plot(data[:, 0], data[:, 1])

plt.savefig("./output/eight.png")

input_step = cfg.train_data.input_timestep

input_data = data[:-1, :]
# 最初の学習を始めるためのヒントを切り出す
# 最初からinputstepまで
target_data = data[input_step:, :]

mydataset = myDataset(input_data, target_data, input_step)

mydataloader = myDataloader(mydataset, cfg.train_data.split_ratio, cfg.train_data.batch_size)
train_loader, validation_loader, test_loader = mydataloader.prepare_data()

model = FNN(input_dim=input_step*2, hidden_dim=cfg.model.hidden_dim, output_dim=2)
loss_fn = nn.MSELoss()

optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.optimizer.learning_rate, weight_decay=cfg.optimizer.weight_decay)

trainer = Trainer(model, train_loader, validation_loader, optimizer, loss_fn, cfg.train_data.epoch,"output")

trainer.train_model()

wandb.finish()

predict=Predict(cfg,model)

test_loss=predict.test_loss(test_loader,loss_fn)
close_test=predict.predict()

fig=plt.figure(figsize=(12,8))
ax=fig.add_subplot(111)

ax.plot(close_test[:,0],close_test[:,1])
fig.savefig(f"result/{cfg.wandb.train_name}/cloes_test.pdf")