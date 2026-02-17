import numpy as np
import matplotlib.pyplot as plt
import os

from src.dataset.dataset import myDataset
from src.dataloader.dataloader import myDataloader

os.chdir(os.path.dirname(os.path.abspath(__file__)))
os.makedirs("output", exist_ok=True)

min_value = 0
max_value = 2 * np.pi
points_per_round = 100
num_rounds = 5
theta = np.linspace(min_value, max_value * num_rounds, points_per_round * num_rounds)
x = np.cos(theta)
y = np.sin(theta*2)

data = np.stack([x, y], axis=1)

plt.plot(data[:, 0], data[:, 1])

plt.savefig("./output/eight.png")

input_step = 2

input_data = data[:-1, :] # 最後から1つ前まで(0 ~ timestep-1)
target_data = data[input_step:, :] # input_stepから最後まで(input_step ~ timestep)

mydataset = myDataset(input_data, target_data, input_step)

split_ratio = [0.7, 0.2, 0.1]
batch_size = 50

mydataloader = myDataloader(mydataset, split_ratio, batch_size)
train_loader, validation_loader, test_loader = mydataloader.prepare_data()