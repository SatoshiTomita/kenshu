import numpy as np
import torch
from torch.utils.data import Dataset

class myDataset(Dataset):
    def __init__(self, data, input_step):
        # data: [Total_Points, 2] の numpy配列を想定
        self.samples = []
        self.targets = []
        
        # 10ステップの窓をスライドさせながら、入力(10点)と正解(次の一点)を作る
        for i in range(len(data) - input_step):
            # iからi+10までが入力
            self.samples.append(data[i : i + input_step])
            # i+10がターゲット
            self.targets.append(data[i + input_step])
            
        self.samples = torch.tensor(np.array(self.samples), dtype=torch.float32)
        self.targets = torch.tensor(np.array(self.targets), dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        # ここで返る input_seq は [10, 2] の形状になります
        return self.samples[index], self.targets[index]