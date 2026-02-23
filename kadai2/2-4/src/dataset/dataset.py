import numpy as np
import torch
from torch.utils.data import Dataset

class myDataset(Dataset):
    def __init__(self, 
            data1: np.ndarray, 
            data2: np.ndarray,
            input_step: int):

        self.data1 = torch.from_numpy(data1).float()
        self.data2 = torch.from_numpy(data2).float()
        self.step = input_step
    
    def __len__(self):
        return len(self.data2)

    def __getitem__(self, index):
        return self.data1[index:index+self.step].view(-1), self.data2[index]