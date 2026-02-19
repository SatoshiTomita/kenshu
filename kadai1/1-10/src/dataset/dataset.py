import numpy as np
import torch
from torch.utils.data import Dataset

class myDataset(Dataset):
    def __init__(self, 
            data1: np.ndarray, 
            data2: np.ndarray,
            input_step: int):

        self.data2 = torch.from_numpy(data2).float()
        self.data1 = torch.from_numpy(data1).float()
        self.step = input_step
    
    def __len__(self):
        return len(self.data2)
    
    def __getitem__(self, index):
        # index番目の「1周分（100ステップ）のデータ」を丸ごと返す
        # RNNは [Seq, Dim] の形を期待するので view(-1) は不要です
        input_seq = self.data1[index]  # shape: [100, 2]
        target_seq = self.data2[index] # shape: [100, 2]
        
        return input_seq, target_seq