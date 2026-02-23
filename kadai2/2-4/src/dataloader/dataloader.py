from torch.utils.data import Dataset, DataLoader

class myDataloader():
    def __init__(self, 
                batch_size: int):
        self.batch_size = batch_size
		
    def prepare_data(self, dataset: Dataset, shuffle: bool):
        dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                                shuffle=shuffle, num_workers=0)
        return dataloader
