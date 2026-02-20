from torch.utils.data import Dataset, DataLoader, random_split

class myDataloader():
    def __init__(self, 
                mydataset: Dataset, 
                ratio: list,
                batch_size: int):
        self.mydataset = mydataset
        self.ratio = ratio
        self.batch_size = batch_size
		
    def prepare_data(self):
        # データを訓練,検証,テストに分割
        train_data, validation_data, test_data = random_split(self.mydataset, lengths=self.ratio)
        # 作成したDatasetをモデルに送る
        # 読み込みを行うDatasetを渡す
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True, num_workers=0)
        validation_loader = DataLoader(validation_data, batch_size=self.batch_size)
        test_loader = DataLoader(test_data, batch_size=self.batch_size)
        
        return train_loader, validation_loader, test_loader