from __future__ import annotations

from torch.utils.data import DataLoader, Dataset


class myDataloader:
    def __init__(self, batch_size: int, num_workers: int = 0, pin_memory: bool = False):
        self.batch_size = int(batch_size)
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)

    def prepare_data(self, dataset: Dataset, shuffle: bool) -> DataLoader:
        # Datasetからミニバッチを作る
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )
