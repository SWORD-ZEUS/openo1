from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

class PRM800KDataModule(LightningDataModule):
    def __init__(self, train_dataset, val_dataset, batch_size):
        super().__init__()
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.batch_size = batch_size

    def train_dataloader(self):
        collate_fn = getattr(self.train_dataset, 'collate_fn', None)
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=collate_fn
        )

    def val_dataloader(self):
        collate_fn = getattr(self.val_dataset, 'collate_fn', None)
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            collate_fn=collate_fn
        )
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=4)
