import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import os
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam

class SFTTrainer(pl.LightningModule):
    def __init__(self, model, train_dataset, val_dataset, config):
        super().__init__()
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.save_hyperparameters(config)

    def training_step(self, batch, batch_idx):
        self.model.train()
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = DeepSpeedCPUAdam(self.model.parameters(), 
                                     lr=float(self.config['learning_rate']),
                                     betas=(0.9, 0.999),
                                     eps=1e-8,
                                     weight_decay=0)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=self.config['total_steps']
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }