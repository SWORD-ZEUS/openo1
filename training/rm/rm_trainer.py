import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
import os
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from deepspeed.ops.adam import DeepSpeedCPUAdam

class RMTrainer(pl.LightningModule):
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
        
        # 检查loss是否为NaN
        if torch.isnan(loss):
            # 获取当前batch的一些信息用于调试
            batch_info = {
                'input_shape': batch['input_ids'].shape,
                'labels': batch['labels'],
                'step_start_idx': batch['step_start_idx'],
                'step_end_idx': batch['step_end_idx'],
            }
            error_msg = (
                f"训练过程中出现NaN损失值!\n"
                f"Batch信息: {batch_info}\n"
            )
            raise ValueError(error_msg)
        
        # 记录各个损失组件
        self.log('train_total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('train_cls_loss', outputs.logits['cls_loss'], on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_ranking_loss', outputs.logits['ranking_loss'], on_step=True, on_epoch=True, sync_dist=True)
        
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        outputs = self.model(**batch)
        loss = outputs.loss
        
        # 检查loss是否为NaN
        if torch.isnan(loss):
            batch_info = {
                'batch_idx': batch_idx,
                'input_shape': batch['input_ids'].shape,
                'labels': batch['labels'],
            }
            error_msg = (
                f"验证过程中出现NaN损失值!\n"
                f"Batch信息: {batch_info}"
            )
            raise ValueError(error_msg)
        
        # 记录各个损失组件
        self.log('val_total_loss', loss, on_epoch=True, prog_bar=True, sync_dist=True)
        self.log('val_cls_loss', outputs.logits['cls_loss'], on_epoch=True, sync_dist=True)
        self.log('val_ranking_loss', outputs.logits['ranking_loss'], on_epoch=True, sync_dist=True)

        return loss

    def test_step(self, batch, batch_idx):
        self.model.eval()
        outputs = self.model(**batch)
        loss = outputs.loss
        self.log(
            'test_loss', 
            loss, 
            on_epoch=True, 
            prog_bar=True, 
            sync_dist=True
        )
        
        return loss

    def configure_optimizers(self):
        # 计算总的训练步数
        total_samples = len(self.train_dataset)  # 训练集的样本数量
        total_steps = self.config['num_epochs'] * (total_samples // (self.config['batch_size_per_gpu']*self.config['gpus_per_node']))

        optimizer = AdamW(self.model.parameters(), 
                          lr=float(self.config['learning_rate']),
                          betas=(0.9, 0.999),
                          eps=1e-8,
                          weight_decay=0)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config['warmup_steps'],
            num_training_steps=total_steps  # 使用计算得出的total_steps
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step',
                'frequency': 1
            }
        }
