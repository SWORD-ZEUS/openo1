import sys
import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.reward_model.reward_model import RewardModel
from dataset.dataset_rm import RewardModelDataset
from dataset.data_module import PRM800KDataModule
from training.rm.rm_trainer import RMTrainer
import time

def main():
    # 加载配置
    with open("/zhuangkai/openo1/configs/rm_config.yaml", 'r') as file:
        config = yaml.safe_load(file)
    batch_size = config['batch_size_per_gpu']
    config['deepspeed_config']['train_micro_batch_size_per_gpu'] = batch_size
    only_train_head = config['only_train_head']
    num_labels = config['num_labels']

    # 设置 Wandb
    if os.environ["NODE_RANK"] == "0":
        if config['wandb']['use_wandb']:
            os.environ["WANDB_API_KEY"] = config['wandb']['api_key']
            logger = WandbLogger(project=config['wandb']['project'], 
                                name=config['wandb']['name']+f"-{time.strftime('%m%d%H%M%S')}",
                                tags=config['wandb']['tags'],
                                notes=config['wandb']['notes'],
                                group=config['wandb']['group'],
                                job_type=config['wandb']['job_type'],
                                save_dir=config['wandb']['save_dir'],
            )
            print("Initialize Wandb Logger successfully")
        else:
            logger = TensorBoardLogger(
                save_dir=config['wandb']['dir'],
                name=config['wandb']['project'],
                version=config['wandb']['name']+f"-{time.strftime('%m%d%H%M%S')}",
            )
            print("Initialize TensorBoard Logger successfully")

    model_path = os.path.join(config['download_model_dir'], config['model_name'])
    print(f"Model path: {model_path}")
    model = RewardModel(model_path, only_train_head, num_labels, training=True)

    train_dataset = RewardModelDataset(config['train_data_path'], model_path, config['max_length'])
    val_dataset = RewardModelDataset(config['val_data_path'], model_path, config['max_length'])

    data_module = PRM800KDataModule(train_dataset, val_dataset, config['batch_size_per_gpu'])

    
    trainer = RMTrainer(model, train_dataset, val_dataset, config)
    strategy = DeepSpeedStrategy(config=config['deepspeed_config'])

    # strategy = DeepSpeedStrategy(stage=3, offload_optimizer=True, offload_parameters=True)
    checkpoint_path = getattr(config, 'checkpoint_path', None)
    if checkpoint_path:
        print(f"Loading model from {checkpoint_path}")
        pl_trainer = pl.Trainer(
            resume_from_checkpoint=checkpoint_path,
            max_epochs=config['num_epochs'],
            devices=-1,
            accelerator="gpu",
            strategy=strategy,
            precision=16,
            accumulate_grad_batches=config['gradient_accumulation_steps'],
            val_check_interval=config['val_interval'],
            logger=logger,
            callbacks=[pl.callbacks.ModelCheckpoint(
                dirpath=config['weight_save_dir'], 
                save_top_k=1, 
                monitor="val_loss",
                save_last=True,
            )]
        )
    else:
        pl_trainer = pl.Trainer(
            max_epochs=config['num_epochs'],
            devices=-1,  # 使用所有可用的 GPU
            accelerator="gpu",
            strategy=strategy,
            precision=16,  # Use mixed precision
            accumulate_grad_batches=config['gradient_accumulation_steps'],
            val_check_interval=config['val_interval'],
            logger=logger,
            callbacks=[pl.callbacks.ModelCheckpoint(
                dirpath=os.path.join(config['weight_save_dir'], time.strftime('%m%d%H%M%S')), 
                save_top_k=1, 
                monitor="val_loss",
                save_last=True,
                )]
        )

    pl_trainer.fit(trainer, datamodule=data_module)

if __name__ == "__main__":
    main()
