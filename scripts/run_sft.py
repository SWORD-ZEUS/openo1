import sys
import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator.generator import Generator
from dataset.dataset import PRM800KDataset
from dataset.data_module import PRM800KDataModule
from training.sft.sft_trainer import SFTTrainer
from configs.sft_config import SFTConfig
import time

def main():
    # 加载配置
    with open("/zhuangkai/openo1/configs/sft_config.yaml", 'r') as file:
        config = yaml.safe_load(file)

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
                                dir=config['wandb']['dir'],
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
    model = Generator(model_path, training=True)

    train_dataset = PRM800KDataset(config['train_data_path'], model_path, config['max_length'])
    val_dataset = PRM800KDataset(config['test_data_path'], model_path, config['max_length'])

    data_module = PRM800KDataModule(train_dataset, val_dataset, config['batch_size'])

    trainer = SFTTrainer(model, train_dataset, val_dataset, config)

    strategy = DeepSpeedStrategy(stage=3, offload_optimizer=True, offload_parameters=True)

    pl_trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        devices=-1,  # 使用所有可用的 GPU
        accelerator="gpu",
        strategy=strategy,
        precision=16,  # Use mixed precision
        accumulate_grad_batches=config['gradient_accumulation_steps'],
        val_check_interval=config['val_interval'],
        logger=logger,
        callbacks=[pl.callbacks.ModelCheckpoint(dirpath=config['weight_save_dir'], save_top_k=3, monitor="val_loss")]
    )

    pl_trainer.fit(trainer, datamodule=data_module)

if __name__ == "__main__":
    main()