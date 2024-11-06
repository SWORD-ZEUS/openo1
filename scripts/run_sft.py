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
import time

def main(config_path):
    # 加载配置
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    batch_size = config['batch_size_per_gpu']
    config['deepspeed_config']['train_micro_batch_size_per_gpu'] = batch_size

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
    config['model_path'] = model_path
    print(f"Model path: {model_path}")
    model = Generator(config, training=True)

    train_dataset = PRM800KDataset(config['train_data_path'], model_path, config['max_length'])
    val_dataset = PRM800KDataset(config['val_data_path'], model_path, config['max_length'])

    data_module = PRM800KDataModule(train_dataset, val_dataset, config['batch_size_per_gpu'])

    strategy = DeepSpeedStrategy(config=config['deepspeed_config'])

    # strategy = DeepSpeedStrategy(stage=3, offload_optimizer=True, offload_parameters=True)
    checkpoint_path = config.get('checkpoint_path', None)
    print(f"checkpoint_path: {checkpoint_path}")
    
    if checkpoint_path:
        print(f"Loading model from {checkpoint_path}")
        # 使用 load_from_checkpoint 方法加载检查点
        from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
        client_sd = get_fp32_state_dict_from_zero_checkpoint(checkpoint_path)
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
        )
        trainer.load_state_dict(client_sd, strict=True)
    else:
        trainer = SFTTrainer(model, train_dataset, val_dataset, config)

    # 创建 trainer
    pl_trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        devices=-1,
        accelerator="gpu",
        strategy=strategy,
        precision=16,
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

    # 开始训练
    pl_trainer.fit(trainer, datamodule=data_module)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='/zhuangkai/openo1/configs/sft_config.yaml')
    args = parser.parse_args()
    main(args.config)
