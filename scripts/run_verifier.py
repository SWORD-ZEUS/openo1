import sys
import os
import yaml
import pytorch_lightning as pl
from pytorch_lightning.strategies import DeepSpeedStrategy
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger
from pytorch_lightning.loggers import WandbLogger

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.verifier.verifier import Verifier
from dataset.dataset_verifier import VerifierModelDataset
from dataset.data_module import PRM800KDataModule
from training.verifier.verifier_trainer import VerifierTrainer
import time

def main(args):
    # 加载配置
    with open(args.config, 'r') as file:
        config = yaml.safe_load(file)
    
    batch_size = config['batch_size_per_gpu']
    config['deepspeed_config']['train_micro_batch_size_per_gpu'] = batch_size
    num_labels = config['fine_tuning']['num_labels']
    if num_labels == 1:
        task = "regression"
    else:
        task = "classification"

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
                save_dir=config['wandb']['save_dir'],
                name=config['wandb']['project'],
                version=config['wandb']['name']+f"-{time.strftime('%m%d%H%M%S')}",
            )
            print("Initialize TensorBoard Logger successfully")

    model_path = os.path.join(config['download_model_dir'], config['model_name'])
    config['model_path'] = model_path
    print(f"Model path: {model_path}")
    model = Verifier(config, training=config["fine_tuning"]["training_mode"])

    train_dataset = VerifierModelDataset(config['train_data_path'], model_path, config['max_length'], task)
    val_dataset = VerifierModelDataset(config['val_data_path'], model_path, config['max_length'], task)

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
        trainer = VerifierTrainer(
            model=model,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            config=config,
        )
        trainer.load_state_dict(client_sd, strict=True)
    else:
        trainer = VerifierTrainer(model, train_dataset, val_dataset, config)

    pl_trainer = pl.Trainer(
        max_epochs=config['num_epochs'],
        devices=-1,  # 使用所有可用的 GPU
        accelerator="gpu",
        strategy=strategy,
        precision=16,  # Use mixed precision
        accumulate_grad_batches=config['gradient_accumulation_steps'],
        val_check_interval=config['val_interval'],
        logger=logger,
        log_every_n_steps=config['log_every_n_steps'],
        callbacks=[pl.callbacks.ModelCheckpoint(
            dirpath=os.path.join(config['weight_save_dir'], time.strftime('%m%d%H%M%S')), 
            save_top_k=1, 
            monitor="val_loss",
            save_last=True,
            )]
    )

    pl_trainer.fit(trainer, datamodule=data_module)

if __name__ == "__main__":
    import argparse
    args = argparse.ArgumentParser()
    args.add_argument("--config", type=str, default="/zhuangkai/openo1/configs/verifier_config.yaml")
    args = args.parse_args()
    main(args)
