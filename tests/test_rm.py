import sys
import os
import yaml
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.reward_model.reward_model import RewardModel
from dataset.dataset_rm import RewardModelDataset

class RewardModelTester(pl.LightningModule):
    def __init__(self, model,config):
        super().__init__()
        self.model = model
        self.test_predictions = []
        self.test_labels = []
        self.config = config

    def test_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].long()  # 确保输入数据为 Float 类型
        attention_mask = batch['attention_mask'].float()  # 确保输入数据为 Float 类型
        labels = batch['labels'].float() # 确保标签为 Float 类型

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        predictions = outputs.logits.squeeze()

        self.test_predictions.extend(predictions.cpu().numpy())
        self.test_labels.extend(labels.cpu().numpy())

        self.log('test_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def on_test_epoch_end(self):
        from sklearn.metrics import mean_squared_error, r2_score
        mse = mean_squared_error(self.test_labels, self.test_predictions)
        r2 = r2_score(self.test_labels, self.test_predictions)

        self.log('test_mse', mse)
        self.log('test_r2', r2)

        print(f"Mean Squared Error: {mse}")
        print(f"R2 Score: {r2}")

        results = {
            "mse": mse,
            "r2": r2,
            "predictions": self.test_predictions,
            "labels": self.test_labels
        }

        import json
        with open(self.config['test_settings']['test_results_path'], "w") as f:
            json.dump(results, f)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Test Reward Model")
    parser.add_argument("--config", type=str, default="/zhuangkai/openo1/configs/rm_config.yaml", help="Path to config file")
    parser.add_argument("--load_lora", action="store_true", help="Whether to load LoRA weights")
    args = parser.parse_args()

    config = load_config(args.config)

    # 加载模型
    model_path = os.path.join(config['download_model_dir'], config['model_name'])
    model = RewardModel(model_path, only_train_head=not args.load_lora, num_labels=config['num_labels'], training=False)
    model = model.float()

    # 加载LoRA权重（如果指定）
    if args.load_lora:
        lora_weights_path = config['test_settings']['load_lora_weights_path']
        print(f"Loading LoRA weights from {lora_weights_path}")
        client_sd = get_fp32_state_dict_from_zero_checkpoint(lora_weights_path)
        model.load_state_dict(client_sd, strict=False)

    # 创建 PyTorch Lightning 模块
    rm_tester = RewardModelTester(model,config)

    # 加载测试数据集
    test_dataset = RewardModelDataset(config['test_data_path'], model_path, config['max_length'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size_per_gpu'], num_workers=4)

    # 设置 logger
    log_dir = config['test_settings']['log_dir']
    logger = TensorBoardLogger(log_dir, name="reward_model_test")

    # 创建 Trainer
    trainer = pl.Trainer(
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
    )

    # 运行测试
    trainer.test(rm_tester, dataloaders=test_dataloader)

if __name__ == "__main__":
    main()
