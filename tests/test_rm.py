#运行：torchrun --nproc_per_node={gpus_per_node} tests/test_rm.py 两个参数：--only_train_head 和 --load_trained_weights
import sys
import os
import yaml
import torch
import argparse
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.strategies import DeepSpeedStrategy
from torch.utils.data import DataLoader
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint
# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.reward_model.reward_model import RewardModel
from dataset.dataset_rm import RewardModelDataset
from utils.process_state_dict import process_state_dict
import time

class RewardModelTester(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.model.eval()
        self.task = "classification" if config.get('num_labels', 1) > 1 else "regression"
   

    def forward(self, input_ids, attention_mask, labels=None):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def predict_step(self, batch, batch_idx):
        input_ids = batch['input_ids'].long()
        attention_mask = batch['attention_mask'].float()
        labels = batch['labels'].float()

        outputs = self(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        logits = outputs.logits
        
        if self.task == "classification":
            # 对分类任务使用 softmax
            predictions = torch.softmax(logits, dim=-1)
        else:
            # 回归任务保持不变
            predictions = logits.squeeze()

        # 确保 predictions 是至少 1 维的
        if predictions.ndim == 0:
            predictions = predictions.unsqueeze(0)

        return {
            'predictions': predictions,
            'labels': labels
        }

    def on_predict_epoch_end(self):
        from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, f1_score, precision_score, recall_score
        
        # 从 trainer 中获取预测结果
        predictions = self.trainer.predict_loop.predictions
        
        # 收集所有预测结果
        all_predictions = []
        all_labels = []
        
        # 直接处理预测结果
        for batch_result in predictions:
            if isinstance(batch_result, dict) and 'predictions' in batch_result and 'labels' in batch_result:
                # 将 float16 转换为 float32
                predictions_tensor = batch_result['predictions'].float()
                labels_tensor = batch_result['labels'].float()
                
                all_predictions.extend(predictions_tensor.cpu())
                all_labels.extend(labels_tensor.cpu())
        
        if not all_predictions or not all_labels:
            print("Warning: No predictions or labels collected")
            return
        
        # 转换为张量
        all_predictions = torch.stack(all_predictions)
        all_labels = torch.stack(all_labels)
        
        # 根据任务类型计算不同的指标
        results = {}
        
        if self.task == "classification":
            # 对于分类任务，将预测转换为类别
            pred_classes = torch.argmax(all_predictions, dim=-1).numpy()
            true_labels = all_labels.numpy()
            
            # 计算分类指标
            accuracy = accuracy_score(true_labels, pred_classes)
            f1 = f1_score(true_labels, pred_classes, average='weighted')
            precision = precision_score(true_labels, pred_classes, average='weighted')
            recall = recall_score(true_labels, pred_classes, average='weighted')
            
            print(f"Accuracy: {accuracy}")
            print(f"F1 Score: {f1}")
            print(f"Precision: {precision}")
            print(f"Recall: {recall}")
            
            results = {
                "accuracy": float(accuracy),
                "f1": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "predictions": pred_classes.tolist(),
                "labels": true_labels.tolist()
            }
        else:
            # 对于回归任务，保持原有的指标计算
            mse = mean_squared_error(all_labels.numpy(), all_predictions.numpy())
            r2 = r2_score(all_labels.numpy(), all_predictions.numpy())
            
            print(f"Mean Squared Error: {mse}")
            print(f"R2 Score: {r2}")
            
            results = {
                "mse": float(mse),
                "r2": float(r2),
                "predictions": all_predictions.tolist(),
                "labels": all_labels.tolist()
            }

        # 只在主进程上保存结果
        if self.trainer.is_global_zero:
            import json
            path = os.path.join(self.config['test_settings']['test_results_dir'], 
                               f"predict_results-{time.strftime('%m%d%H%M%S')}.json")
            with open(path, "w") as f:
                json.dump(results, f)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main():
    parser = argparse.ArgumentParser(description="Test Reward Model")
    parser.add_argument("--config", type=str, default="/zhuangkai/openo1/configs/rm_config.yaml", help="Path to config file")
    parser.add_argument("--load_trained_weights", action="store_true", help="Whether to load trained weights")
    parser.add_argument("--only_train_head", action="store_true", help="Only train the regression head")
    args = parser.parse_args()

    config = load_config(args.config)

    # 加载模型
    model_path = os.path.join(config['download_model_dir'], config['model_name'])
    config['model_path'] = model_path
    model = RewardModel(config, training=False)
    model = model.float()

    # 加载训练权重（如果指定）
    if args.load_trained_weights:
        trained_weights_path = config['test_settings']['load_trained_weights_path']
        print(f"Loading trained weights from {trained_weights_path}")
        client_sd = get_fp32_state_dict_from_zero_checkpoint(trained_weights_path)
        processed_sd = process_state_dict(client_sd)
        # 尝试加载处理后的权重
        try:
            model.load_state_dict(processed_sd, strict=True)
            print("\n成功加载处理后的权重")
        except Exception as e:
            print(f"\n加载权重时出错: {str(e)}")
            raise e

    # 创建 PyTorch Lightning 模块
    rm_tester = RewardModelTester(model, config)

    # 加载测试数据集
    task = rm_tester.task
    test_dataset = RewardModelDataset(config['test_data_path'], model_path, config['max_length'], task)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size_per_gpu'], num_workers=4)

    # 设置 logger
    log_dir = config['test_settings']['log_dir']
    logger = TensorBoardLogger(log_dir, name="reward_model_test")

    # 创建 DeepSpeed 策略
    deepspeed_config = config['deepspeed_config']
    strategy = DeepSpeedStrategy(config=deepspeed_config)

    # 创建 Trainer
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=-1,  # 使用所有可用的 GPU
        strategy=strategy,
        logger=logger,
        precision=16,  # 使用混合精度
    )

    # 运行测试
    trainer.predict(rm_tester, dataloaders=test_dataloader)

if __name__ == "__main__":
    main()
