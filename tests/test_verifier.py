#运行：torchrun --nproc_per_node={gpus_per_node} tests/test_verifier.py 两个参数：--only_train_head 和 --load_trained_weights
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

from models.verifier.verifier import Verifier
from dataset.test_dataset_verifier import VerifierModelTestDataset  
from utils.process_state_dict import process_state_dict, process_state_dict4verifier
import time

class VerifierModelTester(pl.LightningModule):
    def __init__(self, model, config):
        super().__init__()
        self.model = model
        self.config = config
        self.model.eval()
        self.task = "verifier"  # verifier 固定为"verifier"任务
        print(f"Task: {self.task}")
   
    # def forward(self, input_ids, attention_mask, labels=None, step_start_idx=None, step_end_idx=None):
    #     return self.model.generate(input_ids=input_ids, attention_mask=attention_mask, labels=labels, 
    #                      step_start_idx=step_start_idx, step_end_idx=step_end_idx)

    def predict_step(self, batch, batch_idx):
        messages = batch['messages']
        input_ids = batch['input_ids'].long()
        attention_mask = batch['attention_mask'].float()
        labels = batch['labels']
        step_start_idx = batch['step_start_idx'].long()
        step_end_idx = batch['step_end_idx'].long()

        # 使用generate获取生成文本和分类结果
        generated_ids = self.model.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # step_start_idx=step_start_idx,
            # step_end_idx=step_end_idx,
            max_new_tokens=256,
        )

        _, cls_logits = self.model(
            input_ids, 
            attention_mask=attention_mask,
            step_start_idx=step_start_idx,
            step_end_idx=step_end_idx
        ).logits
        # 解码生成的文本
        generated_texts = self.model.tokenizer.batch_decode(
            generated_ids[:, input_ids.size(1):], 
            skip_special_tokens=True
        )

        return {
            'predictions': {
                'generated_text': generated_texts,
                'cls_preds': torch.softmax(cls_logits, dim=-1)
            },
            'labels': labels,
            'messages': messages
        }

    def on_predict_epoch_end(self):
        from sklearn.metrics import accuracy_score, f1_score
        
        predictions = self.trainer.predict_loop.predictions
        all_generated_texts = []
        all_original_texts = []
        all_cls_predictions = []
        all_cls_labels = []
        predict_gt = []

        for batch_result in predictions:
            if isinstance(batch_result, dict):
                preds = batch_result['predictions']
                labels = batch_result['labels']
                messages = batch_result['messages']
                
                all_generated_texts.extend(preds['generated_text'])
                all_original_texts.extend(labels['response'])
                cls_preds = torch.argmax(preds['cls_preds'], dim=-1).cpu() - 1
                all_cls_predictions.extend(cls_preds)
                all_cls_labels.extend(labels['rating'].cpu())

                for i in range(len(messages)):
                    predict_gt.append({
                        'message': messages[i],
                        'response': labels['response'][i],
                        'generated_text': preds['generated_text'][i],
                        'rating': labels['rating'][i].item(),
                        'cls_pred': cls_preds[i].item()
                    })

        results = {
            "cls_accuracy": float(accuracy_score(all_cls_labels, all_cls_predictions)),
            "cls_f1": float(f1_score(all_cls_labels, all_cls_predictions, average='weighted')),
            "predict_gt": predict_gt
        }

        # 保存结果
        if self.trainer.is_global_zero:
            import json
            path = os.path.join(
                self.config['test_settings']['test_results_dir'],
                f"predict_results-{time.strftime('%m%d%H%M%S')}.json"
            )
            with open(path, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

def load_config(config_path):
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

def main():
    parser = argparse.ArgumentParser(description="Test Verifier Model")
    parser.add_argument("--config", type=str, default="/zhuangkai/openo1/configs/verifier_config_adapter.yaml",
                      help="Path to config file")
    parser.add_argument("--load_trained_weights", action="store_true",
                      help="Whether to load trained weights")
    args = parser.parse_args()

    config = load_config(args.config)

    # 加载模型
    model_path = os.path.join(config['download_model_dir'], config['model_name'])
    config["model_path"] = model_path
    config["fine_tuning"]["only_train_head"] = config["test_settings"]["only_train_head"]
    config["fine_tuning"]["method"] = config["test_settings"]["fine_tuning_method"]
    config["is_test"] = True
    model = Verifier(config, training="verifier")  # 测试模式
    model = model.float()

    # Create directories if they don't exist
    os.makedirs(config['test_settings']['test_results_dir'], exist_ok=True)
    os.makedirs(config['test_settings']['log_dir'], exist_ok=True)

    # 加载训练权重
    if args.load_trained_weights:
        trained_weights_path = config['test_settings']['load_trained_weights_path']
        print(f"Loading trained weights from {trained_weights_path}")
        client_sd = get_fp32_state_dict_from_zero_checkpoint(trained_weights_path)
        processed_sd = process_state_dict4verifier(client_sd)
        try:
            model.load_state_dict(processed_sd, strict=True)
            print("\n成功加载处理后的权重")
        except Exception as e:
            print(f"\n加载权重时出错: {str(e)}")
            raise e

    # 创建测试模型
    verifier_tester = VerifierModelTester(model, config)

    # 加载测试数据集
    test_dataset = VerifierModelTestDataset(config['test_data_path'], model_path, config['max_length'])
    test_dataloader = DataLoader(test_dataset,
                               batch_size=config['batch_size_per_gpu'], 
                               num_workers=4,
                               collate_fn=test_dataset.collate_fn)

    # 设置 logger
    log_dir = config['test_settings']['log_dir']
    logger = TensorBoardLogger(log_dir, name="verifier_test")

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
    trainer.predict(verifier_tester, dataloaders=test_dataloader)

if __name__ == "__main__":
    main()