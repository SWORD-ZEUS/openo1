#查看formatted_text以及labels
import unittest
import json
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dataset.dataset import PRM800KDataset
import torch
from tqdm import tqdm

class TestPRM800KDataset(unittest.TestCase):
    def setUp(self):
        # 设置训练集和验证集的数据路径和tokenizer名称
        self.train_data_path = "/zhuangkai/openo1/dataset/prm800k/processsed/phase2_train_gt_pre_generated_solution_to_step_list.jsonl"
        self.val_data_path = "/zhuangkai/openo1/dataset/prm800k/processsed/phase2_validation_gt_pre_generated_solution_to_step_list.jsonl"
        self.tokenizer_name = "/storage/zhuangkai/data/openo1/download_model_weights/Meta-Llama-3.1-8B-Instruct"
        
        # 加载训练集和验证集
        self.train_dataset = PRM800KDataset(self.train_data_path, self.tokenizer_name)
        self.val_dataset = PRM800KDataset(self.val_data_path, self.tokenizer_name)

    def test_dataset_output(self):
        # 测试训练集
        self.check_dataset(self.train_dataset, "Training")

        # 测试验证集
        self.check_dataset(self.val_dataset, "Validation")

    def check_dataset(self, dataset, dataset_type):
        for idx in tqdm(range(len(dataset)), desc=f"检查 {dataset_type} 数据集"):
            item = dataset[idx]
            
            # 获取formatted_text
            messages = dataset.data[idx]
            formatted_text = dataset.tokenizer.apply_chat_template(messages, tokenize=False)
            
            print("Formatted Text:")
            print(formatted_text)
            
            print("\nDecoded Input IDs (including padding tokens):")
            decoded_text = dataset.tokenizer.decode(item['input_ids'])
            print(decoded_text)
            
            print("\nInput IDs:")
            print(item['input_ids'])
            
            print("\nAttention Mask:")
            print(item['attention_mask'])
            
            print("\nLabels:")
            print(item['labels'])
            
            # 只打印一个样本后就退出循环
            break

if __name__ == "__main__":
    unittest.main()
