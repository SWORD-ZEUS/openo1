#前期训练时，需要检查input_ids和thinking_step_ids的维度是否为0，不然会报错
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

    def test_tokenizer_output(self):
        # 测试训练集
        self.check_dataset(self.train_dataset, "Training")

        # 测试验证集
        self.check_dataset(self.val_dataset, "Validation")

    def check_dataset(self, dataset, dataset_type):
        for idx in tqdm(range(len(dataset)), desc=f"Checking {dataset_type} Dataset"):
            input_text, thinking_step = dataset.data[idx]
        
            # 编码输入文本，不进行填充
            encoded_input = dataset.tokenizer.encode_plus(
                input_text,
                add_special_tokens=False,  # 不添加特殊token
                max_length=dataset.max_length,
                padding=False,  # 不进行填充
                truncation=True,
                return_tensors='pt'
            )

            # 编码思考步骤，不进行填充
            encoded_thinking_step = dataset.tokenizer.encode_plus(
                thinking_step,
                add_special_tokens=False,  # 不添加特殊token
                max_length=dataset.max_length,
                padding=False,  # 不进行填充
                truncation=True,
                return_tensors='pt'
            )

            input_ids = encoded_input['input_ids'].squeeze()
            thinking_step_ids = encoded_thinking_step['input_ids'].squeeze()

            # 检查input_ids和thinking_step_ids的维度
            self.assertFalse(input_ids.dim() == 0, f"TypeError: input_ids at index {idx} in {dataset_type} dataset has dimension 0.")
            self.assertFalse(thinking_step_ids.dim() == 0, f"TypeError: thinking_step_ids at index {idx} in {dataset_type} dataset has dimension 0.")

if __name__ == "__main__":
    unittest.main()
