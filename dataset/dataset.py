import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class PRM800KDataset(Dataset):
    def __init__(self, data_path, tokenizer_name, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_length = max_length
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        # 实现加载 PRM800K 数据集的逻辑
        # 返回一个包含 (input, thinking_step) 对的列表
        pass

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, thinking_step = self.data[idx]
        
        encoded_input = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        encoded_thinking_step = self.tokenizer.encode_plus(
            thinking_step,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded_input['input_ids'].squeeze(),
            'attention_mask': encoded_input['attention_mask'].squeeze(),
            'labels': encoded_thinking_step['input_ids'].squeeze()
        }
