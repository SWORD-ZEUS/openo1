import json
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class BaseDataset(Dataset):
    def __init__(self, data_path, tokenizer_name, max_length=512, task="regression"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        self.max_length = max_length
        self.task = task
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        """由子类实现具体的数据加载逻辑"""
        raise NotImplementedError
        
    def __len__(self):
        return len(self.data)
        
    def _get_step_indices(self, input_ids, attention_mask, rating):
        """获取step的开始和结束索引"""
        start_token = 128006  # <|start_header_id|>
        end_token = 128009   # <|eot_id|>
        
        rating = rating
        start_positions = (input_ids == start_token).nonzero(as_tuple=True)[0]
        if len(start_positions) > 0:
            step_start_idx = start_positions[-1].item() + 3
        else:
            step_start_idx = 0
            
        end_positions = (input_ids == end_token).nonzero(as_tuple=True)[0]
        if len(end_positions) > 0:
            step_end_idx = end_positions[-1].item()
        else:
            step_end_idx = len(input_ids) - 1
            
        if step_start_idx >= step_end_idx:
            if len(start_positions) >= 2:
                step_start_idx = start_positions[-2].item() + 3
                input_ids[step_end_idx+1:] = self.tokenizer.pad_token_id
                attention_mask[step_end_idx+1:] = 0
                rating = 0
            else:
                step_start_idx = 0
                step_end_idx = end_positions[-1].item() if len(end_positions) > 0 else self.max_length - 1
                rating = -1
                
        return {'start': step_start_idx, 'end': step_end_idx, 'rating': rating}

    def _process_rating(self, rating):
        """处理评分"""
        if self.task == "classification":
            rating += 1  # 将0/1转换为1/2或将-1/0转换为0/1
            return torch.tensor(rating, dtype=torch.long)
        return torch.tensor(rating, dtype=torch.float) 