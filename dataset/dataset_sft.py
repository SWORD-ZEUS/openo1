import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch
from utils.prompts import GENERATOR_DATASET_PROMPT

class PRM800KDataset(Dataset):
    def __init__(self, data_path, tokenizer_name, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        self.max_length = max_length
        self.patterns = {
            'generator': torch.tensor([128006, 78191, 128007])
        }
        self.end_token = 128009
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                question = item['problem']
                steps = item['steps']
                # 处理问题的所有步骤
                messages = self.process_question_steps(question, steps)
                data.append(messages)
        return data

    @classmethod
    def process_question_steps(self, question, steps):
        previous_steps = steps.get('previous_steps', [])  # 使用get方法，默认为空列表
        current_step = steps['current_step']
        messages = [
            {"role": "system", "content": GENERATOR_DATASET_PROMPT},
            {"role": "user", "content": question}
        ]
        # 只有当previous_steps非空时才添加
        if previous_steps:
            for step in previous_steps:
                messages.append({"role": "assistant", "content": step})
        messages.append({"role": "assistant", "content": current_step})
        return messages

    def __len__(self):
        return len(self.data)
    
    def _find_pattern_positions(self, input_ids, pattern):
        """通用的模式匹配方法"""
        positions = []
        for i in range(len(input_ids) - len(pattern) + 1):
            if torch.all(input_ids[i:i+len(pattern)] == pattern):
                positions.append(i)
        return positions

    def _find_next_end_token(self, input_ids, start_pos):
        """在指定位置后查找结束标记"""
        end_positions = (input_ids[start_pos:] == self.end_token).nonzero(as_tuple=True)[0]
        return (start_pos + end_positions[0].item()) if len(end_positions) > 0 else None
    
    def _verify_segment(self, input_ids, pattern_key):
        """验证特定段落的完整性"""
        pattern = self.patterns[pattern_key]
        start_positions = self._find_pattern_positions(input_ids, pattern)
        
        if not start_positions:
            return None, None
            
        start_idx = start_positions[-1] + len(pattern)
        end_idx = self._find_next_end_token(input_ids, start_idx)
        
        if end_idx is None:
            return None, None
            
        return start_idx, end_idx

    def __getitem__(self, idx):
        messages = self.data[idx]
        
        # 使用apply_chat_template格式化消息
        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        # 编码格式化后的文本
        encoded = self.tokenizer.encode_plus(
            formatted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()

        # 创建labels，初始化为全-100
        labels = torch.full_like(input_ids, -100)

        start_idx, end_idx = self._verify_segment(input_ids, 'generator')
        if start_idx is not None:
            labels[start_idx:end_idx+1] = input_ids[start_idx:end_idx+1]


        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }