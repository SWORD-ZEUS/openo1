import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class RewardModelDataset(Dataset):
    def __init__(self, data_path, tokenizer_name, max_length=512, task="regression"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        self.max_length = max_length
        self.task = task
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                question = item['problem']
                steps = item['steps']
                
                # 获取前置steps和当前step对
                previous_steps = steps[:-1]  # 所有前置步骤
                current_step, wrong_step = steps[-1]  # 最后一个元组包含正确和错误的step
                
                # 构造正确样本
                correct_messages = [
                    {"role": "system", "content": "You are a helpful assistant. For each question, your task is to assess the quality and correctness of the last step. Each step starts with '<|start_header_id|>assistant<|end_header_id|> and ends with '<|eot_id|>'. You should focus on the last one. The rating should be between 0, 1, where 0 means the step is incorrect, and 1 means the step is correct."},
                    {"role": "user", "content": question}
                ]
                # 添加前置steps
                for step in previous_steps:
                    correct_messages.append({"role": "assistant", "content": step})
                # 添加当前正确step
                correct_messages.append({"role": "assistant", "content": current_step})
                
                # 构造错误样本
                wrong_messages = correct_messages[:-1]  # 复制除最后一个step外的所有消息
                wrong_messages.append({"role": "assistant", "content": wrong_step})
                
                # 将正确和错误样本作为一对存储
                data.append({
                    'correct': (correct_messages, 0),
                    'wrong': (wrong_messages, -1)
                })
        
        print(f"\n=== 数据统计 ===")
        print(f"总样本对数: {len(data)}")
        print("===================\n")
        
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        correct_messages, correct_rating = item['correct']
        wrong_messages, wrong_rating = item['wrong']
        
        # 处理正确样本
        correct_formatted = self.tokenizer.apply_chat_template(correct_messages, tokenize=False)
        correct_encoded = self.tokenizer.encode_plus(
            correct_formatted,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 处理错误样本
        wrong_formatted = self.tokenizer.apply_chat_template(wrong_messages, tokenize=False)
        wrong_encoded = self.tokenizer.encode_plus(
            wrong_formatted,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 获取正确样本的索引
        correct_indices = self._get_step_indices(correct_encoded['input_ids'].squeeze(), correct_encoded['attention_mask'].squeeze())
        # 获取错误样本的索引
        wrong_indices = self._get_step_indices(wrong_encoded['input_ids'].squeeze(), wrong_encoded['attention_mask'].squeeze())
        
        # 处理rating
        if self.task == "classification":
            correct_rating += 1  # 将0转换为1
            wrong_rating += 1    # 将-1转换为0
            correct_rating_tensor = torch.tensor(correct_rating, dtype=torch.long)
            wrong_rating_tensor = torch.tensor(wrong_rating, dtype=torch.long)
        else:
            correct_rating_tensor = torch.tensor(correct_rating, dtype=torch.float)
            wrong_rating_tensor = torch.tensor(wrong_rating, dtype=torch.float)
        
        return {
            'correct_input_ids': correct_encoded['input_ids'].squeeze(),
            'correct_attention_mask': correct_encoded['attention_mask'].squeeze(),
            'correct_labels': correct_rating_tensor,
            'correct_step_start_idx': correct_indices['start'],
            'correct_step_end_idx': correct_indices['end'],
            'wrong_input_ids': wrong_encoded['input_ids'].squeeze(),
            'wrong_attention_mask': wrong_encoded['attention_mask'].squeeze(),
            'wrong_labels': wrong_rating_tensor,
            'wrong_step_start_idx': wrong_indices['start'],
            'wrong_step_end_idx': wrong_indices['end']
        }

    def _get_step_indices(self, input_ids, attention_mask):
        """辅助函数：获取step的开始和结束索引"""
        start_token = 128006  # <|start_header_id|>
        end_token = 128009   # <|eot_id|>
        
        # 找到最后一个start_token的位置
        start_positions = (input_ids == start_token).nonzero(as_tuple=True)[0]
        if len(start_positions) > 0:
            # 跳过<|start_header_id|>assistant<|end_header_id|>
            step_start_idx = start_positions[-1].item() + 3
        else:
            step_start_idx = 0
            
        # 开始找end_token
        end_positions = (input_ids == end_token).nonzero(as_tuple=True)[0]
        if len(end_positions) > 0:
            # content结束的位置是eot
            step_end_idx = end_positions[-1].item()
        else:
            step_end_idx = len(input_ids) - 1
        
        # 如果step_start_idx大于或等于step_end_idx，则尝试使用倒数第二个start_token
        if step_start_idx >= step_end_idx:
            if len(start_positions) >= 2:
                step_start_idx = start_positions[-2].item() + 3
                # 默认上一个step是对的
                rating = 0
                # 将被截断的step部分用padding填充
                input_ids[step_end_idx+1:] = self.tokenizer.pad_token_id
                attention_mask[step_end_idx+1:] = 0
            else:
                # 如果没有第二个start_token，则设置默认值
                step_start_idx = 0
                step_end_idx = end_positions[-1].item() if len(end_positions) > 0 else self.max_length - 1
                rating = -1
        
        return {'start': step_start_idx, 'end': step_end_idx}
