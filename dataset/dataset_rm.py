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
        # 用于统计不同rating的样本数量
        rating_counts = {-1: 0, 0: 0}
        
        # 按rating分类存储数据
        rating_data = {-1: [], 0: []}
        
        # 首先读取所有数据并按rating分类
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                question = item['problem']
                steps = item['steps']
                
                # 处理整个问题的所有步骤
                processed_data = self.process_question_steps(question, steps)
                for messages, rating in processed_data:
                    rating_data[rating].append((messages, rating))
                    rating_counts[rating] += 1
        
        # 计算需要的复制倍数
        max_count = max(rating_counts.values())
        multipliers = {
            -1: int(max_count / rating_counts[-1]) if rating_counts[-1] > 0 else 0,
            0: 1  # 假设0是多数类
        }
        
        # 对少数类别进行过采样
        for rating in [-1, 0]:
            if rating == -1:
                data.extend(rating_data[rating] * multipliers[rating])
            else:
                data.extend(rating_data[rating])
        
        print(f"\n=== 数据采样统计 ===")
        print(f"原始数据分布: {rating_counts}")
        final_counts = {
            -1: len(rating_data[-1]) * multipliers[-1],
            0: len(rating_data[0])
        }
        print(f"采样后分布: {final_counts}")
        print(f"采样倍数: {multipliers}")
        print("===================\n")
        
        return data

    def process_question_steps(self, question, steps):
        data = []
        messages = [
            {"role": "system", "content": "You are a helpful assistant. For each question, your task is to assess the quality and correctness of the last step. Each step starts with '<|start_header_id|>assistant<|end_header_id|> and ends with '<|eot_id|>'. You should focus on the last one. The rating should be between 0, 1, where 0 means the step is incorrect, and 1 means the step is correct."},
            {"role": "user", "content": question}
        ]
        for step, rating in steps:
            messages.append({"role": "assistant", "content": step})
            data.append((messages.copy(), rating))  # 使用 copy 避免引用问题
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages, rating = self.data[idx]
        
        # 使用apply_chat_template格式化消息
        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        # 先编码整个文本，找到特殊token的位置
        encoded = self.tokenizer.encode_plus(
            formatted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        
        # 找到最后一个assistant消息的开始和结束位置
        # 通常格式是: <|start_header_id|>assistant<|end_header_id|> content <|eot_id|>
        start_token = 128006  # <|start_header_id|>
        end_token = 128009   # <|eot_id|>
        
        # 找到所有start_token的位置
        start_positions = (input_ids == start_token).nonzero(as_tuple=True)[0]
        if len(start_positions) > 0:
            # 先尝试最后一个start_token
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

        # 确保step_start_idx小于step_end_idx
        if step_start_idx >= step_end_idx:
            # 最后一次保护，避免step_start_idx继续大于等于step_end_idx
            step_start_idx = 0
            step_end_idx = len(input_ids) - 1

        # 将rating从-1, 0, 1转换为0, 1, 2
        if self.task == "classification":
            rating += 1  # 将-1, 0, 1转换为0, 1, 2
            rating_tensor = torch.tensor(rating, dtype=torch.long)
        else:
            rating_tensor = torch.tensor(rating, dtype=torch.float)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': rating_tensor,
            'step_start_idx': step_start_idx,
            'step_end_idx': step_end_idx
        }
