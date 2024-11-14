from .base_dataset import BaseDataset
import json
import torch
from utils.prompts import VERIFIER_DATASET_PROMPT

class VerifierModelDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 定义verifier回复的标记模式
        self.verifier_start_pattern = torch.tensor([128006, 424, 3125, 128007])
        self.step_start_pattern = torch.tensor([128006, 78191, 128007])
        self.end_token = 128009

    def load_data(self, data_path):
        data = []
        rating_counts = {}
        rating_data = {}
        
        # 读取数据并按rating分类
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                question = item['problem']
                steps = item['steps']
                
                # 处理问题的所有步骤
                processed_data = self.process_question_steps(question, steps)
                for messages, rating in processed_data:
                    if rating not in rating_counts:
                        rating_counts[rating] = 0
                        rating_data[rating] = []
                    rating_data[rating].append((messages, rating))
                    rating_counts[rating] += 1
        
        # 找出样本数最多的类别
        majority_class = max(rating_counts.items(), key=lambda x: x[1])[0]
        max_count = rating_counts[majority_class]
        
        # 计算每个类别的过采样倍数
        multipliers = {
            rating: int(max_count / count) if count > 0 else 0 
            for rating, count in rating_counts.items()
        }
        multipliers[majority_class] = 1  # 多数类不需要过采样
        
        # 对少数类进行过采样
        for rating in rating_counts.keys():
            if rating != majority_class:
                data.extend(rating_data[rating] * multipliers[rating])
            else:
                data.extend(rating_data[rating])
        
        # 打印数据统计信息
        print(f"\n=== Data Sampling Statistics ===")
        print(f"Original distribution: {rating_counts}")
        final_counts = {
            rating: len(rating_data[rating]) * multipliers[rating]
            for rating in rating_counts.keys()
        }
        print(f"Final distribution: {final_counts}")
        print(f"Sampling multipliers: {multipliers}")
        print("===========================\n")
        
        return data

    def process_question_steps(self, question, steps):
        """处理问题和步骤"""
        data = []
        previous_steps = steps.get('previous_steps', [])  # 使用get方法，默认为空列表
        current_step = steps['current_step']
        current_rating = steps['current_rating']
        current_resp = steps['current_response']
        messages = [
            {"role": "system", "content": VERIFIER_DATASET_PROMPT},
            {"role": "user", "content": question}
        ]
        # 只有当previous_steps非空时才添加
        if previous_steps:
            for step in previous_steps:
                messages.append({"role": "assistant", "content": step})
        messages.append({"role": "assistant", "content": current_step})
        messages.append({"role": "verifier", "content": current_resp})
        data.append((messages.copy(), current_rating))
        return data

    def __getitem__(self, idx):
        messages, rating = self.data[idx]
        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        encoded = self.tokenizer.encode_plus(
            formatted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        
        # 首先验证step的完整性
        indices = self._get_step_indices(input_ids, attention_mask)
        if indices is None:
            return None
            
        # 然后验证response的完整性
        if not self._verify_response(input_ids):
            return None
            
        labels = torch.full_like(input_ids, -100)
        
        # 设置response labels
        start_idx = self._find_pattern_index(input_ids)
        if start_idx is not None:
            end_idx = self._find_end_token(input_ids, start_idx + 4)
            if end_idx is not None:
                labels[start_idx+4:end_idx+1] = input_ids[start_idx+4:end_idx+1]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': {
                'rating': self._process_rating(rating),
                'response': labels
            },
            'step_start_idx': indices['start'],
            'step_end_idx': indices['end']
        }

    def _verify_response(self, input_ids):
        """验证response是否完整"""
        start_idx = self._find_pattern_index(input_ids)
        if start_idx is None:
            return False
        
        end_idx = self._find_end_token(input_ids, start_idx + 4)
        return end_idx is not None

    def _find_pattern_index(self, input_ids):
        """查找开始模式的位置"""
        for i in range(len(input_ids) - len(self.verifier_start_pattern) + 1):
            if torch.all(input_ids[i:i+len(self.verifier_start_pattern)] == self.verifier_start_pattern):
                return i
        return None

    def _find_end_token(self, input_ids, start_pos):
        """查找结束标记的位置"""
        for j in range(start_pos, len(input_ids)):
            if input_ids[j] == self.end_token:
                return j
        return None

    def _get_step_indices(self, input_ids, attention_mask):
        """
        重写基类方法，查找最后一组assistant回答的位置
        返回: dict with start和end索引
        """
        # 查找所有可能的step起始位置
        start_positions = []
        for i in range(len(input_ids) - len(self.step_start_pattern) + 1):
            if torch.all(input_ids[i:i+len(self.step_start_pattern)] == self.step_start_pattern):
                start_positions.append(i)
        
        if not start_positions:
            return None
            
        # 获取最后一个step的起始位置
        last_start = start_positions[-1]
        start_idx = last_start + len(self.step_start_pattern)
        
        # 在start_idx之后查找结束标记
        end_idx = None
        for i in range(start_idx, len(input_ids)):
            if input_ids[i] == self.end_token:
                end_idx = i
                break
                
        if end_idx is None:
            return None
            
        return {
            'start': start_idx,
            'end': end_idx
        }

    def collate_fn(self, batch):
        """自定义的collate_fn函数"""
        # 过滤None值
        batch = [b for b in batch if b is not None]
        if not batch:
            return None
        
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': {
                'rating': torch.stack([item['labels']['rating'] for item in batch]),
                'response': torch.stack([item['labels']['response'] for item in batch])
            },
            'step_start_idx': torch.tensor([item['step_start_idx'] for item in batch]),
            'step_end_idx': torch.tensor([item['step_end_idx'] for item in batch])
        }
