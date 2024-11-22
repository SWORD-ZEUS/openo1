from .base_dataset import BaseDataset
import json
import torch
from utils.prompts import VERIFIER_DATASET_PROMPT

class VerifierModelDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 定义标记模式
        self.patterns = {
            'verifier': torch.tensor([128006, 424, 3125, 128007]),
            'step': torch.tensor([128006, 78191, 128007])
        }
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
                messages, rating = self.process_question_steps(question, steps)
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
    
    @classmethod
    def process_question_steps(cls, question, steps):
        """处理问题和步骤"""
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
        return messages, current_rating

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

    def _get_step_indices_verifier(self, input_ids, attention_mask):
        """获取step的索引"""
        start_idx, end_idx = self._verify_segment(input_ids, 'step')
        if start_idx is None:
            return None
            
        return {'start': start_idx, 'end': end_idx}

    def _verify_response(self, input_ids):
        """验证response的完整性"""
        start_idx, end_idx = self._verify_segment(input_ids, 'verifier')
        return start_idx is not None

    def __getitem__(self, idx):
        messages, rating = self.data[idx]
        encoded = self.tokenizer.encode_plus(
            self.tokenizer.apply_chat_template(messages, tokenize=False),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        
        # 验证完整性
        indices = self._get_step_indices_verifier(input_ids, attention_mask)
        if indices is None or not self._verify_response(input_ids):
            return None
        
        # 设置response labels
        labels = torch.full_like(input_ids, -100)
        start_idx, end_idx = self._verify_segment(input_ids, 'verifier')
        if start_idx is not None:
            labels[start_idx:end_idx+1] = input_ids[start_idx:end_idx+1]

        return {
            'messages': messages,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': {
                'rating': self._process_rating(rating),
                'response': labels
            },
            'step_start_idx': indices['start'],
            'step_end_idx': indices['end']
        }

    def collate_fn(self, batch):
        """自定义的collate_fn函数"""
        # 过滤None值
        batch = [b for b in batch if b is not None]
        if not batch:
            print("Empty batch!")
            return None
        
        return {
            'messages': [item['messages'] for item in batch],
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'labels': {
                'rating': torch.stack([item['labels']['rating'] for item in batch]),
                'response': torch.stack([item['labels']['response'] for item in batch])
            },
            'step_start_idx': torch.tensor([item['step_start_idx'] for item in batch]),
            'step_end_idx': torch.tensor([item['step_end_idx'] for item in batch])
        }
