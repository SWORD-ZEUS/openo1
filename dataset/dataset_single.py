from .base_dataset import BaseRewardDataset
import json

class SingleInputDataset(BaseRewardDataset):
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
            data.append((messages, rating))
        return data
    
    def __getitem__(self, idx):
        messages, rating = self.data[idx]
        formatted = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False
        )
        
        encoded = self.tokenizer.encode_plus(
            formatted,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        indices = self._get_step_indices(
            encoded['input_ids'].squeeze(),
            encoded['attention_mask'].squeeze()
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': self._process_rating(rating),
            'step_start_idx': indices['start'],
            'step_end_idx': indices['end']
        } 

