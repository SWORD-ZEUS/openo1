from .base_dataset import BaseRewardDataset
import json
import torch

class RewardModelDataset(BaseRewardDataset):
    def load_data(self, data_path):
        """加载并处理偏好数据"""
        data = []
        rating_counts = {-1: 0, 0: 0}
        rating_data = {-1: [], 0: []}
        
        # 读取数据并按rating分类
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                question = item['problem']
                steps = item['steps']
                
                # 处理问题的所有步骤
                processed_data = self.process_question_steps(question, steps)
                for messages, rating in processed_data:
                    rating_data[rating].append((messages, rating))
                    rating_counts[rating] += 1
        
        # 计算过采样倍数
        max_count = max(rating_counts.values())
        multipliers = {
            -1: int(max_count / rating_counts[-1]) if rating_counts[-1] > 0 else 0,
            0: 1  # 假设0是多数类
        }
        
        # 对少数类进行过采样
        for rating in [-1, 0]:
            if rating == -1:
                data.extend(rating_data[rating] * multipliers[rating])
            else:
                data.extend(rating_data[rating])
        
        # 打印数据统计信息
        print(f"\n=== Data Sampling Statistics ===")
        print(f"Original distribution: {rating_counts}")
        final_counts = {
            -1: len(rating_data[-1]) * multipliers[-1],
            0: len(rating_data[0])
        }
        print(f"Final distribution: {final_counts}")
        print(f"Sampling multipliers: {multipliers}")
        print("===========================\n")
        
        return data

    def process_question_steps(self, question, steps):
        """处理问题和步骤"""
        data = []
        messages = [
            {"role": "system", "content": "You are a helpful assistant. For each question, your task is to assess the quality and correctness of the last step. Each step starts with '<|start_header_id|>assistant<|end_header_id|> and ends with '<|eot_id|>'. You should focus on the last one. The rating should be between 0, 1, where 0 means the step is incorrect, and 1 means the step is correct."},
            {"role": "user", "content": question}
        ]
        for step, rating in steps:
            messages.append({"role": "assistant", "content": step})
            data.append((messages.copy(), rating))
        return data

    def __getitem__(self, idx):
        """获取单个数据样本"""
        messages, rating = self.data[idx]
        
        # 使用apply_chat_template格式化消息
        formatted_text = self.tokenizer.apply_chat_template(messages, tokenize=False)
        
        # 编码文本
        encoded = self.tokenizer.encode_plus(
            formatted_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # 获取输入ID和注意力掩码
        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        
        # 获取step索引
        indices = self._get_step_indices(input_ids, attention_mask)
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': self._process_rating(rating),
            'step_start_idx': indices['start'],
            'step_end_idx': indices['end']
        }
