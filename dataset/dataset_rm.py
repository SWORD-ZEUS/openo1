import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class RewardModelDataset(Dataset):
    def __init__(self, data_path, tokenizer_name, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        self.max_length = max_length
        self.data = self.load_data(data_path)

    def load_data(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                question = item['problem']
                steps_ratings = item['steps_ratings']  # 假设数据中包含步骤和评分
                data.append(self.process_question_steps_ratings(question, steps_ratings))
        return data

    def process_question_steps_ratings(self, question, steps_ratings):
        messages = [
            {"role": "system", "content": "You are a helpful assistant. For each question, provide only one step of the solution at a time. After giving each step, wait for the next prompt before continuing."},
            {"role": "user", "content": question}
        ]
        ratings = []
        for step, rating in steps_ratings:
            messages.append({"role": "assistant", "content": step})
            ratings.append(rating)
        return messages, ratings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages, ratings = self.data[idx]
        
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

        # 将评分转换为张量
        ratings_tensor = torch.tensor(ratings, dtype=torch.float)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'ratings': ratings_tensor
        }