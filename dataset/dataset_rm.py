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
                steps = item['steps']
                data.extend(self.process_question_steps(question, steps))
        return data

    def process_question_steps(self, question, steps):
        data = []
        messages = [
            {"role": "system", "content": "You are a helpful assistant. For each question, provide only one step of the solution at a time. After giving each step, wait for the next prompt before continuing."},
            {"role": "user", "content": question}
        ]
        for step, rating in steps:
            messages.append({"role": "assistant", "content": step})
            data.append((messages, rating))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        messages, rating = self.data[idx]
        
        # 使用apply_chat_template格式化消息
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
        rating_tensor = torch.tensor(rating, dtype=torch.float)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': rating_tensor
        }
