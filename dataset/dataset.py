import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer

class PRM800KDataset(Dataset):
    def __init__(self, data_path, tokenizer_name, max_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
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
        processed_data = []
        current_input = question
        for step in steps:
            processed_data.append((current_input, step))
            current_input = f"{current_input}\n{step}"
        return processed_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        input_text, thinking_step = self.data[idx]
        
        encoded_input = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        encoded_thinking_step = self.tokenizer.encode_plus(
            thinking_step,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoded_input['input_ids'].squeeze(),
            'attention_mask': encoded_input['attention_mask'].squeeze(),
            'labels': encoded_thinking_step['input_ids'].squeeze()
        }
