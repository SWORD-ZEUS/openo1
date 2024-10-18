import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

class PRM800KDataset(Dataset):
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
                data.append(self.process_question_steps(question, steps))
        return data

    def process_question_steps(self, question, steps):
        # TODO: 后续需要用新的system_prompt再sft训练一次
        messages = [
            {"role": "system", "content": "You are a helpful assistant. For each question, provide only one step of the solution at a time. After giving each step, wait for the next prompt before continuing."},
            {"role": "user", "content": question}
        ]
        for step in steps:
            messages.append({"role": "assistant", "content": step})
        return messages

    def __len__(self):
        return len(self.data)

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

        # 查找所有assistant回复的开始和结束位置
        start_pattern = torch.tensor([128006, 78191, 128007])
        end_token = 128009

        # 在input_ids中查找开始模式
        for i in range(len(input_ids) - 2):
            if torch.all(input_ids[i:i+3] == start_pattern):
                # 从这个位置开始查找结束标记
                for j in range(i+3, len(input_ids)):
                    if input_ids[j] == end_token:
                        # 设置labels，注意不包括开始模式
                        labels[i+3:j+1] = input_ids[i+3:j+1]
                        break

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }