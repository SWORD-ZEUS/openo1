import json
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import torch

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
        
        # 编码输入文本，不进行填充
        encoded_input = self.tokenizer.encode_plus(
            input_text,
            add_special_tokens=False,  # 不添加特殊token
            max_length=self.max_length,
            padding=False,  # 不进行填充
            truncation=True,
            return_tensors='pt'
        )

        # 编码思考步骤，不进行填充
        encoded_thinking_step = self.tokenizer.encode_plus(
            thinking_step,
            add_special_tokens=False,  # 不添加特殊token
            max_length=self.max_length,
            padding=False,  # 不进行填充
            truncation=True,
            return_tensors='pt'
        )


        input_ids = encoded_input['input_ids'].squeeze()
        thinking_step_ids = encoded_thinking_step['input_ids'].squeeze()
        # 添加错误检查
        if isinstance(input_ids, torch.Tensor) and input_ids.dim() == 0:
            print(f"problem: {input_text}")
        if isinstance(thinking_step_ids, torch.Tensor) and thinking_step_ids.dim() == 0:
            print(f"thinking_step: {thinking_step}")


        # 计算剩余空间
        remaining_length = self.max_length - len(input_ids) - len(thinking_step_ids)

        if remaining_length < 0:
            # 如果总长度超过max_length，截断thinking_step
            thinking_step_ids = thinking_step_ids[:remaining_length]
        else:
            # 否则，在thinking_step后面添加填充
            thinking_step_ids = torch.cat([thinking_step_ids, torch.full((remaining_length,), self.tokenizer.pad_token_id)])

        # 拼接input_ids和thinking_step_ids
        combined_ids = torch.cat([input_ids, thinking_step_ids])

        # 创建attention_mask
        attention_mask = torch.ones_like(combined_ids)
        attention_mask[combined_ids == self.tokenizer.pad_token_id] = 0

        # 创建labels，将input部分和padding部分的label设为-100
        labels = combined_ids.clone()
        labels[:len(input_ids)] = -100  # 输入部分设为-100
        labels[combined_ids == self.tokenizer.pad_token_id] = -100  # padding部分设为-100

        return {
            'input_ids': combined_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
