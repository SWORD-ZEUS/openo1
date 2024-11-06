import json
from .base_dataset import BaseRewardDataset

class RewardModelDataset(BaseRewardDataset):
    def load_data(self, data_path):
        data = []
        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                question = item['problem']
                steps = item['steps']
                
                # 获取前置steps和当前step对
                previous_steps = steps[:-1]  # 所有前置步骤
                current_step, wrong_step = steps[-1]  # 最后一个元组包含正确和错误的step
                
                # 构造正确样本
                correct_messages = [
                    {"role": "system", "content": "You are a helpful assistant. For each question, your task is to assess the quality and correctness of the last step. Each step starts with '<|start_header_id|>assistant<|end_header_id|> and ends with '<|eot_id|>'. You should focus on the last one. The rating should be between 0, 1, where 0 means the step is incorrect, and 1 means the step is correct."},
                    {"role": "user", "content": question}
                ]
                # 添加前置steps
                for step in previous_steps:
                    correct_messages.append({"role": "assistant", "content": step})
                # 添加当前正确step
                correct_messages.append({"role": "assistant", "content": current_step})
                
                # 构造错误样本
                wrong_messages = correct_messages[:-1]  # 复制除最后一个step外的所有消息
                wrong_messages.append({"role": "assistant", "content": wrong_step})
                
                # 将正确和错误样本作为一对存储
                data.append({
                    'correct': (correct_messages, 0),
                    'wrong': (wrong_messages, -1)
                })
                
        print(f"\n=== 数据统计 ===")
        print(f"总样本对数: {len(data)}")
        print("===================\n")
        
        return data

    def __getitem__(self, idx):
        item = self.data[idx]
        correct_messages, correct_rating = item['correct']
        wrong_messages, wrong_rating = item['wrong']
        
        # 处理正确样本
        correct_formatted = self.tokenizer.apply_chat_template(correct_messages, tokenize=False)
        correct_encoded = self.tokenizer.encode_plus(
            correct_formatted,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # 处理错误样本
        wrong_formatted = self.tokenizer.apply_chat_template(wrong_messages, tokenize=False)
        wrong_encoded = self.tokenizer.encode_plus(
            wrong_formatted,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'correct_input_ids': correct_encoded['input_ids'].squeeze(),
            'correct_attention_mask': correct_encoded['attention_mask'].squeeze(),
            'correct_labels': self._process_rating(correct_rating),
            'correct_step_start_idx': self._get_step_indices(correct_encoded['input_ids'].squeeze(), 
                                                           correct_encoded['attention_mask'].squeeze())['start'],
            'correct_step_end_idx': self._get_step_indices(correct_encoded['input_ids'].squeeze(), 
                                                         correct_encoded['attention_mask'].squeeze())['end'],
            'wrong_input_ids': wrong_encoded['input_ids'].squeeze(),
            'wrong_attention_mask': wrong_encoded['attention_mask'].squeeze(),
            'wrong_labels': self._process_rating(wrong_rating),
            'wrong_step_start_idx': self._get_step_indices(wrong_encoded['input_ids'].squeeze(), 
                                                         wrong_encoded['attention_mask'].squeeze())['start'],
            'wrong_step_end_idx': self._get_step_indices(wrong_encoded['input_ids'].squeeze(), 
                                                       wrong_encoded['attention_mask'].squeeze())['end']
        }
