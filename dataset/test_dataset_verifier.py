from dataset.dataset_verifier import VerifierModelDataset
import torch

class VerifierModelTestDataset(VerifierModelDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, idx):
        messages_orig, rating = self.data[idx]
        
        # 移除最后一个verifier的回复
        messages = messages_orig[:-1]
        
        encoded = self.tokenizer.encode_plus(
            self.tokenizer.apply_chat_template(messages, tokenize=False),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        input_ids = encoded['input_ids'].squeeze()
        attention_mask = encoded['attention_mask'].squeeze()
        
        # 获取step的索引
        indices = self._get_step_indices_verifier(input_ids)
        if indices is None:
            return None

        return {
            'messages': messages,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': {
                'rating': self._process_rating(rating),
                'response': messages_orig[-1]["content"] if messages_orig else ""
            },
            'step_start_idx': indices['start'],
            'step_end_idx': indices['end']
        }

    def collate_fn(self, batch):
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
                'response': [item['labels']['response'] for item in batch]
            },
            'step_start_idx': torch.tensor([item['step_start_idx'] for item in batch]),
            'step_end_idx': torch.tensor([item['step_end_idx'] for item in batch])
        }