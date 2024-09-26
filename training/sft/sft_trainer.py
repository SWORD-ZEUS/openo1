import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from peft import LoraConfig, get_peft_model

class SFTTrainer:
    def __init__(self, model, train_dataset, val_dataset, config):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 只优化LoRA参数
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=config.warmup_steps, 
            num_training_steps=config.total_steps
        )
        
    def train(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True)
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # 在每个 epoch 结束后进行验证
            val_loss = self.validate()
            print(f"Epoch {epoch+1} Validation Loss: {val_loss:.4f}")
    
    def validate(self):
        self.model.eval()
        val_dataloader = DataLoader(self.val_dataset, batch_size=self.config.batch_size)
        total_loss = 0
        with torch.no_grad():
            for batch in val_dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                total_loss += loss.item()
        
        return total_loss / len(val_dataloader)
