import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from peft import LoraConfig, get_peft_model
import os
from torch.cuda.amp import autocast, GradScaler
import wandb
import time

class SFTTrainer:
    def __init__(self, model, train_dataset, val_dataset, config, save_path):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config
        self.save_path = save_path
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # 只优化LoRA参数
        self.optimizer = AdamW(self.model.parameters(), lr=config.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer, 
            num_warmup_steps=config.warmup_steps, 
            num_training_steps=config.total_steps
        )
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)
        
        if config.use_wandb:
            wandb.init(
                project=config.wandb_project, 
                entity=config.wandb_entity, 
                config=vars(config),
                name=config.wandb_name+f"_{time.strftime('%m%d%H%M%S')}",
                tags=config.wandb_tags,
                notes=config.wandb_notes,
                group=config.wandb_group,
                job_type=config.wandb_job_type,
                dir=config.wandb_dir
            )
    
    def train(self):
        train_dataloader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, shuffle=True, pin_memory=True, num_workers=4)
        
        global_step = 0
        steps_per_epoch = len(train_dataloader)
        val_step = int(steps_per_epoch * self.config.val_interval)
        
        for epoch in range(self.config.num_epochs):
            self.model.train()
            epoch_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}")):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                epoch_loss += loss.item()
                
                loss.backward()
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                global_step += 1
                
                if self.config.use_wandb:
                    wandb.log({
                        "step": global_step,
                        "train_loss": loss.item(),
                        "learning_rate": self.scheduler.get_last_lr()[0]
                    })
                
                if (step + 1) % val_step == 0 or (step + 1) == steps_per_epoch:
                    avg_train_loss = epoch_loss / (step + 1)
                    val_loss = self.validate()
                    
                    if self.config.use_wandb:
                        wandb.log({
                            "epoch": epoch + (step + 1) / steps_per_epoch,
                            "avg_train_loss": avg_train_loss,
                            "val_loss": val_loss,
                        })
                    
                    print(f"Epoch {epoch+1} Step {step+1}/{steps_per_epoch} - Train Loss: {avg_train_loss:.4f}, Validation Loss: {val_loss:.4f}")
                    
                    # Save model weights
                    save_path = os.path.join(self.save_path, f"checkpoint_epoch{epoch+1}_step{step+1}.pt")
                    torch.save(self.model.state_dict(), save_path)
                    print(f"Model saved to {save_path}")
        
        if self.config.use_wandb:
            wandb.finish()
    
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
