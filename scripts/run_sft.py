from models.generator.generator import Generator
from data.dataset import PRM800KDataset
from training.sft.sft_trainer import SFTTrainer
from configs.sft_config import SFTConfig

def main():
    config = SFTConfig()
    
    model = Generator(config.model_name)
    
    train_dataset = PRM800KDataset('path/to/train/data', config.model_name, config.max_length)
    val_dataset = PRM800KDataset('path/to/val/data', config.model_name, config.max_length)
    
    trainer = SFTTrainer(model, train_dataset, val_dataset, config)
    trainer.train()

if __name__ == "__main__":
    main()
