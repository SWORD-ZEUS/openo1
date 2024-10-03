import sys
import os

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator.generator import Generator
from dataset.dataset import PRM800KDataset
from training.sft.sft_trainer import SFTTrainer
from configs.sft_config import SFTConfig


def main():
    config = SFTConfig()
    model_path = os.path.join(config.download_model_dir, config.model_name)
    print(f"Model path: {model_path}")
    model = Generator(model_path)
    

    train_dataset = PRM800KDataset(config.train_data_path, model_path, config.max_length)
    val_dataset = PRM800KDataset(config.test_data_path, model_path, config.max_length)
    
    save_path = config.weight_save_path
    
    trainer = SFTTrainer(model, train_dataset, val_dataset, config, save_path)
    trainer.train()

if __name__ == "__main__":
    main()
