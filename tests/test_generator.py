import sys
import os
import torch

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.generator.generator import Generator
from configs.sft_config import SFTConfig

def main():
    config = SFTConfig()
    model_path = os.path.join(config.download_model_dir, config.model_name)
    lora_weights_path = config.weight_save_path

    # 初始化 Generator，设置为推理模式
    generator = Generator(model_path, training=False)
    
    # 加载训练好的 LoRA 权重
    generator.model.load_state_dict(torch.load(os.path.join(lora_weights_path, "pytorch_model.bin")), strict=False)
    
    # 将模型设置为评估模式
    generator.model.eval()

    # 测试生成
    prompt = "Solve the following problem step by step: What is the sum of the first 10 positive integers?"
    generated_steps = generator.generate_thinking_step(prompt, max_length=200)
    
    print("Generated steps:")
    print(generated_steps)

if __name__ == "__main__":
    main()
