import sys
import os
import torch
import yaml
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT
import deepspeed
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.generator.generator import Generator

class GeneratorModule(pl.LightningModule):
    def __init__(self, model_path, lora_weights_path=None):
        super().__init__()
        self.generator = Generator(model_path, training=False)
        self.lora_weights_path = lora_weights_path

    def setup(self, stage=None):
        if self.lora_weights_path:
            client_sd = get_fp32_state_dict_from_zero_checkpoint(self.lora_weights_path)
            self.generator.model.load_state_dict(client_sd, strict=False)
        self.generator.model.eval()

    def forward(self, input_ids, attention_mask):
        return self.generator.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=1024)

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        outputs = self(batch['input_ids'], batch['attention_mask'])
        return self.generator.tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # 加载配置
    with open('configs/sft_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    model_path = os.path.join(config['download_model_dir'], config['model_name'])
    lora_weights_path = config['test_settings'].get('load_lora_weights_path', None)  # 可选的 LoRA 权重路径
    print(f"Loading model from {model_path}")
    print(f"Loading LoRA weights from {lora_weights_path}")

    model = GeneratorModule(model_path, lora_weights_path)

    is_log = config['test_settings']['is_log']
    log_dir = config['test_settings']['log_dir']
    trainer = pl.Trainer(accelerator='auto', devices=1, default_root_dir=log_dir, logger=is_log)
    
    # 修改提示信息的格式，明确要求逐步生成文本
    messages = [
        {"role": "system", "content": "You are a helpful assistant. For each question, provide only one step of the solution at a time. After giving each step, wait for the next prompt before continuing."},
        {"role": "user", "content": "Simplify: $\\frac{\\sqrt{2.5^2-0.7^2}}{2.7-2.5}$."},
#         {"role": "assistant", "content": """To simplify the given expression, we need to start by evaluating the numerator, which involves finding the difference of squares. 

# Step 1: Evaluate the squares in the numerator: $2.5^2 = 6.25$ and $0.7^2 = 0.49$."""},
#         {"role": "assistant", "content": """Now that we have the squares, we can proceed with finding the difference.

# Step 2: Find the difference between $2.5^2$ and $0.7^2$: $6.25 - 0.49 = 5.76$."""},
#         {"role": "assistant", "content": """Now that we have the difference of squares, we can proceed with finding the square root.

# Step 3: Find the square root of $5.76$: $\sqrt{5.76} = 2.4$."""},
#         {"role": "assistant", "content": """Now that we have the square root, we can proceed with evaluating the denominator.

# Step 4: Evaluate the denominator: $2.7 - 2.5 = 0.2$."""},

    ]
    
    # 使用 apply_chat_template 格式化消息
    formatted_prompt = model.generator.tokenizer.apply_chat_template(messages, tokenize=False)
    
    inputs = model.generator.tokenizer.encode_plus(formatted_prompt, return_tensors="pt", max_length=512, truncation=True)
    input_data = {'input_ids': inputs['input_ids'].squeeze(), 'attention_mask': inputs['attention_mask'].squeeze()}

    # 使用 DataLoader 来包装输入数据
    from torch.utils.data import DataLoader
    dataset = [input_data]  # 将输入数据作为一个单独的样本
    dataloader = DataLoader(dataset, batch_size=1)

    predictions = trainer.predict(model, dataloader)

    print("Generated steps:(trimmed)")
    print(predictions[0][len(formatted_prompt):])
    print("Generated steps:(full)")
    print(predictions[0])

if __name__ == "__main__":
    main()
