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
        return self.generator.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=512)

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> STEP_OUTPUT:
        outputs = self(batch['input_ids'], batch['attention_mask'])
        return self.generator.tokenizer.decode(outputs[0], skip_special_tokens=True)

def main():
    # 加载配置
    with open('configs/sft_config.yaml', 'r') as file:
        config = yaml.safe_load(file)

    model_path = os.path.join(config['download_model_dir'], config['model_name'])
    lora_weights_path = config.get('load_lora_weights_path', None)  # 可选的 LoRA 权重路径
    print(f"Loading model from {model_path}")
    print(f"Loading LoRA weights from {lora_weights_path}")

    model = GeneratorModule(model_path, lora_weights_path)

    trainer = pl.Trainer(accelerator='auto', devices=1)
    # prompt = "what is the result of 1+1"
    # prompt = "Compute\n\\[\\frac{1}{\\cos^2 10^\\circ} + \\frac{1}{\\sin^2 20^\\circ} + \\frac{1}{\\sin^2 40^\\circ}.\\]"
    prompt = """<s>[INST] <<SYS>>
You are a helpful, respectful, and honest assistant. Always provide accurate information and if you're not sure about something, say so.
<</SYS>>

Hello! [/INST]"""
    inputs = model.generator.tokenizer.encode_plus(prompt, return_tensors="pt")
    input_data = {'input_ids': inputs['input_ids'].squeeze(), 'attention_mask': inputs['attention_mask'].squeeze()}

    # 使用 DataLoader 来包装输入数据
    from torch.utils.data import DataLoader
    dataset = [input_data]  # 将输入数据作为一个单独的样本
    dataloader = DataLoader(dataset, batch_size=1)

    predictions = trainer.predict(model, dataloader)

    print("Generated steps:")
    print(predictions[0])

if __name__ == "__main__":
    main()
