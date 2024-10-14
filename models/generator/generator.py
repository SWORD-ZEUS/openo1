from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch import nn

class Generator(nn.Module):
    def __init__(self, model_path, lora_r=8, lora_alpha=32, lora_dropout=0.1, training=True):
        """
        初始化Generator类
        
        Args:
            model_path (str): Llama 3.1 8B模型的路径
            lora_r (int, optional): LoRA的r值。 Defaults to 8.
            lora_alpha (int, optional): LoRA的alpha值。 Defaults to 32.
            lora_dropout (float, optional): LoRA的dropout值。 Defaults to 0.1.
            training (bool, optional): 是否处于训练模式。 Defaults to True.
        """
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # # 设置填充标记
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

        # 配置LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=not training,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.model = get_peft_model(self.model, peft_config)
    

    def generate_thinking_step(self, prompt, max_length=100):
        """
        生成一个思考步骤
        
        Args:
            prompt (str): 输入提示
            max_length (int): 生成的最大长度
        
        Returns:
            str: 生成的思考步骤
        """
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def forward(self, input_ids, attention_mask, labels=None):
        """
        模型前向传播
        
        Args:
            input_ids (torch.Tensor): 输入ID
            attention_mask (torch.Tensor): 注意力掩码
            labels (torch.Tensor, optional): 标签
        
        Returns:
            transformers.modeling_outputs.CausalLMOutputWithCrossAttentions: 模型输出
        """
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

    def set_inference_mode(self, inference_mode):
        """
        设置模型的推理模式
        
        Args:
            inference_mode (bool): 是否处于推理模式
        """
        for name, module in self.model.named_modules():
            if 'lora' in name:
                module.inference_mode = inference_mode

    def gradient_checkpointing_enable(self):
        """
        启用梯度检查点
        """
        self.model.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        """
        禁用梯度检查点
        """
        self.model.gradient_checkpointing_disable()
