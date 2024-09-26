from transformers import LlamaForCausalLM, LlamaTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch

class Generator:
    def __init__(self, model_path, lora_r=8, lora_alpha=32, lora_dropout=0.1):
        """
        初始化Generator类
        
        Args:
            model_path (str): Llama 3 7B模型的路径
            lora_r (int, optional): LoRA的r值。 Defaults to 8.
            lora_alpha (int, optional): LoRA的alpha值。 Defaults to 32.
            lora_dropout (float, optional): LoRA的dropout值。 Defaults to 0.1.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = LlamaForCausalLM.from_pretrained(model_path)
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)

        # 配置LoRA
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout
        )
        self.model = get_peft_model(self.model, peft_config)
        self.model.to(self.device)
    
    def generate_thinking_step(self, prompt, max_length=100):
        """
        生成一个思考步骤
        
        Args:
            prompt (str): 输入提示
            max_length (int): 生成的最大长度
        
        Returns:
            str: 生成的思考步骤
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
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
