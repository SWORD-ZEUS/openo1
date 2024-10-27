from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch import nn
from models.adapter import Adapter
from transformers.modeling_outputs import CausalLMOutputWithPast

class Generator(nn.Module):
    def __init__(self, config, training=True):
        """
        初始化Generator类
        
        Args:
            config (dict): 配置信息，包含模型路径和微调方式等
            training (bool, optional): 是否处于训练模式。 Defaults to True.
        """
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(
            config['model_path'], 
            torch_dtype=torch.float16
        )
        self.tokenizer = AutoTokenizer.from_pretrained(config['model_path'])
        self.config = config
        
        # 根据配置选择微调方式
        if config['fine_tuning']['method'] == 'lora':
            self._init_lora(training)
        elif config['fine_tuning']['method'] == 'adapter':
            self._init_adapter()
            
    def _init_lora(self, training):
        """初始化LoRA"""
        lora_config = self.config['fine_tuning']['lora_config']
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=not training,
            r=lora_config['r'],
            lora_alpha=lora_config['alpha'],
            lora_dropout=lora_config['dropout']
        )
        self.model = get_peft_model(self.model, peft_config)

    def _init_adapter(self):
        """初始化Adapter"""
        adapter_config = self.config['fine_tuning']['adapter_config']
        hidden_size = self.model.config.hidden_size
        adapter_size = adapter_config['hidden_size']
        adapter_dropout = adapter_config['adapter_dropout']
        adapter_layers = adapter_config['adapter_layers']

        # 为指定的transformer层添加adapter
        for i, layer in enumerate(self.model.model.layers):
            if i in adapter_layers:
                adapter = Adapter(hidden_size, adapter_size, adapter_dropout)
                layer.adapter = adapter

        # 冻结原始模型参数
        for param in self.model.parameters():
            param.requires_grad = False
        # 解冻adapter参数
        for name, param in self.model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True

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

    def forward(self, input_ids, attention_mask, labels=None, position_ids=None):
        """
        模型前向传播
        
        Args:
            input_ids: 输入的token ids
            attention_mask: 注意力掩码
            labels: 标签 (可选)
            position_ids: 位置编码 (可选)
        """
        if self.config['fine_tuning']['method'] == 'adapter':
            # 确保输入在正确的设备上并且类型正确
            device = input_ids.device
            dtype = torch.float16 if self.model.dtype == torch.float16 else torch.float32
            
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device) if attention_mask is not None else None
            position_ids = position_ids.to(device) if position_ids is not None else None
            labels = labels.to(device) if labels is not None else None
            
            # 获取序列长度
            sequence_length = input_ids.shape[1]
            
            # 将注意力掩码转换为4D格式
            if attention_mask is not None:
                attention_mask = _prepare_4d_causal_attention_mask(
                    attention_mask,
                    sequence_length,
                    dtype,
                    device
                )
            
            # 获取原始模型的所有模块
            model_layers = self.model.model.layers
            
            # 获取输入嵌入
            inputs_embeds = self.model.model.embed_tokens(input_ids)
            hidden_states = inputs_embeds
            
            # 处理 position_ids
            if position_ids is None:
                # 创建位置编码
                position_ids = torch.arange(
                    input_ids.shape[1], 
                    dtype=torch.long, 
                    device=input_ids.device
                ).unsqueeze(0).expand(input_ids.shape[0], -1)
            
            # 逐层处理
            for i, layer in enumerate(model_layers):
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_attentions=False
                )
                hidden_states = layer_outputs[0]
                
                # 如果该层有adapter，则应用adapter
                if hasattr(layer, 'adapter'):
                    hidden_states = layer.adapter(hidden_states)
            
            # 通过最后的层范数
            hidden_states = self.model.model.norm(hidden_states)
            
            # 计算语言模型的输出
            lm_logits = self.model.lm_head(hidden_states)
            
            # 计算损失
            loss = None
            if labels is not None:
                # 重塑 logits 和标签
                shift_logits = lm_logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                # 计算损失
                loss_fct = torch.nn.CrossEntropyLoss()
                loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            return CausalLMOutputWithPast(
                loss=loss,
                logits=lm_logits,
                past_key_values=None,
                hidden_states=hidden_states,
                attentions=None
            )
        else:
            return self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                labels=labels
            )

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

def _expand_mask(mask, dtype):
    """
    扩展注意力掩码从 [batch_size, seq_length] 到 [batch_size, 1, seq_length, seq_length]
    
    Args:
        mask: 输入掩码
        dtype: 目标浮点数据类型(torch.float16 或 torch.float32)
    """
    batch_size, seq_length = mask.shape
    causal_mask = torch.triu(
        torch.ones((seq_length, seq_length), dtype=dtype, device=mask.device), 
        diagonal=1
    )
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    
    # [batch_size, 1, seq_length, 1] * [1, 1, 1, seq_length]
    expanded_mask = mask.to(dtype).unsqueeze(1).unsqueeze(2)
    expanded_mask = expanded_mask.expand(batch_size, 1, seq_length, seq_length)
    
    expanded_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
    expanded_mask = expanded_mask.masked_fill(causal_mask.bool(), torch.finfo(dtype).min)
    
    return expanded_mask
        

def _prepare_4d_causal_attention_mask(
    attention_mask: torch.Tensor,
    sequence_length: int,
    dtype: torch.dtype,
    device: torch.device,
):
    """
    准备4D的因果注意力掩码
    
    Args:
        attention_mask: 原始注意力掩码 [batch_size, seq_length]
        sequence_length: 序列长度
        dtype: 数据类型
        device: 设备
    """
    # 创建因果掩码
    causal_mask = torch.triu(
        torch.ones((sequence_length, sequence_length), dtype=dtype, device=device),
        diagonal=1,
    )
    
    # 扩展维度
    causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
    
    # 处理注意力掩码
    if attention_mask is not None:
        # 扩展注意力掩码维度 [batch_size, 1, seq_length, seq_length]
        expanded_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        expanded_mask = expanded_mask.expand(-1, -1, sequence_length, -1)
        expanded_mask = expanded_mask.to(dtype)
        
        # 组合因果掩码和注意力掩码
        expanded_mask = (1.0 - expanded_mask) * torch.finfo(dtype).min
        causal_mask = causal_mask.to(dtype)
        expanded_mask = expanded_mask.masked_fill(causal_mask.bool(), torch.finfo(dtype).min)
        
        return expanded_mask
    else:
        # 如果没有注意力掩码，只返回因果掩码
        return causal_mask.to(dtype)
