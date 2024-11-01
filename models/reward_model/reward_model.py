# from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, LlamaForSequenceClassification
from transformers import modeling_outputs
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch import nn
from models.adapter import Adapter
from models.cross_attention import CrossAttentionLayer

class RewardModel(nn.Module):
    def __init__(self,
                 config,
                 training=True):
        """
        初始化RewardModelUsingAdapter类
        
        Args:
            config (dict): 配置信息，包含模型路径和微调方式等
            training (bool, optional): 是否处于训练模式。 Defaults to True.
        """
        self.config = config
        super().__init__()

        # load base model and tokenizer
        assert 'model_path' in self.config, "model_path is required in config"
        self.num_labels = self.config['fine_tuning'].get('num_labels', 3)
        self.model = LlamaForSequenceClassification.from_pretrained(
            self.config['model_path'],
            torch_dtype=torch.float16,
            num_labels=self.num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_path'])
        self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

        # Checking fine-tune method
        # Default to use adapter
        assert 'fine_tuning' in self.config, "Need to specify fine_tuning config"
        self.only_train_head = self.config['fine_tuning'].get('only_train_head', False)
        if self.only_train_head:
            print("Only train head")
            # only train the linear head
            self.peft_method = None
            for param in self.model.parameters():
                param.requires_grad = False
            # Then only enable score head training
            for param in self.model.score.parameters():
                param.requires_grad = True
        else:
            self.peft_method = self.config['fine_tuning'].get('method', 'adapter')
            if self.peft_method == 'adapter':
                # using adapter
                print("Using adapter")
                self._init_adapter()
            elif self.peft_method == 'lora':
                # using lora
                print("Using lora")
                self._init_lora(training)
            elif self.peft_method == 'cross_attention':
                # using cross attention
                print("Using cross attention")
                self._init_cross_attention()
            else:
                raise ValueError(f"Unsupported fine-tuning method: {self.peft_method}")
        
        # Adding regression/classification head
        # And set loss function
        self.score = nn.Linear(self.model.config.hidden_size, self.num_labels, bias=True)
        self.task = "regression" if self.num_labels == 1 else "classification"
        if self.task == "classification":
            print("Classification task")
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            print("Regression task")
            self.loss_fn = nn.MSELoss()
    
    def _init_adapter(self):
        """ Add adpaters to base model"""
        # load adapter config
        assert 'adapter_config' in self.config['fine_tuning'], "Need to specify adapter config"
        adapter_config = self.config['fine_tuning']['adapter_config']
        hidden_size = self.model.config.hidden_size
        adapter_size = adapter_config['hidden_size']
        adapter_dropout = adapter_config['adapter_dropout']
        adapter_layers = adapter_config['adapter_layers']

        # add adapter to base model layers specified in 'adapter_layers'
        for i, layer in enumerate(self.model.model.layers):
            if i in adapter_layers:
                adapter = Adapter(hidden_size, adapter_size, adapter_dropout)
                layer.adapter = adapter
        
        # freeze the base model
        for name, param in self.model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def _init_lora(self, training):
        """ Add LoRA to base model"""
        assert 'lora_config' in self.config['fine_tuning'], "Need to specify lora config"
        lora_config = self.config['fine_tuning']['lora_config']
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            inference_mode=not training,
            r=lora_config['r'],
        )
        self.model = get_peft_model(self.model, peft_config)
    
    def _init_cross_attention(self):
        """ Add cross attention to base model"""
        assert 'cross_attention_config' in self.config['fine_tuning'], "Need to specify cross attention config"
        cross_attn_config = self.config['fine_tuning']['cross_attention_config']
        hidden_size = self.model.config.hidden_size
        num_attention_heads = cross_attn_config['num_attention_heads']
        dropout = cross_attn_config['dropout']
        cross_attention_layers = cross_attn_config['cross_attention_layers']

        # 在指定层添加交叉注意力层
        for i, layer in enumerate(self.model.model.layers):
            if i in cross_attention_layers:
                cross_attn = CrossAttentionLayer(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    dropout=dropout
                )
                layer.cross_attention = cross_attn
        
        # 冻结基础模型参数
        for name, param in self.model.named_parameters():
            if 'cross_attention' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

    def forward(self,
                input_ids,
                attention_mask,
                labels=None,
                position_ids=None,
                step_start_idx=None,
                step_end_idx=None,
                **kwargs):
        """
        模型前向传播 - 方案2：将最后一个step单独输入，并与完整输入的特征取平均
        
        Args:
            input_ids (torch.Tensor): 输入ID
            attention_mask (torch.Tensor): 注意力掩码
            labels (torch.Tensor, optional): 标签 (可选)
            position_ids (torch.Tensor, optional): 位置编码 (可选)
            step_start_idx (int, optional): 最后一个 step 的起始索引 (可选)
            step_end_idx (int, optional): 最后一个 step 的结束索引 (可选)
        
        Returns:
            transformers.modeling_outputs.SequenceClassifierOutputWithPast: 模型输出
        """
        if self.peft_method == 'cross_attention':
            # forward with adapter
            
            # check data dtype and device
            device = input_ids.device
            dtype = torch.float16 if self.model.dtype == torch.float16 else torch.float32
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            position_ids = position_ids.to(device) if position_ids is not None else None
            labels = labels.to(device) if labels is not None else None

            # prepare attention mask
            sequence_length = input_ids.shape[1]
            attention_mask = _prepare_4d_causal_attention_mask(
                attention_mask,
                sequence_length,
                dtype,
                device
            )

            # prepare input embeddings
            inputs_embeds = self.model.model.embed_tokens(input_ids)
            hidden_states = inputs_embeds

            # prepare position ids
            if position_ids is None:
                position_ids = torch.arange(
                    sequence_length, 
                    dtype=torch.long, 
                    device=device
                ).unsqueeze(0).expand(input_ids.shape[0], -1)

            # 批量处理last_step特征
            batch_size = input_ids.shape[0]
            last_step_features = []
            max_step_length = max([end_idx - start_idx for start_idx, end_idx in zip(step_start_idx, step_end_idx)])
            
            padded_last_steps = torch.zeros((batch_size, max_step_length), dtype=torch.long, device=device)
            for i in range(batch_size):
                step_length = step_end_idx[i] - step_start_idx[i]
                padded_last_steps[i, :step_length] = input_ids[i, step_start_idx[i]:step_end_idx[i]]
                
            last_step_embeds = self.model.model.embed_tokens(padded_last_steps)
            
            # 通过模型层处理
            for i, layer in enumerate(self.model.model.layers):
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_attentions=False
                )
                hidden_states = layer_outputs[0]
                
                # 批量应用交叉注意力
                if hasattr(layer, 'cross_attention'):
                    hidden_states = layer.cross_attention(hidden_states, last_step_embeds)

            # 最后的层归一化
            hidden_states = self.model.model.norm(hidden_states)

            # 获取最后一个非pad token的特征
            sequence_lengths = torch.eq(input_ids, self.model.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            final_features = hidden_states[torch.arange(batch_size, device=device), sequence_lengths]

            # 通过分类头
            logits = self.model.score(final_features)

            # 计算损失
            loss = None
            if labels is not None:
                if self.task == "classification":
                    loss = self.loss_fn(logits, labels)
                else:
                    loss = self.loss_fn(logits.squeeze(), labels.squeeze())

            return modeling_outputs.SequenceClassifierOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=None,
                hidden_states=hidden_states,
                attentions=None,
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