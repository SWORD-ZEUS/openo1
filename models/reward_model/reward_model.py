# from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoTokenizer, LlamaForSequenceClassification
from transformers import modeling_outputs
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch import nn
from models.adapter import Adapter

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
        if self.peft_method == 'adapter':
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
            
            # forward through the model
            for i, layer in enumerate(self.model.model.layers):
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    use_cache=False,
                    output_attentions=False
                )
                hidden_states = layer_outputs[0]
                if hasattr(layer, 'adapter'):
                    hidden_states = layer.adapter(hidden_states)
            hidden_states = self.model.model.norm(hidden_states)

            # 获取完整输入的最后一个token特征
            sequence_lengths = torch.eq(input_ids, self.model.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            batch_size = input_ids.shape[0]
            full_sequence_features = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

            # 获取最后一个step的输入
            last_step_features = []
            for i in range(batch_size):
                # 检查step长度是否有效
                step_length = step_end_idx[i] - step_start_idx[i]
                if step_length <= 0:
                    # 如果step长度无效，使用完整序列的最后一个token特征
                    last_step_features.append(full_sequence_features[i])
                    continue

                # 提取最后一个step的文本
                last_step_ids = input_ids[i, step_start_idx[i]:step_end_idx[i]]
                last_step_mask = attention_mask[i, step_start_idx[i]:step_end_idx[i]]
                
                # 对最后一个step单独进行编码
                last_step_hidden = self.model.model.embed_tokens(last_step_ids)
                last_step_attention = _prepare_4d_causal_attention_mask(
                    last_step_mask.unsqueeze(0),
                    last_step_ids.shape[0],
                    dtype=hidden_states.dtype,
                    device=hidden_states.device
                )

                # 通过模型层处理最后一个step
                step_hidden = last_step_hidden.unsqueeze(0)  # 添加batch维度
                for layer in self.model.model.layers:
                    layer_outputs = layer(
                        step_hidden,
                        attention_mask=last_step_attention,
                        position_ids=None,
                        use_cache=False,
                        output_attentions=False
                    )
                    step_hidden = layer_outputs[0]
                    if hasattr(layer, 'adapter'):
                        step_hidden = layer.adapter(step_hidden)
                
                step_hidden = self.model.model.norm(step_hidden)
                # 取最后一个token的特征
                last_step_feature = step_hidden[0, -1]
                last_step_features.append(last_step_feature)

            last_step_features = torch.stack(last_step_features)
            
            # 将完整序列特征和最后一个step特征取平均
            combined_features = (full_sequence_features + last_step_features) / 2
            logits = self.model.score(combined_features)

        else:
            # forward with LoRA or only train head
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True  # 确保输出 hidden states
            )
            
            # 获取最后一层的 hidden states
            hidden_states = outputs.hidden_states[-1]
            
            # 获取完整输入的最后一个token特征
            sequence_lengths = torch.eq(input_ids, self.model.config.pad_token_id).int().argmax(-1) - 1
            sequence_lengths = sequence_lengths % input_ids.shape[-1]
            batch_size = input_ids.shape[0]
            full_sequence_features = hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

            # 获取最后一个step的特征
            last_step_features = []
            for i in range(batch_size):
                # 检查step长度是否有效
                step_length = step_end_idx[i] - step_start_idx[i]
                if step_length <= 0:
                    # 如果step长度无效，使用完整序列的最后一个token特征
                    last_step_features.append(full_sequence_features[i])
                    continue
                
                # 提取最后一个step的输入
                last_step_ids = input_ids[i, step_start_idx[i]:step_end_idx[i]].unsqueeze(0)
                last_step_mask = attention_mask[i, step_start_idx[i]:step_end_idx[i]].unsqueeze(0)
                
                # 对最后一个step单独进行前向传播
                step_outputs = self.model.model(
                    input_ids=last_step_ids,
                    attention_mask=last_step_mask,
                    output_hidden_states=True
                )
                step_hidden = step_outputs.hidden_states[-1]
                last_step_features.append(step_hidden[0, -1])

            last_step_features = torch.stack(last_step_features)
            
            # 将完整序列特征和最后一个step特征取平均
            combined_features = (full_sequence_features + last_step_features) / 2
            logits = self.model.score(combined_features)

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