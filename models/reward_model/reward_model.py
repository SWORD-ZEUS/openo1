from typing import Optional, Tuple, Union
import torch
from torch import nn, Tensor
from transformers import AutoTokenizer, LlamaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from peft import get_peft_model, LoraConfig, TaskType
from models.adapter import Adapter

class RewardModel(nn.Module):
    def __init__(self,
                 config: dict,
                 training: bool = True):
        """
        初始化RewardModel
        
        Args:
            config: 配置字典,包含model_path和fine_tuning等配置
            training: 是否为训练模式
        """
        super().__init__()
        self.config = config
        self._init_base_model()
        self._setup_fine_tuning(training)
        self._setup_task()

    def _init_base_model(self):
        """初始化基础模型和分词器"""
        assert 'model_path' in self.config, "model_path is required"
        self.num_labels = self.config['fine_tuning'].get('num_labels', 2)
        
        self.model = LlamaForSequenceClassification.from_pretrained(
            self.config['model_path'],
            torch_dtype=torch.float16,
            num_labels=self.num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_path'])
        self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        self.model.config.pad_token_id = self.tokenizer.pad_token_id

    def _setup_fine_tuning(self, training: bool):
        """设置微调方法"""
        assert 'fine_tuning' in self.config, "fine_tuning config required"
        
        self.only_train_head = self.config['fine_tuning'].get('only_train_head', False)
        if self.only_train_head:
            self._setup_head_only_training()
            return

        self.peft_method = self.config['fine_tuning'].get('method', 'adapter')
        if self.peft_method == 'adapter':
            self._init_adapter()
        elif self.peft_method == 'lora':
            self._init_lora(training)
        else:
            raise ValueError(f"Unsupported fine-tuning method: {self.peft_method}")

    def _setup_head_only_training(self):
        """配置只训练头部的设置"""
        print("Only training head")
        self.peft_method = None
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.score.parameters():
            param.requires_grad = True

    def _setup_task(self):
        """设置任务类型和损失函数"""
        self.score = nn.Linear(self.model.config.hidden_size, self.num_labels)
        self.task = "regression" if self.num_labels == 1 else "classification"
        self.loss_fn = nn.MSELoss() if self.task == "regression" else nn.CrossEntropyLoss()
        print(f"{self.task.capitalize()} task")

    def _get_step_features(self, hidden_states: Tensor, 
                          step_start_idx: Optional[Tensor] = None,
                          step_end_idx: Optional[Tensor] = None,
                          input_ids: Optional[Tensor] = None) -> Tensor:
        """获取最后一个step的特征向量
        
        Args:
            hidden_states: 隐藏状态
            step_start_idx: step开始索引
            step_end_idx: step结束索引 
            input_ids: 输入ID
        """
        if step_start_idx is not None and step_end_idx is not None:
            return self._get_step_features_with_indices(hidden_states, step_start_idx, step_end_idx)
        return self._get_step_features_default(hidden_states, input_ids)

    def _get_step_features_with_indices(self, hidden_states: Tensor,
                                      step_start_idx: Tensor,
                                      step_end_idx: Tensor) -> Tensor:
        """使用给定索引获取step特征"""
        step_hidden_states = []
        for i in range(hidden_states.shape[0]):
            step_length = step_end_idx[i] - step_start_idx[i]
            if step_length <= 0:
                step_hidden_states.append(hidden_states[i, -1, :])
            else:
                step_hidden_states.append(hidden_states[i, step_start_idx[i]:step_end_idx[i], :].mean(dim=0))
        step_hidden_states = torch.stack(step_hidden_states)
        # 添加这一行,通过score层
        return self.model.score(step_hidden_states)

    def _get_step_features_default(self, hidden_states: Tensor, input_ids: Tensor) -> Tensor:
        """使用默认方式获取step特征"""
        sequence_lengths = torch.eq(input_ids, self.model.config.pad_token_id).int().argmax(-1) - 1
        sequence_lengths = sequence_lengths % input_ids.shape[-1]
        batch_size = input_ids.shape[0]
        logits = self.model.score(hidden_states)
        return logits[torch.arange(batch_size, device=logits.device), sequence_lengths]

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
        模型前向传播
        
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

            # 获取最后一个 step 的特征向量
            logits = self._get_step_features(hidden_states, step_start_idx, step_end_idx, input_ids)

            # compute loss if labels provided
            loss = None
            if labels is not None:
                if self.task == "classification":
                    loss = self.loss_fn(logits, labels)
                else:
                    loss = self.loss_fn(logits.squeeze(), labels.squeeze())

            return_dict = SequenceClassifierOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=None,
                hidden_states=hidden_states,
                attentions=None,
            )
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
            
            # 获取最后一个 step 的特征向量
            logits = self._get_step_features(hidden_states, step_start_idx, step_end_idx, input_ids)

            # 计算损失
            loss = None
            if labels is not None:
                if self.task == "classification":
                    loss = self.loss_fn(logits, labels)
                else:
                    loss = self.loss_fn(logits.squeeze(), labels.squeeze())

            return_dict = SequenceClassifierOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions,
            )

        return return_dict

    
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