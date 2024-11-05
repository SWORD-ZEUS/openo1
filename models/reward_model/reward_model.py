from typing import Optional, Tuple, Union
import torch
from torch import nn, Tensor
from transformers import AutoTokenizer, LlamaForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutputWithPast
from peft import get_peft_model, LoraConfig, TaskType
from models.adapter import Adapter

class RewardModel(nn.Module):
    def __init__(self, config: dict, training: bool = True):
        """初始化奖励模型
        
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
        # 设置分类损失和排序损失
        self.cls_loss_fn = nn.CrossEntropyLoss() if self.task == "classification" else nn.MSELoss()
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
        return torch.stack(step_hidden_states)

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

    def compute_loss(self, correct_logits, wrong_logits, correct_labels, wrong_labels):
        """计算综合损失：分类损失 + 排序损失(logistics loss)
        
        Args:
            correct_logits: 正确样本的预测分数
            wrong_logits: 错误样本的预测分数
            correct_labels: 正确样本的标签
            wrong_labels: 错误样本的标签
        """
        # 计算分类损失
        cls_loss = self.cls_loss_fn(torch.cat([correct_logits, wrong_logits], dim=0),
                                  torch.cat([correct_labels, wrong_labels], dim=0))
        
        # 计算排序损失 (correct应该比wrong得分高)
        if self.task == "classification":
            # 对于分类任务，使用softmax后的正类概率作为排序分数
            correct_scores = torch.softmax(correct_logits, dim=-1)[:, 1]  # 取正类概率
            wrong_scores = torch.softmax(wrong_logits, dim=-1)[:, 1]
        else:
            correct_scores = correct_logits.squeeze()
            wrong_scores = wrong_logits.squeeze()
        
        # 计算logistics ranking loss
        diff = correct_scores - wrong_scores
        ranking_loss = torch.mean(torch.log(1 + torch.exp(-diff)))
        
        # 总损失 = 分类损失 + λ * 排序损失
        lambda_weight = 1.0  # 可以通过配置文件调整权重
        total_loss = cls_loss + lambda_weight * ranking_loss
        
        return {
            'loss': total_loss,
            'cls_loss': cls_loss,
            'ranking_loss': ranking_loss
        }

    def _process_sample(self, input_ids, attention_mask, position_ids=None):
        """处理单个样本的前向传播
        
        Args:
            input_ids: 输入ID
            attention_mask: 注意力掩码
            position_ids: 位置编码 (可选)
        """
        device = input_ids.device
        dtype = torch.float16 if self.model.dtype == torch.float16 else torch.float32
        
        # 准备注意力掩码
        sequence_length = input_ids.shape[1]
        attention_mask_4d = _prepare_4d_causal_attention_mask(
            attention_mask,
            sequence_length,
            dtype,
            device
        )
        
        # 准备position ids
        if position_ids is None:
            position_ids = torch.arange(
                sequence_length,
                dtype=torch.long,
                device=device
            ).unsqueeze(0).expand(input_ids.shape[0], -1)
        
        if self.peft_method == 'adapter':
            # 前向传播
            hidden_states = self.model.model.embed_tokens(input_ids)
            
            # 通过模型层
            for i, layer in enumerate(self.model.model.layers):
                layer_outputs = layer(
                    hidden_states,
                    attention_mask=attention_mask_4d,
                    position_ids=position_ids,
                    use_cache=False,
                    output_attentions=False
                )
                hidden_states = layer_outputs[0]
                if hasattr(layer, 'adapter'):
                    hidden_states = layer.adapter(hidden_states)
            hidden_states = self.model.model.norm(hidden_states)
        else:
            # LoRA或只训练头部的情况
            outputs = self.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_hidden_states=True
            )
            hidden_states = outputs.hidden_states[-1]
        
        return hidden_states

    def forward(self,
                correct_input_ids,
                correct_attention_mask,
                wrong_input_ids,
                wrong_attention_mask,
                correct_labels=None,
                wrong_labels=None,
                position_ids=None,
                correct_step_start_idx=None,
                correct_step_end_idx=None,
                wrong_step_start_idx=None,
                wrong_step_end_idx=None,
                **kwargs):
        """前向传播，分别处理正确和错误样本"""
        
        # 处理正确和错误样本
        correct_hidden_states = self._process_sample(correct_input_ids, correct_attention_mask, position_ids)
        wrong_hidden_states = self._process_sample(wrong_input_ids, wrong_attention_mask, position_ids)
        
        # 获取特征向量
        correct_logits = self._get_step_features(
            correct_hidden_states, 
            correct_step_start_idx, 
            correct_step_end_idx, 
            correct_input_ids
        )
        wrong_logits = self._get_step_features(
            wrong_hidden_states, 
            wrong_step_start_idx, 
            wrong_step_end_idx, 
            wrong_input_ids
        )
        
        # 计算损失
        loss_dict = None
        if correct_labels is not None and wrong_labels is not None:
            loss_dict = self.compute_loss(
                correct_logits, 
                wrong_logits,
                correct_labels, 
                wrong_labels
            )

        return SequenceClassifierOutputWithPast(
            loss=loss_dict['loss'] if loss_dict else None,
            logits={
                'correct_logits': correct_logits,
                'wrong_logits': wrong_logits,
                'cls_loss': loss_dict['cls_loss'] if loss_dict else None,
                'ranking_loss': loss_dict['ranking_loss'] if loss_dict else None
            },
            past_key_values=None,
            hidden_states={
                'correct': correct_hidden_states,
                'wrong': wrong_hidden_states
            },
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