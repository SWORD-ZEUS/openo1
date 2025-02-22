from typing import Optional, Tuple, Union
import torch
from torch import nn, Tensor
from transformers import AutoTokenizer, LlamaModel,  LlamaForCausalLM, AutoModelForCausalLM, LlamaForSequenceClassification, LlamaPreTrainedModel, GenerationMixin
from transformers.modeling_outputs import SequenceClassifierOutputWithPast, CausalLMOutputWithPast
from peft import get_peft_model, LoraConfig, TaskType

# debug
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# debug

from models.adapter import Adapter
from transformers import LlamaConfig, GenerationConfig
import types

class Verifier(nn.Module):
    def __init__(self,
                 config: dict,
                 training: Optional[str] = None):
        """
        初始化Verifier
        
        Args:
            config: 配置字典,包含model_path和fine_tuning等配置
            training: 是否为训练模式
        """
        super().__init__()
        self.config = config
        self._init_base_model()
        self._setup_fine_tuning(training)
        self._setup_task(training)

    def _init_base_model(self):
        """初始化基础模型和分词器"""
        assert 'model_path' in self.config, "model_path is required"
        self.num_labels = self.config['fine_tuning'].get('num_labels', 3)
        
        # 初始化为CausalLM而不是基础LlamaModel
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['model_path'],
            torch_dtype=torch.float16,
        )
        
        # 初始化分词器
        self.tokenizer = AutoTokenizer.from_pretrained(self.config['model_path'])
        self.tokenizer.pad_token = "<|reserved_special_token_0|>"
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        
        # 只需要初始化分类头和回归头
        # 1. 分类头部用于验证答案步骤 
        self.classification = nn.Linear(self.model.config.hidden_size, self.num_labels)
        
        # 2. 回归头部用于预测胜率
        self.win_rate = nn.Linear(self.model.config.hidden_size, 1)

    def _setup_fine_tuning(self, training: Optional[str] = None):
        """
        设置训练方法
        training: 训练模式，None为非训练模式，'verifier'为训语言头和分类头，'predictor'为训练胜率头
        """
        # # 无论训不训练，训练哪种功能，语言头都是冻住的
        # for param in self.response.parameters():
        #         param.requires_grad = False

        ##### 如果training=None，非训练模式。冻住整个个模型
        if training is None:
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.classification.parameters():
                param.requires_grad = False
            for param in self.win_rate.parameters():
                param.requires_grad = False

        ##### 否则即为训练模式，设置训练方法
        else:
            # 先确定训练哪个头
            if training == 'verifier':
                for param in self.win_rate.parameters():
                    param.requires_grad = False
            elif training == 'predictor':
                for param in self.classification.parameters():
                    param.requires_grad = False
            else:
                raise ValueError(f"Unsupported training requirement: {training}")
        
        # 然后设置peft方法
        assert 'fine_tuning' in self.config, "fine_tuning config required"
        self.only_train_head = self.config['fine_tuning'].get('only_train_head', False)
        # Option 1: 只训练头部/不用PEFT
        if self.only_train_head:
            self._setup_head_only_training()
            self.peft_method = None
            return

        self.peft_method = self.config['fine_tuning'].get('method', 'adapter')
        if self.peft_method == 'adapter':
            print("using adapter")
            self._init_adapter()
        elif self.peft_method == 'lora':
            print("using lora")
            self._init_lora(training)
        else:
            raise ValueError(f"Unsupported fine-tuning method: {self.peft_method}")

    def _setup_head_only_training(self):
        """配置只训练头部的设置"""
        print("Only training head")
        self.peft_method = None
        for param in self.model.parameters():
            param.requires_grad = False
        # for param in self.model.score.parameters():
        #     param.requires_grad = True

    def _setup_task(self, training: Optional[str] = None):
        """设置任务类型和损失函数"""
        self.task = training
        if self.task is None:
            self.loss_fn = None
            print("Not in training mode, no task specified")
            return
        elif self.task == 'verifier':
            # 对于verifier任务,response和classification头的训练都使用CrossEntropyLoss
            self.loss_fn = nn.CrossEntropyLoss()
        elif self.task == 'predictor':
            # 对于predictor任务,使用回归loss
            self.loss_fn = nn.MSELoss()
        else:
            raise ValueError(f"Unsupported task: {self.task}")

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
        # logits = self.model.score(hidden_states)
        # return logits[torch.arange(batch_size, device=logits.device), sequence_lengths]
        return hidden_states[torch.arange(batch_size, device=hidden_states.device), sequence_lengths]

    def _init_adapter(self):
        """Add adapters to base model and register hooks"""
        assert 'adapter_config' in self.config['fine_tuning'], "Need to specify adapter config"
        adapter_config = self.config['fine_tuning']['adapter_config']
        hidden_size = self.model.config.hidden_size
        adapter_size = adapter_config['hidden_size']
        adapter_dropout = adapter_config['adapter_dropout']
        adapter_layers = adapter_config['adapter_layers']

        # 添加adapter到指定层
        for i, layer in enumerate(self.model.model.layers):
            if i in adapter_layers:
                # adapter = Adapter(hidden_size, adapter_size, adapter_dropout)
                adapter = Adapter(hidden_size, adapter_size, adapter_dropout, dtype=self.model.dtype)
                layer.adapter = adapter
        
        # freeze基础模型参数
        for name, param in self.model.named_parameters():
            if 'adapter' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
            
        # 绑定自定义前向传播
        self.model = bind_forward_for_verifier_causalLM(self)

    def _init_lora(self, training):
        """ Add LoRA to base model"""
        assert 'lora_config' in self.config['fine_tuning'], "Need to specify lora config"
        is_test = self.config['is_test']
        lora_config = self.config['fine_tuning']['lora_config']
        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=is_test,
            lora_alpha=lora_config['alpha'],
            r=lora_config['r'],
            lora_dropout=lora_config['dropout'],
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
            labels (to be decided): 标签 (待定)
            position_ids (torch.Tensor, optional): 位置编码 (可选)
            step_start_idx (int, optional): 最后一个 step 的起始索引 (可选)
            step_end_idx (int, optional): 最后一个 step 的结束索引 (可选)
        
        Returns:
            transformers.modeling_outputs.SequenceClassifierOutputWithPast: 模型输出
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_hidden_states=True  # 确保输出 hidden states
        )
        
        # 获取最后一层的 hidden states
        if self.peft_method == 'lora':
            hidden_states = outputs.hidden_states[-1]
        elif self.peft_method == 'adapter':
            hidden_states = outputs.hidden_states
        
        # 计算loss
        loss = None
        if self.task == 'verifier':
            # 过分类和语言头
            lm_logits = self.model.lm_head(hidden_states)
            cls_logits = self._get_step_features(hidden_states, step_start_idx, step_end_idx, input_ids)
            cls_logits = self.classification(cls_logits)
            logits = (lm_logits, cls_logits)
            # logits = lm_logits
            
            lm_loss = None
            cls_loss = None
            # assert isinstance(labels, dict) and len(labels) == 2, "labels should be a dictionary with 2 elements"
            if labels is not None:
                lm_labels, cls_labels = labels['response'], labels['rating'] # unpack the labels
                shift_lm_logits = lm_logits[..., :-1, :].contiguous()
                shift_lm_labels = lm_labels[..., 1:].contiguous()
                lm_loss = self.loss_fn(shift_lm_logits.view(-1, shift_lm_logits.size(-1)), shift_lm_labels.view(-1))
                cls_loss = self.loss_fn(cls_logits, cls_labels)
                loss = lm_loss + cls_loss
            else:
                loss = None
        elif self.task == 'predictor':
            # 过回归头
            logits = self._get_step_features(hidden_states, step_start_idx, step_end_idx, input_ids)
            logits = self.win_rate(logits)

            assert labels is not None, "labels are required for predictor task"
            loss = self.loss_fn(logits.squeeze(), labels.squeeze())
        else:
            raise ValueError(f"Unsupported task: {self.task}")

        if cls_loss is None or lm_loss is None:
            print(f"labels: {labels}")
            print(f"step_start_idx: {step_start_idx}")
            print(f"step_end_idx: {step_end_idx}")

        return_dict = {
            'loss': {
            'total_loss': loss,
            'lm_loss': lm_loss if lm_loss is not None else None,
            'cls_loss': cls_loss if cls_loss is not None else None
            },
            'logits': logits,
            'hidden_states': hidden_states,
            'attentions': None,
            'past_key_values': None
        }


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
    
def bind_forward_for_verifier_causalLM(self):
    """为 verifier 的 CausalLM 绑定自定义前向传播方法"""
    def custom_forward(
        model_self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        inputs_embeds=None,
        use_cache=None, 
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        past_key_values=None,
        cache_position=None,  # 添加生成时需要的参数
        **kwargs  # 添加kwargs以接收其他可能的参数
    ):

        batch_size, sequence_length = input_ids.shape[:2] 
 
        inputs_embeds = model_self.model.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # 3. 获取位置编码
        if position_ids is None:
            device = input_ids.device if input_ids is not None else inputs_embeds.device
            position_ids = torch.arange(
                sequence_length, 
                dtype=torch.long,
                device=device
            ).unsqueeze(0).expand(batch_size, -1)

        # 4. 准备注意力掩码
        dtype = torch.float16 if model_self.dtype == torch.float16 else torch.float32
        attention_mask = _prepare_4d_causal_attention_mask(
            attention_mask,
            sequence_length,
            dtype,
            hidden_states.device
        )

        # 5. 通过每一层并应用 adapter
        for i, layer in enumerate(model_self.model.layers):
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                output_attentions=output_attentions,
                use_cache=use_cache
            )
            hidden_states = layer_outputs[0]
            
            # 应用 adapter(如果存在)
            if hasattr(layer, 'adapter'):
                hidden_states = layer.adapter(hidden_states)

        # 6. 最后的 Layer Norm
        hidden_states = model_self.model.norm(hidden_states)

        # 7. 语言模型头
        logits = model_self.lm_head(hidden_states)

        # 8. 组装输出
        return CausalLMOutputWithPast(
            logits=logits,
            hidden_states=hidden_states,
            attentions=None,
            past_key_values=None
        )
    
    # 使用 types.MethodType 动态绑定方法
    self.model.forward = types.MethodType(custom_forward, self.model)

    return self.model

if __name__ == "__main__":
    config = {
        'model_path': '/storage/zhangyingqi/weights_n_config/Meta-Llama-3.1-8B-Instruct',
        'fine_tuning': {
            'num_labels': 3,
            'only_train_head': True,
        }
    }
    verifier = Verifier(config, training='verifier')

    batch_size = 2
    seq_length = 32
    input_ids = torch.randint(0, 30000, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    lm_labels = torch.randint(0, 30000, (batch_size, seq_length))
    cls_labels = torch.randint(0, 3, (batch_size,))
    labels = (lm_labels, cls_labels)
    step_start_idx = torch.tensor([10, 15])
    step_end_idx = torch.tensor([20, 25])

    outputs = verifier(input_ids, attention_mask, labels, step_start_idx, step_end_idx)
    assert outputs.loss is not None, "Loss should not be None in training mode"
    lm_logits, cls_logits = outputs.logits
    assert lm_logits.shape == (batch_size, seq_length, verifier.model.config.vocab_size)
    assert cls_logits.shape == (batch_size, verifier.num_labels)

    print("Basic verifier test passed!")