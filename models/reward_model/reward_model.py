# from transformers import LlamaForCausalLM, LlamaTokenizer
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import LlamaModel
from transformers import modeling_outputs
from peft import get_peft_model, LoraConfig, TaskType
import torch
from torch import nn

class RewardModel(nn.Module):
    def __init__(self,
                 model_path,
                 only_train_head=True,
                 lora_r=8,
                 lora_alpha=32,
                 lora_dropout=0.1,
                 num_labels=1,
                 training=True):
        """
        初始化Generator类
        
        Args:
            model_path (str): Llama 3.1 8B模型的路径
            only_train_head (bool, optional): If True, only train the regression head. If False, also train LoRA parameters. Defaults to True.
            lora_r (int, optional): LoRA的r值。 Defaults to 8.
            lora_alpha (int, optional): LoRA的alpha值。 Defaults to 32.
            lora_dropout (float, optional): LoRA的dropout值。 Defaults to 0.1.
            num_labels (int, optional): if set to 1, it's actually a regression task. If you want to model it as a classification task, set it to the number of classes. Defaults to 1.
            training (bool, optional): 是否处于训练模式。 Defaults to True.
        """
        super().__init__()
        self.model = LlamaModel.from_pretrained(model_path, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.tokenizer.pad_token = "<|reserved_special_token_0|>"

        # # 设置填充标记
        # if self.tokenizer.pad_token is None:
        #     self.tokenizer.pad_token = self.tokenizer.eos_token

        self.only_train_head = only_train_head
        self.num_labels = num_labels
        # Train LoRA and regression head
        if not self.only_train_head:
            # 配置LoRA
            peft_config = LoraConfig(
                task_type=TaskType.FEATURE_EXTRACTION,
                inference_mode=not training,
                r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout
            )
            self.model = get_peft_model(self.model, peft_config)
        # Only train regression head
        # freeze the base model
        else:
            for param in self.model.parameters():
                param.requires_grad = False

        self.score = nn.Linear(self.model.config.hidden_size, num_labels, bias=True)
        self.task = "regression" if num_labels == 1 else "classification"
        if self.task == "classification":
            self.loss_fn = nn.CrossEntropyLoss()
        else:
            self.loss_fn = nn.MSELoss()
    

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

    def forward(self,
                input_ids,
                attention_mask,
                labels=None,
                **kwargs):
        """
        模型前向传播
        
        Args:
            input_ids (torch.Tensor): 输入ID
            attention_mask (torch.Tensor): 注意力掩码
            labels (torch.Tensor, optional): 标签
        
        Returns:
            transformers.modeling_outputs.SequenceClassifierOutputWithPast: 模型输出
        """
        transformer_outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = transformer_outputs[0]
        # get the last non-padding token in hidden states
        batch_size = input_ids.shape[0]
        assert self.tokenizer.pad_token is not None, "The tokenizer does not have a padding token"
        sequence_lengths = torch.eq(input_ids, self.tokenizer.pad_token_id).int().argmax(-1) - 1
        pooled_hidden_states = hidden_states[torch.arange(batch_size, device=input_ids.device), sequence_lengths]
        scores = self.score(pooled_hidden_states)
        # compute loss
        assert labels is not None, "labels should not be None in training"
        if self.task == "regression":
            loss = self.loss_fn(scores.squeeze(), labels.squeeze())
        elif self.task == "classification":
            # loss = self.loss_fn(scores.view(-1, self.num_labels), labels.view(-1))
            loss = self.loss_fn(scores.view(-1, self.num_labels).float(), labels.view(-1).long())

        return modeling_outputs.SequenceClassifierOutputWithPast(
            loss=loss,
            logits=scores,
            past_key_values=transformer_outputs.past_key_values,
            hidden_states=transformer_outputs.hidden_states,
            attentions=transformer_outputs.attentions,
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
