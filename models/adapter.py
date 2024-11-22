import torch
from torch import nn

class Adapter(nn.Module):
    def __init__(self, hidden_size, adapter_size, adapter_dropout=0.1, dtype=None):
        super().__init__()
        self.down_project = nn.Linear(hidden_size, adapter_size)
        self.up_project = nn.Linear(adapter_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(adapter_dropout)
        self.activation = nn.GELU()

        # 如果指定了dtype,则转换参数类型
        if dtype is not None:
            self.to(dtype)

    def forward(self, hidden_states):
        # 使用输入的数据类型
        dtype = hidden_states.dtype
        device = hidden_states.device
        
        residual = hidden_states
        hidden_states = self.layer_norm(hidden_states.to(dtype))
        hidden_states = self.down_project(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.up_project(hidden_states)
        
        return (hidden_states + residual).to(dtype)
