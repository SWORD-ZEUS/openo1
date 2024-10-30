import torch
from torch import nn
import math
import torch.nn.functional as F

class CrossAttentionLayer(nn.Module):
    def __init__(self, hidden_size, num_attention_heads=8, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        self.scaling = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_size, hidden_size)
        self.k_proj = nn.Linear(hidden_size, hidden_size)
        self.v_proj = nn.Linear(hidden_size, hidden_size)
        self.o_proj = nn.Linear(hidden_size, hidden_size)
        
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, y):
        """
        x: [batch_size, seq_len_x, hidden_size] - 完整输入的特征
        y: [batch_size, seq_len_y, hidden_size] - 最后一个step的特征
        """
        residual = x
        x = self.norm1(x)
        y = self.norm2(y)
        
        batch_size = x.shape[0]
        
        # 多头注意力投影
        q = self.q_proj(x).view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(y).view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(y).view(batch_size, -1, self.num_attention_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力分数
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output + residual 