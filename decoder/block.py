import torch
from torch import nn
from multi_head_attention import MultiHeadAttention

class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(DecoderBlock, self).__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.ln2 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadAttention(embed_dim, num_heads)
        self.relu = nn.ReLU()
        
    def forward(self, x, mask=None):
        x = self.ln1(x + self.attn(x, mask))
        x = self.ln2(x + self.relu(x))
        return x